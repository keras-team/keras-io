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
from tensorflow import keras
from tensorflow.keras import layers
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
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
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

/Users/vrindaprabhu/.pyenv/versions/3.7.8/envs/keras-env/lib/python3.7/site-packages/ipykernel_launcher.py:2: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  
/Users/vrindaprabhu/.pyenv/versions/3.7.8/envs/keras-env/lib/python3.7/site-packages/ipykernel_launcher.py:3: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  This is separate from the ipykernel package so we can avoid doing imports until

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
model.add(layers.LSTM(128, input_shape=(MAXLEN, len(chars))))
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

2022-06-13 20:07:13.074867: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 128)               72192     
                                                                 
 repeat_vector (RepeatVector  (None, 4, 128)           0         
 )                                                               
                                                                 
 lstm_1 (LSTM)               (None, 4, 128)            131584    
                                                                 
 dense (Dense)               (None, 4, 12)             1548      
                                                                 
=================================================================
Total params: 205,324
Trainable params: 205,324
Non-trainable params: 0
_________________________________________________________________

```
</div>
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
1407/1407 [==============================] - 45s 28ms/step - loss: 1.7565 - accuracy: 0.3568 - val_loss: 1.5590 - val_accuracy: 0.4144
1/1 [==============================] - 9s 9s/step
Q 296+37  T 333  ☒ 366 
1/1 [==============================] - 0s 184ms/step
Q 428+58  T 486  ☒ 496 
1/1 [==============================] - 0s 182ms/step
Q 462+307 T 769  ☒ 800 
1/1 [==============================] - 0s 182ms/step
Q 119+65  T 184  ☒ 266 
1/1 [==============================] - 0s 183ms/step
Q 37+863  T 900  ☒ 736 
1/1 [==============================] - 0s 182ms/step
Q 46+1    T 47   ☒ 1   
1/1 [==============================] - 0s 183ms/step
Q 29+834  T 863  ☒ 340 
1/1 [==============================] - 0s 181ms/step
Q 88+61   T 149  ☒ 174 
1/1 [==============================] - 0s 183ms/step
Q 74+72   T 146  ☒ 770 
1/1 [==============================] - 0s 172ms/step
Q 88+61   T 149  ☒ 174 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 2
1407/1407 [==============================] - 128s 91ms/step - loss: 1.3471 - accuracy: 0.4968 - val_loss: 1.1716 - val_accuracy: 0.5587
1/1 [==============================] - 0s 70ms/step
Q 38+794  T 832  ☒ 818 
1/1 [==============================] - 0s 79ms/step
Q 37+863  T 900  ☒ 818 
1/1 [==============================] - 0s 78ms/step
Q 71+715  T 786  ☒ 778 
1/1 [==============================] - 0s 78ms/step
Q 249+3   T 252  ☒ 268 
1/1 [==============================] - 0s 79ms/step
Q 240+274 T 514  ☒ 518 
1/1 [==============================] - 0s 79ms/step
Q 48+84   T 132  ☒ 128 
1/1 [==============================] - 0s 70ms/step
Q 19+462  T 481  ☒ 478 
1/1 [==============================] - 0s 78ms/step
Q 8+837   T 845  ☒ 838 
1/1 [==============================] - 0s 77ms/step
Q 694+1   T 695  ☒ 699 
1/1 [==============================] - 0s 71ms/step
Q 53+547  T 600  ☒ 508 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 3
1407/1407 [==============================] - 57s 41ms/step - loss: 1.0337 - accuracy: 0.6178 - val_loss: 0.9271 - val_accuracy: 0.6608
1/1 [==============================] - 0s 69ms/step
Q 68+760  T 828  ☒ 824 
1/1 [==============================] - 0s 70ms/step
Q 35+408  T 443  ☒ 441 
1/1 [==============================] - 0s 79ms/step
Q 11+313  T 324  ☑ 324 
1/1 [==============================] - 0s 70ms/step
Q 885+83  T 968  ☒ 964 
1/1 [==============================] - 0s 72ms/step
Q 344+22  T 366  ☒ 365 
1/1 [==============================] - 0s 77ms/step
Q 97+26   T 123  ☒ 124 
1/1 [==============================] - 0s 78ms/step
Q 390+22  T 412  ☒ 414 
1/1 [==============================] - 0s 70ms/step
Q 299+9   T 308  ☒ 304 
1/1 [==============================] - 0s 70ms/step
Q 664+50  T 714  ☒ 715 
1/1 [==============================] - 0s 78ms/step
Q 133+86  T 219  ☒ 214 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 4
1407/1407 [==============================] - 103s 73ms/step - loss: 0.8580 - accuracy: 0.6878 - val_loss: 0.7967 - val_accuracy: 0.7047
1/1 [==============================] - 0s 63ms/step
Q 405+72  T 477  ☒ 478 
1/1 [==============================] - 0s 64ms/step
Q 45+427  T 472  ☒ 478 
1/1 [==============================] - 0s 68ms/step
Q 935+986 T 1921 ☒ 1809
1/1 [==============================] - 0s 62ms/step
Q 785+555 T 1340 ☒ 1333
1/1 [==============================] - 0s 63ms/step
Q 3+793   T 796  ☒ 795 
1/1 [==============================] - 0s 63ms/step
Q 494+966 T 1460 ☒ 1353
1/1 [==============================] - 0s 63ms/step
Q 2+646   T 648  ☒ 647 
1/1 [==============================] - 0s 62ms/step
Q 723+302 T 1025 ☒ 1032
1/1 [==============================] - 0s 61ms/step
Q 374+1   T 375  ☒ 373 
1/1 [==============================] - 0s 62ms/step
Q 846+361 T 1207 ☒ 1100
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 5
1407/1407 [==============================] - 51s 36ms/step - loss: 0.7526 - accuracy: 0.7281 - val_loss: 0.7243 - val_accuracy: 0.7346
1/1 [==============================] - 0s 62ms/step
Q 277+631 T 908  ☒ 913 
1/1 [==============================] - 0s 64ms/step
Q 89+221  T 310  ☒ 312 
1/1 [==============================] - 0s 62ms/step
Q 202+73  T 275  ☒ 271 
1/1 [==============================] - 0s 63ms/step
Q 106+781 T 887  ☒ 888 
1/1 [==============================] - 0s 64ms/step
Q 446+85  T 531  ☒ 532 
1/1 [==============================] - 0s 63ms/step
Q 57+89   T 146  ☒ 142 
1/1 [==============================] - 0s 63ms/step
Q 618+78  T 696  ☒ 697 
1/1 [==============================] - 0s 63ms/step
Q 710+378 T 1088 ☒ 1073
1/1 [==============================] - 0s 61ms/step
Q 42+497  T 539  ☒ 540 
1/1 [==============================] - 0s 63ms/step
Q 37+964  T 1001 ☒ 990 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 6
1407/1407 [==============================] - 78s 55ms/step - loss: 0.6643 - accuracy: 0.7602 - val_loss: 0.6047 - val_accuracy: 0.7829
1/1 [==============================] - 0s 110ms/step
Q 555+9   T 564  ☒ 565 
1/1 [==============================] - 0s 109ms/step
Q 562+20  T 582  ☒ 584 
1/1 [==============================] - 0s 111ms/step
Q 32+57   T 89   ☑ 89  
1/1 [==============================] - 0s 108ms/step
Q 8+109   T 117  ☒ 115 
1/1 [==============================] - 0s 109ms/step
Q 971+460 T 1431 ☒ 1439
1/1 [==============================] - 0s 108ms/step
Q 69+98   T 167  ☑ 167 
1/1 [==============================] - 0s 95ms/step
Q 716+2   T 718  ☑ 718 
1/1 [==============================] - 0s 109ms/step
Q 432+42  T 474  ☑ 474 
1/1 [==============================] - 0s 108ms/step
Q 845+553 T 1398 ☒ 1399
1/1 [==============================] - 0s 110ms/step
Q 8+303   T 311  ☒ 310 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 7
1407/1407 [==============================] - 66s 47ms/step - loss: 0.4769 - accuracy: 0.8299 - val_loss: 0.3222 - val_accuracy: 0.8991
1/1 [==============================] - 0s 63ms/step
Q 776+8   T 784  ☑ 784 
1/1 [==============================] - 0s 63ms/step
Q 78+664  T 742  ☒ 743 
1/1 [==============================] - 0s 62ms/step
Q 617+22  T 639  ☑ 639 
1/1 [==============================] - 0s 62ms/step
Q 416+31  T 447  ☑ 447 
1/1 [==============================] - 0s 63ms/step
Q 70+218  T 288  ☑ 288 
1/1 [==============================] - 0s 65ms/step
Q 435+8   T 443  ☑ 443 
1/1 [==============================] - 0s 64ms/step
Q 343+367 T 710  ☑ 710 
1/1 [==============================] - 0s 58ms/step
Q 828+72  T 900  ☒ 890 
1/1 [==============================] - 0s 58ms/step
Q 35+32   T 67   ☑ 67  
1/1 [==============================] - 0s 59ms/step
Q 339+46  T 385  ☑ 385 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 8
1407/1407 [==============================] - 50s 35ms/step - loss: 0.2484 - accuracy: 0.9272 - val_loss: 0.1729 - val_accuracy: 0.9559
1/1 [==============================] - 0s 67ms/step
Q 212+80  T 292  ☑ 292 
1/1 [==============================] - 0s 62ms/step
Q 587+8   T 595  ☑ 595 
1/1 [==============================] - 0s 61ms/step
Q 23+666  T 689  ☑ 689 
1/1 [==============================] - 0s 63ms/step
Q 25+749  T 774  ☑ 774 
1/1 [==============================] - 0s 70ms/step
Q 266+61  T 327  ☑ 327 
1/1 [==============================] - 0s 72ms/step
Q 79+96   T 175  ☒ 174 
1/1 [==============================] - 0s 62ms/step
Q 27+83   T 110  ☑ 110 
1/1 [==============================] - 0s 64ms/step
Q 478+385 T 863  ☒ 853 
1/1 [==============================] - 0s 69ms/step
Q 8+77    T 85   ☒ 84  
1/1 [==============================] - 0s 52ms/step
Q 16+15   T 31   ☑ 31  
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 9
1407/1407 [==============================] - 82s 58ms/step - loss: 0.1507 - accuracy: 0.9616 - val_loss: 0.1279 - val_accuracy: 0.9657
1/1 [==============================] - 0s 62ms/step
Q 374+1   T 375  ☑ 375 
1/1 [==============================] - 0s 64ms/step
Q 415+3   T 418  ☑ 418 
1/1 [==============================] - 0s 63ms/step
Q 25+186  T 211  ☑ 211 
1/1 [==============================] - 0s 62ms/step
Q 6+246   T 252  ☑ 252 
1/1 [==============================] - 0s 63ms/step
Q 565+0   T 565  ☑ 565 
1/1 [==============================] - 0s 64ms/step
Q 680+98  T 778  ☑ 778 
1/1 [==============================] - 0s 64ms/step
Q 600+180 T 780  ☒ 781 
1/1 [==============================] - 0s 63ms/step
Q 2+920   T 922  ☑ 922 
1/1 [==============================] - 0s 65ms/step
Q 746+71  T 817  ☑ 817 
1/1 [==============================] - 0s 59ms/step
Q 63+929  T 992  ☒ 991 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 10
1407/1407 [==============================] - 45s 32ms/step - loss: 0.0936 - accuracy: 0.9784 - val_loss: 0.0768 - val_accuracy: 0.9817
1/1 [==============================] - 0s 59ms/step
Q 721+70  T 791  ☑ 791 
1/1 [==============================] - 0s 59ms/step
Q 925+861 T 1786 ☑ 1786
1/1 [==============================] - 0s 59ms/step
Q 63+880  T 943  ☑ 943 
1/1 [==============================] - 0s 58ms/step
Q 59+239  T 298  ☑ 298 
1/1 [==============================] - 0s 57ms/step
Q 420+15  T 435  ☑ 435 
1/1 [==============================] - 0s 59ms/step
Q 360+86  T 446  ☑ 446 
1/1 [==============================] - 0s 61ms/step
Q 650+972 T 1622 ☑ 1622
1/1 [==============================] - 0s 59ms/step
Q 526+273 T 799  ☑ 799 
1/1 [==============================] - 0s 58ms/step
Q 693+576 T 1269 ☑ 1269
1/1 [==============================] - 0s 58ms/step
Q 756+11  T 767  ☑ 767 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 11
1407/1407 [==============================] - 57s 41ms/step - loss: 0.0746 - accuracy: 0.9814 - val_loss: 0.1443 - val_accuracy: 0.9523
1/1 [==============================] - 0s 68ms/step
Q 430+43  T 473  ☑ 473 
1/1 [==============================] - 0s 68ms/step
Q 78+976  T 1054 ☒ 1053
1/1 [==============================] - 0s 69ms/step
Q 773+32  T 805  ☑ 805 
1/1 [==============================] - 0s 67ms/step
Q 207+88  T 295  ☑ 295 
1/1 [==============================] - 0s 72ms/step
Q 931+959 T 1890 ☑ 1890
1/1 [==============================] - 0s 67ms/step
Q 16+15   T 31   ☑ 31  
1/1 [==============================] - 0s 66ms/step
Q 546+340 T 886  ☑ 886 
1/1 [==============================] - 0s 74ms/step
Q 797+10  T 807  ☑ 807 
1/1 [==============================] - 0s 73ms/step
Q 357+106 T 463  ☑ 463 
1/1 [==============================] - 0s 67ms/step
Q 819+3   T 822  ☑ 822 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 12
1407/1407 [==============================] - 73s 52ms/step - loss: 0.0537 - accuracy: 0.9872 - val_loss: 0.0895 - val_accuracy: 0.9696
1/1 [==============================] - 0s 86ms/step
Q 149+372 T 521  ☑ 521 
1/1 [==============================] - 0s 96ms/step
Q 778+3   T 781  ☑ 781 
1/1 [==============================] - 0s 96ms/step
Q 97+26   T 123  ☒ 122 
1/1 [==============================] - 0s 96ms/step
Q 8+973   T 981  ☑ 981 
1/1 [==============================] - 0s 94ms/step
Q 104+57  T 161  ☒ 160 
1/1 [==============================] - 0s 95ms/step
Q 697+924 T 1621 ☑ 1621
1/1 [==============================] - 0s 84ms/step
Q 38+288  T 326  ☑ 326 
1/1 [==============================] - 0s 96ms/step
Q 980+467 T 1447 ☑ 1447
1/1 [==============================] - 0s 98ms/step
Q 937+12  T 949  ☑ 949 
1/1 [==============================] - 0s 97ms/step
Q 368+91  T 459  ☑ 459 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 13
1407/1407 [==============================] - 54s 38ms/step - loss: 0.0465 - accuracy: 0.9884 - val_loss: 0.0403 - val_accuracy: 0.9902
1/1 [==============================] - 0s 49ms/step
Q 59+482  T 541  ☑ 541 
1/1 [==============================] - 0s 48ms/step
Q 263+69  T 332  ☑ 332 
1/1 [==============================] - 0s 51ms/step
Q 1+426   T 427  ☑ 427 
1/1 [==============================] - 0s 48ms/step
Q 542+204 T 746  ☑ 746 
1/1 [==============================] - 0s 50ms/step
Q 77+693  T 770  ☒ 760 
1/1 [==============================] - 0s 49ms/step
Q 830+60  T 890  ☑ 890 
1/1 [==============================] - 0s 51ms/step
Q 227+73  T 300  ☑ 300 
1/1 [==============================] - 0s 48ms/step
Q 399+951 T 1350 ☑ 1350
1/1 [==============================] - 0s 51ms/step
Q 15+83   T 98   ☑ 98  
1/1 [==============================] - 0s 50ms/step
Q 42+711  T 753  ☑ 753 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 14
1407/1407 [==============================] - 64s 46ms/step - loss: 0.0385 - accuracy: 0.9904 - val_loss: 0.0246 - val_accuracy: 0.9955
1/1 [==============================] - 0s 115ms/step
Q 63+988  T 1051 ☑ 1051
1/1 [==============================] - 0s 128ms/step
Q 207+2   T 209  ☑ 209 
1/1 [==============================] - 0s 127ms/step
Q 31+11   T 42   ☑ 42  
1/1 [==============================] - 0s 126ms/step
Q 561+921 T 1482 ☑ 1482
1/1 [==============================] - 0s 128ms/step
Q 631+56  T 687  ☑ 687 
1/1 [==============================] - 0s 126ms/step
Q 8+527   T 535  ☑ 535 
1/1 [==============================] - 0s 126ms/step
Q 8+412   T 420  ☑ 420 
1/1 [==============================] - 0s 128ms/step
Q 698+4   T 702  ☑ 702 
1/1 [==============================] - 0s 127ms/step
Q 5+429   T 434  ☑ 434 
1/1 [==============================] - 0s 125ms/step
Q 73+336  T 409  ☑ 409 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 15
1407/1407 [==============================] - 55s 39ms/step - loss: 0.0379 - accuracy: 0.9896 - val_loss: 0.0261 - val_accuracy: 0.9944
1/1 [==============================] - 0s 41ms/step
Q 523+644 T 1167 ☑ 1167
1/1 [==============================] - 0s 41ms/step
Q 3+941   T 944  ☑ 944 
1/1 [==============================] - 0s 41ms/step
Q 97+410  T 507  ☑ 507 
1/1 [==============================] - 0s 41ms/step
Q 644+14  T 658  ☑ 658 
1/1 [==============================] - 0s 41ms/step
Q 185+5   T 190  ☑ 190 
1/1 [==============================] - 0s 41ms/step
Q 37+16   T 53   ☑ 53  
1/1 [==============================] - 0s 42ms/step
Q 472+1   T 473  ☑ 473 
1/1 [==============================] - 0s 42ms/step
Q 357+621 T 978  ☑ 978 
1/1 [==============================] - 0s 39ms/step
Q 932+78  T 1010 ☑ 1010
1/1 [==============================] - 0s 39ms/step
Q 969+731 T 1700 ☑ 1700
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 16
1407/1407 [==============================] - 45s 32ms/step - loss: 0.0478 - accuracy: 0.9864 - val_loss: 0.0261 - val_accuracy: 0.9940
1/1 [==============================] - 0s 72ms/step
Q 561+42  T 603  ☑ 603 
1/1 [==============================] - 0s 67ms/step
Q 999+429 T 1428 ☑ 1428
1/1 [==============================] - 0s 67ms/step
Q 12+779  T 791  ☑ 791 
1/1 [==============================] - 0s 69ms/step
Q 95+479  T 574  ☑ 574 
1/1 [==============================] - 0s 67ms/step
Q 561+42  T 603  ☑ 603 
1/1 [==============================] - 0s 73ms/step
Q 529+372 T 901  ☑ 901 
1/1 [==============================] - 0s 65ms/step
Q 437+85  T 522  ☑ 522 
1/1 [==============================] - 0s 67ms/step
Q 565+582 T 1147 ☑ 1147
1/1 [==============================] - 0s 73ms/step
Q 334+605 T 939  ☑ 939 
1/1 [==============================] - 0s 72ms/step
Q 684+44  T 728  ☑ 728 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 17
1407/1407 [==============================] - 59s 42ms/step - loss: 0.0152 - accuracy: 0.9971 - val_loss: 0.0110 - val_accuracy: 0.9981
1/1 [==============================] - 0s 67ms/step
Q 542+204 T 746  ☑ 746 
1/1 [==============================] - 0s 68ms/step
Q 6+879   T 885  ☑ 885 
1/1 [==============================] - 0s 67ms/step
Q 555+795 T 1350 ☑ 1350
1/1 [==============================] - 0s 66ms/step
Q 631+56  T 687  ☑ 687 
1/1 [==============================] - 0s 67ms/step
Q 86+498  T 584  ☑ 584 
1/1 [==============================] - 0s 65ms/step
Q 572+796 T 1368 ☑ 1368
1/1 [==============================] - 0s 74ms/step
Q 249+3   T 252  ☑ 252 
1/1 [==============================] - 0s 76ms/step
Q 105+1   T 106  ☑ 106 
1/1 [==============================] - 0s 73ms/step
Q 98+643  T 741  ☑ 741 
1/1 [==============================] - 0s 67ms/step
Q 396+38  T 434  ☑ 434 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 18
1407/1407 [==============================] - 55s 39ms/step - loss: 0.0375 - accuracy: 0.9894 - val_loss: 0.0227 - val_accuracy: 0.9941
1/1 [==============================] - 0s 76ms/step
Q 571+662 T 1233 ☑ 1233
1/1 [==============================] - 0s 67ms/step
Q 69+680  T 749  ☑ 749 
1/1 [==============================] - 0s 66ms/step
Q 354+203 T 557  ☑ 557 
1/1 [==============================] - 0s 71ms/step
Q 40+93   T 133  ☑ 133 
1/1 [==============================] - 0s 66ms/step
Q 305+991 T 1296 ☑ 1296
1/1 [==============================] - 0s 66ms/step
Q 45+769  T 814  ☑ 814 
1/1 [==============================] - 0s 65ms/step
Q 985+95  T 1080 ☑ 1080
1/1 [==============================] - 0s 74ms/step
Q 4+723   T 727  ☒ 728 
1/1 [==============================] - 0s 66ms/step
Q 83+578  T 661  ☑ 661 
1/1 [==============================] - 0s 67ms/step
Q 314+151 T 465  ☑ 465 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 19
1407/1407 [==============================] - 49s 35ms/step - loss: 0.0355 - accuracy: 0.9902 - val_loss: 0.0105 - val_accuracy: 0.9981
1/1 [==============================] - 0s 41ms/step
Q 470+295 T 765  ☑ 765 
1/1 [==============================] - 0s 44ms/step
Q 718+72  T 790  ☑ 790 
1/1 [==============================] - 0s 42ms/step
Q 434+152 T 586  ☑ 586 
1/1 [==============================] - 0s 42ms/step
Q 196+59  T 255  ☑ 255 
1/1 [==============================] - 0s 41ms/step
Q 234+476 T 710  ☑ 710 
1/1 [==============================] - 0s 41ms/step
Q 858+899 T 1757 ☑ 1757
1/1 [==============================] - 0s 43ms/step
Q 443+8   T 451  ☑ 451 
1/1 [==============================] - 0s 41ms/step
Q 7+408   T 415  ☑ 415 
1/1 [==============================] - 0s 42ms/step
Q 4+0     T 4    ☒ 5   
1/1 [==============================] - 0s 42ms/step
Q 52+890  T 942  ☑ 942 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 20
1407/1407 [==============================] - 36s 26ms/step - loss: 0.0245 - accuracy: 0.9936 - val_loss: 0.0117 - val_accuracy: 0.9978
1/1 [==============================] - 0s 41ms/step
Q 296+37  T 333  ☑ 333 
1/1 [==============================] - 0s 40ms/step
Q 268+122 T 390  ☑ 390 
1/1 [==============================] - 0s 40ms/step
Q 99+986  T 1085 ☑ 1085
1/1 [==============================] - 0s 40ms/step
Q 94+142  T 236  ☑ 236 
1/1 [==============================] - 0s 40ms/step
Q 484+924 T 1408 ☑ 1408
1/1 [==============================] - 0s 41ms/step
Q 4+45    T 49   ☑ 49  
1/1 [==============================] - 0s 41ms/step
Q 437+39  T 476  ☑ 476 
1/1 [==============================] - 0s 42ms/step
Q 351+3   T 354  ☑ 354 
1/1 [==============================] - 0s 38ms/step
Q 41+473  T 514  ☑ 514 
1/1 [==============================] - 0s 41ms/step
Q 636+74  T 710  ☑ 710 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 21
1407/1407 [==============================] - 30s 22ms/step - loss: 0.0140 - accuracy: 0.9970 - val_loss: 0.0130 - val_accuracy: 0.9968
1/1 [==============================] - 0s 36ms/step
Q 518+181 T 699  ☑ 699 
1/1 [==============================] - 0s 37ms/step
Q 471+420 T 891  ☑ 891 
1/1 [==============================] - 0s 38ms/step
Q 30+748  T 778  ☑ 778 
1/1 [==============================] - 0s 36ms/step
Q 49+439  T 488  ☑ 488 
1/1 [==============================] - 0s 37ms/step
Q 50+358  T 408  ☑ 408 
1/1 [==============================] - 0s 39ms/step
Q 960+6   T 966  ☑ 966 
1/1 [==============================] - 0s 38ms/step
Q 600+81  T 681  ☑ 681 
1/1 [==============================] - 0s 36ms/step
Q 783+788 T 1571 ☑ 1571
1/1 [==============================] - 0s 38ms/step
Q 4+695   T 699  ☒ 799 
1/1 [==============================] - 0s 38ms/step
Q 85+33   T 118  ☑ 118 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 22
1407/1407 [==============================] - 31s 22ms/step - loss: 0.0200 - accuracy: 0.9944 - val_loss: 0.0065 - val_accuracy: 0.9990
1/1 [==============================] - 0s 36ms/step
Q 14+128  T 142  ☑ 142 
1/1 [==============================] - 0s 37ms/step
Q 569+0   T 569  ☑ 569 
1/1 [==============================] - 0s 37ms/step
Q 857+40  T 897  ☑ 897 
1/1 [==============================] - 0s 38ms/step
Q 75+459  T 534  ☑ 534 
1/1 [==============================] - 0s 37ms/step
Q 845+472 T 1317 ☑ 1317
1/1 [==============================] - 0s 38ms/step
Q 291+2   T 293  ☑ 293 
1/1 [==============================] - 0s 37ms/step
Q 55+301  T 356  ☑ 356 
1/1 [==============================] - 0s 36ms/step
Q 375+1   T 376  ☑ 376 
1/1 [==============================] - 0s 38ms/step
Q 4+652   T 656  ☑ 656 
1/1 [==============================] - 0s 37ms/step
Q 86+447  T 533  ☑ 533 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 23
1407/1407 [==============================] - 26s 19ms/step - loss: 0.0310 - accuracy: 0.9909 - val_loss: 0.0181 - val_accuracy: 0.9953
1/1 [==============================] - 0s 36ms/step
Q 866+263 T 1129 ☑ 1129
1/1 [==============================] - 0s 35ms/step
Q 20+747  T 767  ☑ 767 
1/1 [==============================] - 0s 36ms/step
Q 728+436 T 1164 ☑ 1164
1/1 [==============================] - 0s 35ms/step
Q 37+99   T 136  ☑ 136 
1/1 [==============================] - 0s 35ms/step
Q 834+64  T 898  ☑ 898 
1/1 [==============================] - 0s 34ms/step
Q 354+203 T 557  ☑ 557 
1/1 [==============================] - 0s 35ms/step
Q 999+21  T 1020 ☑ 1020
1/1 [==============================] - 0s 35ms/step
Q 46+982  T 1028 ☑ 1028
1/1 [==============================] - 0s 35ms/step
Q 824+6   T 830  ☑ 830 
1/1 [==============================] - 0s 35ms/step
Q 25+186  T 211  ☑ 211 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 24
1407/1407 [==============================] - 31s 22ms/step - loss: 0.0174 - accuracy: 0.9954 - val_loss: 0.0274 - val_accuracy: 0.9915
1/1 [==============================] - 0s 41ms/step
Q 465+1   T 466  ☑ 466 
1/1 [==============================] - 0s 41ms/step
Q 721+70  T 791  ☑ 791 
1/1 [==============================] - 0s 39ms/step
Q 255+33  T 288  ☑ 288 
1/1 [==============================] - 0s 39ms/step
Q 659+198 T 857  ☒ 867 
1/1 [==============================] - 0s 41ms/step
Q 762+85  T 847  ☑ 847 
1/1 [==============================] - 0s 39ms/step
Q 780+360 T 1140 ☑ 1140
1/1 [==============================] - 0s 41ms/step
Q 33+745  T 778  ☑ 778 
1/1 [==============================] - 0s 39ms/step
Q 143+39  T 182  ☑ 182 
1/1 [==============================] - 0s 40ms/step
Q 35+82   T 117  ☑ 117 
1/1 [==============================] - 0s 41ms/step
Q 661+736 T 1397 ☑ 1397
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 25
1407/1407 [==============================] - 40s 29ms/step - loss: 0.0048 - accuracy: 0.9994 - val_loss: 0.0042 - val_accuracy: 0.9992
1/1 [==============================] - 0s 64ms/step
Q 693+514 T 1207 ☑ 1207
1/1 [==============================] - 0s 82ms/step
Q 604+847 T 1451 ☑ 1451
1/1 [==============================] - 0s 58ms/step
Q 456+831 T 1287 ☑ 1287
1/1 [==============================] - 0s 68ms/step
Q 795+903 T 1698 ☑ 1698
1/1 [==============================] - 0s 57ms/step
Q 752+983 T 1735 ☑ 1735
1/1 [==============================] - 0s 62ms/step
Q 635+44  T 679  ☑ 679 
1/1 [==============================] - 0s 70ms/step
Q 954+952 T 1906 ☑ 1906
1/1 [==============================] - 0s 59ms/step
Q 748+427 T 1175 ☑ 1175
1/1 [==============================] - 0s 61ms/step
Q 617+141 T 758  ☑ 758 
1/1 [==============================] - 0s 57ms/step
Q 147+94  T 241  ☑ 241 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 26
1407/1407 [==============================] - 65s 46ms/step - loss: 0.0305 - accuracy: 0.9917 - val_loss: 0.0060 - val_accuracy: 0.9989
1/1 [==============================] - 0s 238ms/step
Q 70+130  T 200  ☑ 200 
1/1 [==============================] - 0s 232ms/step
Q 706+42  T 748  ☑ 748 
1/1 [==============================] - 0s 252ms/step
Q 49+486  T 535  ☑ 535 
1/1 [==============================] - 0s 199ms/step
Q 23+363  T 386  ☑ 386 
1/1 [==============================] - 0s 215ms/step
Q 360+737 T 1097 ☑ 1097
1/1 [==============================] - 0s 202ms/step
Q 561+42  T 603  ☑ 603 
1/1 [==============================] - 0s 234ms/step
Q 38+933  T 971  ☑ 971 
1/1 [==============================] - 0s 266ms/step
Q 731+93  T 824  ☑ 824 
1/1 [==============================] - 0s 219ms/step
Q 29+78   T 107  ☑ 107 
1/1 [==============================] - 0s 216ms/step
Q 928+174 T 1102 ☑ 1102
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 27
1407/1407 [==============================] - 78s 56ms/step - loss: 0.0204 - accuracy: 0.9944 - val_loss: 0.0057 - val_accuracy: 0.9989
1/1 [==============================] - 0s 38ms/step
Q 38+144  T 182  ☑ 182 
1/1 [==============================] - 0s 39ms/step
Q 60+216  T 276  ☑ 276 
1/1 [==============================] - 0s 38ms/step
Q 637+32  T 669  ☑ 669 
1/1 [==============================] - 0s 37ms/step
Q 950+7   T 957  ☑ 957 
1/1 [==============================] - 0s 37ms/step
Q 644+20  T 664  ☑ 664 
1/1 [==============================] - 0s 37ms/step
Q 9+530   T 539  ☑ 539 
1/1 [==============================] - 0s 34ms/step
Q 414+149 T 563  ☑ 563 
1/1 [==============================] - 0s 35ms/step
Q 26+0    T 26   ☑ 26  
1/1 [==============================] - 0s 34ms/step
Q 1+448   T 449  ☑ 449 
1/1 [==============================] - 0s 36ms/step
Q 707+13  T 720  ☑ 720 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 28
1407/1407 [==============================] - 29s 21ms/step - loss: 0.0185 - accuracy: 0.9948 - val_loss: 0.0820 - val_accuracy: 0.9778
1/1 [==============================] - 0s 39ms/step
Q 70+818  T 888  ☑ 888 
1/1 [==============================] - 0s 37ms/step
Q 971+160 T 1131 ☑ 1131
1/1 [==============================] - 0s 37ms/step
Q 30+760  T 790  ☒ 800 
1/1 [==============================] - 0s 37ms/step
Q 273+43  T 316  ☑ 316 
1/1 [==============================] - 0s 38ms/step
Q 335+875 T 1210 ☑ 1210
1/1 [==============================] - 0s 39ms/step
Q 796+527 T 1323 ☑ 1323
1/1 [==============================] - 0s 37ms/step
Q 23+23   T 46   ☑ 46  
1/1 [==============================] - 0s 37ms/step
Q 364+24  T 388  ☑ 388 
1/1 [==============================] - 0s 38ms/step
Q 22+771  T 793  ☑ 793 
1/1 [==============================] - 0s 37ms/step
Q 41+532  T 573  ☑ 573 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 29
1407/1407 [==============================] - 41s 29ms/step - loss: 0.0137 - accuracy: 0.9966 - val_loss: 0.0035 - val_accuracy: 0.9994
1/1 [==============================] - 0s 57ms/step
Q 5+120   T 125  ☑ 125 
1/1 [==============================] - 0s 58ms/step
Q 83+803  T 886  ☑ 886 
1/1 [==============================] - 0s 59ms/step
Q 780+663 T 1443 ☑ 1443
1/1 [==============================] - 0s 53ms/step
Q 349+88  T 437  ☑ 437 
1/1 [==============================] - 0s 59ms/step
Q 594+54  T 648  ☑ 648 
1/1 [==============================] - 0s 57ms/step
Q 165+921 T 1086 ☑ 1086
1/1 [==============================] - 0s 57ms/step
Q 658+140 T 798  ☑ 798 
1/1 [==============================] - 0s 60ms/step
Q 930+47  T 977  ☑ 977 
1/1 [==============================] - 0s 59ms/step
Q 28+608  T 636  ☑ 636 
1/1 [==============================] - 0s 59ms/step
Q 892+65  T 957  ☑ 957 

```
</div>
You'll get to 99+% validation accuracy after ~30 epochs.

Example available on HuggingFace.

| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/🤗%20Model-Addition%20LSTM-black.svg)](https://huggingface.co/keras-io/addition-lstm) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-Addition%20LSTM-black.svg)](https://huggingface.co/spaces/keras-io/addition-lstm) |

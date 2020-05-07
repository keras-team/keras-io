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
[Sequence to Sequence Learning with Neural Networks](

 http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

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
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 128)               72192     
_________________________________________________________________
repeat_vector (RepeatVector) (None, 4, 128)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 4, 128)            131584    
_________________________________________________________________
dense (Dense)                (None, 4, 12)             1548      
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
1407/1407 [==============================] - 8s 6ms/step - loss: 1.7622 - accuracy: 0.3571 - val_loss: 1.5618 - val_accuracy: 0.4175
Q 99+580  T 679  ☒ 905 
Q 800+652 T 1452 ☒ 1311
Q 900+0   T 900  ☒ 909 
Q 26+12   T 38   ☒ 22  
Q 8+397   T 405  ☒ 903 
Q 14+478  T 492  ☒ 441 
Q 59+589  T 648  ☒ 551 
Q 653+77  T 730  ☒ 601 
Q 10+35   T 45   ☒ 11  
Q 51+185  T 236  ☒ 211 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 2
1407/1407 [==============================] - 8s 6ms/step - loss: 1.3387 - accuracy: 0.5005 - val_loss: 1.1605 - val_accuracy: 0.5726
Q 373+107 T 480  ☒ 417 
Q 910+771 T 1681 ☒ 1610
Q 494+86  T 580  ☒ 555 
Q 829+503 T 1332 ☒ 1283
Q 820+292 T 1112 ☒ 1102
Q 276+741 T 1017 ☒ 1000
Q 208+84  T 292  ☒ 397 
Q 28+349  T 377  ☑ 377 
Q 875+47  T 922  ☒ 930 
Q 654+81  T 735  ☒ 720 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 3
1407/1407 [==============================] - 8s 6ms/step - loss: 1.0369 - accuracy: 0.6144 - val_loss: 0.9534 - val_accuracy: 0.6291
Q 73+290  T 363  ☒ 358 
Q 284+928 T 1212 ☒ 1202
Q 12+775  T 787  ☒ 783 
Q 652+651 T 1303 ☒ 1302
Q 12+940  T 952  ☒ 953 
Q 10+89   T 99   ☒ 10  
Q 86+947  T 1033 ☒ 1023
Q 866+10  T 876  ☒ 873 
Q 196+8   T 204  ☒ 208 
Q 3+763   T 766  ☒ 763 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 4
1407/1407 [==============================] - 8s 6ms/step - loss: 0.8553 - accuracy: 0.6862 - val_loss: 0.8083 - val_accuracy: 0.6976
Q 561+68  T 629  ☒ 625 
Q 1+878   T 879  ☒ 875 
Q 461+525 T 986  ☒ 988 
Q 453+84  T 537  ☒ 535 
Q 92+33   T 125  ☒ 121 
Q 29+624  T 653  ☒ 651 
Q 656+89  T 745  ☑ 745 
Q 30+418  T 448  ☒ 455 
Q 600+3   T 603  ☒ 605 
Q 26+346  T 372  ☒ 375 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 5
1407/1407 [==============================] - 8s 6ms/step - loss: 0.7516 - accuracy: 0.7269 - val_loss: 0.7113 - val_accuracy: 0.7427
Q 522+451 T 973  ☒ 978 
Q 721+69  T 790  ☒ 784 
Q 294+53  T 347  ☒ 344 
Q 80+48   T 128  ☒ 121 
Q 343+182 T 525  ☒ 524 
Q 17+83   T 100  ☒ 90  
Q 132+3   T 135  ☒ 134 
Q 63+963  T 1026 ☒ 1028
Q 427+655 T 1082 ☒ 1084
Q 76+36   T 112  ☒ 114 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 6
1407/1407 [==============================] - 8s 6ms/step - loss: 0.6492 - accuracy: 0.7639 - val_loss: 0.5625 - val_accuracy: 0.7868
Q 20+56   T 76   ☒ 77  
Q 904+74  T 978  ☒ 979 
Q 716+736 T 1452 ☒ 1451
Q 69+512  T 581  ☑ 581 
Q 82+501  T 583  ☒ 584 
Q 297+442 T 739  ☒ 730 
Q 759+30  T 789  ☑ 789 
Q 160+451 T 611  ☒ 612 
Q 765+30  T 795  ☒ 796 
Q 658+37  T 695  ☒ 694 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 7
1407/1407 [==============================] - 8s 6ms/step - loss: 0.4077 - accuracy: 0.8595 - val_loss: 0.3167 - val_accuracy: 0.9025
Q 558+81  T 639  ☑ 639 
Q 795+73  T 868  ☑ 868 
Q 98+93   T 191  ☑ 191 
Q 7+454   T 461  ☑ 461 
Q 64+764  T 828  ☑ 828 
Q 91+14   T 105  ☒ 104 
Q 554+53  T 607  ☑ 607 
Q 7+454   T 461  ☑ 461 
Q 411+46  T 457  ☑ 457 
Q 991+55  T 1046 ☑ 1046
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 8
1407/1407 [==============================] - 8s 6ms/step - loss: 0.2317 - accuracy: 0.9354 - val_loss: 0.2460 - val_accuracy: 0.9119
Q 136+57  T 193  ☑ 193 
Q 896+60  T 956  ☒ 957 
Q 453+846 T 1299 ☑ 1299
Q 86+601  T 687  ☑ 687 
Q 272+230 T 502  ☒ 503 
Q 675+886 T 1561 ☒ 1551
Q 121+634 T 755  ☒ 745 
Q 17+853  T 870  ☑ 870 
Q 9+40    T 49   ☒ 40  
Q 290+80  T 370  ☒ 481 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 9
1407/1407 [==============================] - 8s 6ms/step - loss: 0.1434 - accuracy: 0.9665 - val_loss: 0.1223 - val_accuracy: 0.9686
Q 1+532   T 533  ☑ 533 
Q 298+20  T 318  ☑ 318 
Q 750+28  T 778  ☑ 778 
Q 44+576  T 620  ☑ 620 
Q 988+481 T 1469 ☒ 1479
Q 234+829 T 1063 ☑ 1063
Q 855+19  T 874  ☑ 874 
Q 741+56  T 797  ☑ 797 
Q 7+643   T 650  ☑ 650 
Q 14+598  T 612  ☒ 613 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 10
1407/1407 [==============================] - 8s 6ms/step - loss: 0.1024 - accuracy: 0.9764 - val_loss: 0.0948 - val_accuracy: 0.9750
Q 380+26  T 406  ☑ 406 
Q 813+679 T 1492 ☒ 1592
Q 3+763   T 766  ☑ 766 
Q 677+83  T 760  ☑ 760 
Q 474+13  T 487  ☑ 487 
Q 861+4   T 865  ☑ 865 
Q 83+24   T 107  ☑ 107 
Q 67+177  T 244  ☑ 244 
Q 841+31  T 872  ☑ 872 
Q 740+121 T 861  ☒ 871 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 11
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0780 - accuracy: 0.9812 - val_loss: 0.0537 - val_accuracy: 0.9893
Q 199+36  T 235  ☑ 235 
Q 970+78  T 1048 ☑ 1048
Q 21+610  T 631  ☑ 631 
Q 36+686  T 722  ☑ 722 
Q 476+488 T 964  ☑ 964 
Q 583+1   T 584  ☑ 584 
Q 72+408  T 480  ☑ 480 
Q 0+141   T 141  ☑ 141 
Q 858+837 T 1695 ☒ 1795
Q 27+346  T 373  ☑ 373 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 12
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0481 - accuracy: 0.9900 - val_loss: 0.0298 - val_accuracy: 0.9965
Q 23+44   T 67   ☑ 67  
Q 905+251 T 1156 ☑ 1156
Q 298+46  T 344  ☑ 344 
Q 320+31  T 351  ☑ 351 
Q 854+730 T 1584 ☑ 1584
Q 765+30  T 795  ☑ 795 
Q 60+179  T 239  ☑ 239 
Q 792+76  T 868  ☑ 868 
Q 79+114  T 193  ☑ 193 
Q 354+23  T 377  ☑ 377 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 13
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0547 - accuracy: 0.9857 - val_loss: 0.0956 - val_accuracy: 0.9682
Q 4+568   T 572  ☑ 572 
Q 199+867 T 1066 ☑ 1066
Q 77+727  T 804  ☑ 804 
Q 47+385  T 432  ☑ 432 
Q 21+20   T 41   ☑ 41  
Q 18+521  T 539  ☑ 539 
Q 409+58  T 467  ☑ 467 
Q 201+99  T 300  ☒ 200 
Q 46+205  T 251  ☑ 251 
Q 613+984 T 1597 ☑ 1597
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 14
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0445 - accuracy: 0.9889 - val_loss: 0.0364 - val_accuracy: 0.9914
Q 50+770  T 820  ☑ 820 
Q 338+329 T 667  ☑ 667 
Q 535+529 T 1064 ☑ 1064
Q 50+907  T 957  ☑ 957 
Q 266+30  T 296  ☑ 296 
Q 65+91   T 156  ☑ 156 
Q 43+8    T 51   ☑ 51  
Q 714+3   T 717  ☑ 717 
Q 415+38  T 453  ☑ 453 
Q 432+252 T 684  ☑ 684 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 15
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0324 - accuracy: 0.9920 - val_loss: 0.0196 - val_accuracy: 0.9965
Q 748+45  T 793  ☑ 793 
Q 457+2   T 459  ☑ 459 
Q 205+30  T 235  ☑ 235 
Q 16+402  T 418  ☑ 418 
Q 810+415 T 1225 ☑ 1225
Q 917+421 T 1338 ☑ 1338
Q 803+68  T 871  ☑ 871 
Q 66+351  T 417  ☑ 417 
Q 901+3   T 904  ☑ 904 
Q 26+897  T 923  ☑ 923 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 16
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0353 - accuracy: 0.9906 - val_loss: 0.0174 - val_accuracy: 0.9966
Q 295+57  T 352  ☑ 352 
Q 4+683   T 687  ☑ 687 
Q 608+892 T 1500 ☒ 1400
Q 618+71  T 689  ☑ 689 
Q 43+299  T 342  ☑ 342 
Q 662+9   T 671  ☑ 671 
Q 50+318  T 368  ☑ 368 
Q 33+665  T 698  ☑ 698 
Q 2+11    T 13   ☑ 13  
Q 29+261  T 290  ☑ 290 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 17
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0368 - accuracy: 0.9903 - val_loss: 0.0148 - val_accuracy: 0.9971
Q 4+568   T 572  ☑ 572 
Q 121+316 T 437  ☑ 437 
Q 78+662  T 740  ☑ 740 
Q 883+47  T 930  ☑ 930 
Q 696+78  T 774  ☑ 774 
Q 23+921  T 944  ☑ 944 
Q 768+813 T 1581 ☑ 1581
Q 1+586   T 587  ☑ 587 
Q 276+92  T 368  ☑ 368 
Q 623+9   T 632  ☑ 632 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 18
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0317 - accuracy: 0.9917 - val_loss: 0.0119 - val_accuracy: 0.9985
Q 50+430  T 480  ☑ 480 
Q 583+86  T 669  ☑ 669 
Q 899+342 T 1241 ☑ 1241
Q 164+369 T 533  ☑ 533 
Q 728+9   T 737  ☑ 737 
Q 182+85  T 267  ☑ 267 
Q 81+323  T 404  ☑ 404 
Q 91+85   T 176  ☑ 176 
Q 602+606 T 1208 ☑ 1208
Q 334+193 T 527  ☑ 527 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 19
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0225 - accuracy: 0.9940 - val_loss: 0.0291 - val_accuracy: 0.9915
Q 416+636 T 1052 ☑ 1052
Q 224+330 T 554  ☑ 554 
Q 347+8   T 355  ☑ 355 
Q 918+890 T 1808 ☒ 1809
Q 12+852  T 864  ☑ 864 
Q 535+93  T 628  ☑ 628 
Q 476+98  T 574  ☑ 574 
Q 89+682  T 771  ☑ 771 
Q 731+99  T 830  ☑ 830 
Q 222+45  T 267  ☑ 267 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 20
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0325 - accuracy: 0.9914 - val_loss: 0.0118 - val_accuracy: 0.9980
Q 342+270 T 612  ☑ 612 
Q 20+188  T 208  ☑ 208 
Q 37+401  T 438  ☑ 438 
Q 672+417 T 1089 ☑ 1089
Q 597+12  T 609  ☑ 609 
Q 569+81  T 650  ☑ 650 
Q 58+46   T 104  ☑ 104 
Q 48+46   T 94   ☑ 94  
Q 801+47  T 848  ☑ 848 
Q 356+550 T 906  ☑ 906 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 21
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0226 - accuracy: 0.9941 - val_loss: 0.0097 - val_accuracy: 0.9984
Q 77+188  T 265  ☑ 265 
Q 449+35  T 484  ☑ 484 
Q 76+287  T 363  ☑ 363 
Q 204+231 T 435  ☑ 435 
Q 880+1   T 881  ☑ 881 
Q 571+79  T 650  ☑ 650 
Q 6+126   T 132  ☑ 132 
Q 567+6   T 573  ☑ 573 
Q 284+928 T 1212 ☑ 1212
Q 889+9   T 898  ☑ 898 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 22
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0224 - accuracy: 0.9937 - val_loss: 0.0100 - val_accuracy: 0.9980
Q 694+851 T 1545 ☑ 1545
Q 84+582  T 666  ☑ 666 
Q 900+476 T 1376 ☑ 1376
Q 661+848 T 1509 ☑ 1509
Q 2+210   T 212  ☑ 212 
Q 4+568   T 572  ☑ 572 
Q 699+555 T 1254 ☑ 1254
Q 750+64  T 814  ☑ 814 
Q 299+938 T 1237 ☑ 1237
Q 213+94  T 307  ☑ 307 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 23
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0285 - accuracy: 0.9919 - val_loss: 0.0769 - val_accuracy: 0.9790
Q 70+650  T 720  ☑ 720 
Q 914+8   T 922  ☑ 922 
Q 925+53  T 978  ☑ 978 
Q 19+49   T 68   ☒ 78  
Q 12+940  T 952  ☑ 952 
Q 85+879  T 964  ☑ 964 
Q 652+461 T 1113 ☑ 1113
Q 223+59  T 282  ☑ 282 
Q 361+55  T 416  ☑ 416 
Q 940+1   T 941  ☑ 941 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 24
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0101 - accuracy: 0.9979 - val_loss: 0.0218 - val_accuracy: 0.9937
Q 653+77  T 730  ☑ 730 
Q 73+155  T 228  ☑ 228 
Q 62+355  T 417  ☑ 417 
Q 859+916 T 1775 ☑ 1775
Q 201+153 T 354  ☑ 354 
Q 469+1   T 470  ☑ 470 
Q 52+363  T 415  ☑ 415 
Q 22+706  T 728  ☑ 728 
Q 58+33   T 91   ☑ 91  
Q 371+51  T 422  ☑ 422 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 25
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0174 - accuracy: 0.9952 - val_loss: 0.1332 - val_accuracy: 0.9670
Q 213+94  T 307  ☑ 307 
Q 390+7   T 397  ☑ 397 
Q 14+498  T 512  ☑ 512 
Q 14+312  T 326  ☑ 326 
Q 56+653  T 709  ☑ 709 
Q 37+28   T 65   ☑ 65  
Q 113+70  T 183  ☑ 183 
Q 326+398 T 724  ☑ 724 
Q 137+8   T 145  ☑ 145 
Q 50+19   T 69   ☑ 69  
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 26
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0231 - accuracy: 0.9937 - val_loss: 0.0088 - val_accuracy: 0.9983
Q 886+20  T 906  ☑ 906 
Q 61+790  T 851  ☑ 851 
Q 610+63  T 673  ☑ 673 
Q 27+20   T 47   ☑ 47  
Q 130+32  T 162  ☑ 162 
Q 555+25  T 580  ☑ 580 
Q 95+43   T 138  ☑ 138 
Q 5+427   T 432  ☑ 432 
Q 395+651 T 1046 ☑ 1046
Q 188+19  T 207  ☑ 207 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 27
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0218 - accuracy: 0.9940 - val_loss: 0.0070 - val_accuracy: 0.9987
Q 495+735 T 1230 ☑ 1230
Q 74+607  T 681  ☑ 681 
Q 225+56  T 281  ☑ 281 
Q 581+589 T 1170 ☑ 1170
Q 37+953  T 990  ☑ 990 
Q 17+510  T 527  ☑ 527 
Q 621+73  T 694  ☑ 694 
Q 54+298  T 352  ☑ 352 
Q 636+518 T 1154 ☑ 1154
Q 7+673   T 680  ☑ 680 
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 28
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0114 - accuracy: 0.9971 - val_loss: 0.0238 - val_accuracy: 0.9930
Q 67+12   T 79   ☑ 79  
Q 109+464 T 573  ☑ 573 
Q 4+52    T 56   ☑ 56  
Q 907+746 T 1653 ☑ 1653
Q 153+864 T 1017 ☑ 1017
Q 666+77  T 743  ☑ 743 
Q 65+777  T 842  ☑ 842 
Q 52+60   T 112  ☑ 112 
Q 941+692 T 1633 ☑ 1633
Q 931+666 T 1597 ☑ 1597
```
</div>
    
<div class="k-default-codeblock">
```
Iteration 29
1407/1407 [==============================] - 8s 6ms/step - loss: 0.0262 - accuracy: 0.9929 - val_loss: 0.0643 - val_accuracy: 0.9804
Q 128+86  T 214  ☑ 214 
Q 20+494  T 514  ☑ 514 
Q 34+896  T 930  ☑ 930 
Q 372+15  T 387  ☑ 387 
Q 466+63  T 529  ☑ 529 
Q 327+9   T 336  ☑ 336 
Q 458+85  T 543  ☑ 543 
Q 134+431 T 565  ☑ 565 
Q 807+289 T 1096 ☑ 1096
Q 100+60  T 160  ☑ 160 

```
</div>
You'll get to 99+% validation accuracy after ~30 epochs.


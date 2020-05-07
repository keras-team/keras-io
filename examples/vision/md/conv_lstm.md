# Next-frame prediction with Conv-LSTM

**Author:** [jeammimi](https://github.com/jeammimi)<br>
**Date created:** 2016/11/02<br>
**Last modified:** 2020/05/01<br>
**Description:** Predict the next frame in a sequence using a Conv-LSTM model.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/conv_lstm.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/conv_lstm.py)



---
## Introduction

This script demonstrates the use of a convolutional LSTM model.
The model is used to predict the next frame of an artificially
generated movie which contains moving squares.


---
## Setup



```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pylab as plt

```

---
## Build a model

We create a model which take as input movies of shape
`(n_frames, width, height, channels)` and returns a movie
of identical shape.



```python
seq = keras.Sequential(
    [
        keras.Input(
            shape=(None, 40, 40, 1)
        ),  # Variable-length sequence of 40x40x1 frames
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
        ),
    ]
)
seq.compile(loss="binary_crossentropy", optimizer="adadelta")

```

---
## Generate artificial data

Generate movies with 3 to 7 moving squares inside.
The squares are of shape 1x1 or 2x2 pixels,
and move linearly over time.
For convenience, we first create movies with bigger width and height (80x80)
and at the end we select a 40x40 window.



```python

def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[
                    i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0
                ] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the model to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1) ** np.random.randint(0, 2)
                    noisy_movies[
                        i,
                        t,
                        x_shift - w - 1 : x_shift + w + 1,
                        y_shift - w - 1 : y_shift + w + 1,
                        0,
                    ] += (noise_f * 0.1)

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[
                    i, t, x_shift - w : x_shift + w, y_shift - w : y_shift + w, 0
                ] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies


```

---
## Train the model



```python
epochs = 200

noisy_movies, shifted_movies = generate_movies(n_samples=1200)
seq.fit(
    noisy_movies[:1000],
    shifted_movies[:1000],
    batch_size=10,
    epochs=epochs,
    verbose=2,
    validation_split=0.1,
)

```

<div class="k-default-codeblock">
```
Epoch 1/200
90/90 - 20s - loss: 0.8062 - val_loss: 0.7147
Epoch 2/200
90/90 - 19s - loss: 0.7563 - val_loss: 0.7552
Epoch 3/200
90/90 - 19s - loss: 0.7267 - val_loss: 0.8009
Epoch 4/200
90/90 - 19s - loss: 0.7025 - val_loss: 0.7958
Epoch 5/200
90/90 - 19s - loss: 0.6813 - val_loss: 0.7209
Epoch 6/200
90/90 - 19s - loss: 0.6623 - val_loss: 0.6302
Epoch 7/200
90/90 - 19s - loss: 0.6447 - val_loss: 0.5746
Epoch 8/200
90/90 - 19s - loss: 0.6281 - val_loss: 0.5725
Epoch 9/200
90/90 - 19s - loss: 0.6124 - val_loss: 0.5803
Epoch 10/200
90/90 - 19s - loss: 0.5974 - val_loss: 0.5748
Epoch 11/200
90/90 - 19s - loss: 0.5828 - val_loss: 0.5610
Epoch 12/200
90/90 - 19s - loss: 0.5685 - val_loss: 0.5437
Epoch 13/200
90/90 - 19s - loss: 0.5536 - val_loss: 0.5214
Epoch 14/200
90/90 - 19s - loss: 0.5381 - val_loss: 0.5185
Epoch 15/200
90/90 - 19s - loss: 0.5249 - val_loss: 0.5096
Epoch 16/200
90/90 - 19s - loss: 0.5130 - val_loss: 0.5038
Epoch 17/200
90/90 - 19s - loss: 0.5014 - val_loss: 0.4855
Epoch 18/200
90/90 - 19s - loss: 0.4903 - val_loss: 0.4789
Epoch 19/200
90/90 - 19s - loss: 0.4791 - val_loss: 0.4642
Epoch 20/200
90/90 - 19s - loss: 0.4687 - val_loss: 0.4550
Epoch 21/200
90/90 - 19s - loss: 0.4586 - val_loss: 0.4458
Epoch 22/200
90/90 - 19s - loss: 0.4487 - val_loss: 0.4279
Epoch 23/200
90/90 - 19s - loss: 0.4392 - val_loss: 0.4208
Epoch 24/200
90/90 - 19s - loss: 0.4297 - val_loss: 0.4136
Epoch 25/200
90/90 - 19s - loss: 0.4207 - val_loss: 0.4053
Epoch 26/200
90/90 - 19s - loss: 0.4119 - val_loss: 0.3948
Epoch 27/200
90/90 - 19s - loss: 0.4028 - val_loss: 0.3890
Epoch 28/200
90/90 - 19s - loss: 0.3942 - val_loss: 0.3815
Epoch 29/200
90/90 - 19s - loss: 0.3856 - val_loss: 0.3686
Epoch 30/200
90/90 - 19s - loss: 0.3773 - val_loss: 0.3652
Epoch 31/200
90/90 - 19s - loss: 0.3691 - val_loss: 0.3579
Epoch 32/200
90/90 - 19s - loss: 0.3611 - val_loss: 0.3510
Epoch 33/200
90/90 - 19s - loss: 0.3533 - val_loss: 0.3462
Epoch 34/200
90/90 - 19s - loss: 0.3457 - val_loss: 0.3337
Epoch 35/200
90/90 - 19s - loss: 0.3382 - val_loss: 0.3300
Epoch 36/200
90/90 - 19s - loss: 0.3309 - val_loss: 0.3273
Epoch 37/200
90/90 - 19s - loss: 0.3237 - val_loss: 0.3183
Epoch 38/200
90/90 - 19s - loss: 0.3167 - val_loss: 0.3099
Epoch 39/200
90/90 - 19s - loss: 0.3100 - val_loss: 0.3041
Epoch 40/200
90/90 - 19s - loss: 0.3031 - val_loss: 0.2985
Epoch 41/200
90/90 - 19s - loss: 0.2967 - val_loss: 0.2967
Epoch 42/200
90/90 - 19s - loss: 0.2903 - val_loss: 0.2871
Epoch 43/200
90/90 - 19s - loss: 0.2840 - val_loss: 0.2832
Epoch 44/200
90/90 - 19s - loss: 0.2780 - val_loss: 0.2763
Epoch 45/200
90/90 - 19s - loss: 0.2718 - val_loss: 0.2696
Epoch 46/200
90/90 - 19s - loss: 0.2658 - val_loss: 0.2648
Epoch 47/200
90/90 - 19s - loss: 0.2601 - val_loss: 0.2585
Epoch 48/200
90/90 - 19s - loss: 0.2545 - val_loss: 0.2569
Epoch 49/200
90/90 - 19s - loss: 0.2489 - val_loss: 0.2453
Epoch 50/200
90/90 - 19s - loss: 0.2435 - val_loss: 0.2424
Epoch 51/200
90/90 - 19s - loss: 0.2381 - val_loss: 0.2368
Epoch 52/200
90/90 - 19s - loss: 0.2329 - val_loss: 0.2328
Epoch 53/200
90/90 - 19s - loss: 0.2278 - val_loss: 0.2253
Epoch 54/200
90/90 - 19s - loss: 0.2230 - val_loss: 0.2211
Epoch 55/200
90/90 - 19s - loss: 0.2181 - val_loss: 0.2154
Epoch 56/200
90/90 - 19s - loss: 0.2134 - val_loss: 0.2124
Epoch 57/200
90/90 - 19s - loss: 0.2090 - val_loss: 0.2075
Epoch 58/200
90/90 - 19s - loss: 0.2044 - val_loss: 0.2002
Epoch 59/200
90/90 - 19s - loss: 0.2000 - val_loss: 0.1982
Epoch 60/200
90/90 - 19s - loss: 0.1959 - val_loss: 0.1921
Epoch 61/200
90/90 - 19s - loss: 0.1916 - val_loss: 0.1885
Epoch 62/200
90/90 - 19s - loss: 0.1877 - val_loss: 0.1876
Epoch 63/200
90/90 - 19s - loss: 0.1837 - val_loss: 0.1830
Epoch 64/200
90/90 - 19s - loss: 0.1798 - val_loss: 0.1796
Epoch 65/200
90/90 - 19s - loss: 0.1763 - val_loss: 0.1742
Epoch 66/200
90/90 - 19s - loss: 0.1726 - val_loss: 0.1678
Epoch 67/200
90/90 - 19s - loss: 0.1690 - val_loss: 0.1670
Epoch 68/200
90/90 - 19s - loss: 0.1656 - val_loss: 0.1628
Epoch 69/200
90/90 - 19s - loss: 0.1622 - val_loss: 0.1620
Epoch 70/200
90/90 - 19s - loss: 0.1590 - val_loss: 0.1584
Epoch 71/200
90/90 - 19s - loss: 0.1557 - val_loss: 0.1529
Epoch 72/200
90/90 - 19s - loss: 0.1525 - val_loss: 0.1495
Epoch 73/200
90/90 - 19s - loss: 0.1494 - val_loss: 0.1458
Epoch 74/200
90/90 - 19s - loss: 0.1465 - val_loss: 0.1420
Epoch 75/200
90/90 - 19s - loss: 0.1437 - val_loss: 0.1411
Epoch 76/200
90/90 - 19s - loss: 0.1410 - val_loss: 0.1387
Epoch 77/200
90/90 - 19s - loss: 0.1381 - val_loss: 0.1334
Epoch 78/200
90/90 - 19s - loss: 0.1353 - val_loss: 0.1324
Epoch 79/200
90/90 - 19s - loss: 0.1330 - val_loss: 0.1269
Epoch 80/200
90/90 - 19s - loss: 0.1303 - val_loss: 0.1275
Epoch 81/200
90/90 - 19s - loss: 0.1280 - val_loss: 0.1230
Epoch 82/200
90/90 - 19s - loss: 0.1255 - val_loss: 0.1210
Epoch 83/200
90/90 - 19s - loss: 0.1230 - val_loss: 0.1201
Epoch 84/200
90/90 - 19s - loss: 0.1208 - val_loss: 0.1177
Epoch 85/200
90/90 - 19s - loss: 0.1187 - val_loss: 0.1154
Epoch 86/200
90/90 - 19s - loss: 0.1164 - val_loss: 0.1134
Epoch 87/200
90/90 - 19s - loss: 0.1143 - val_loss: 0.1122
Epoch 88/200
90/90 - 19s - loss: 0.1120 - val_loss: 0.1094
Epoch 89/200
90/90 - 19s - loss: 0.1101 - val_loss: 0.1070
Epoch 90/200
90/90 - 19s - loss: 0.1080 - val_loss: 0.1043
Epoch 91/200
90/90 - 19s - loss: 0.1064 - val_loss: 0.1030
Epoch 92/200
90/90 - 19s - loss: 0.1043 - val_loss: 0.1011
Epoch 93/200
90/90 - 19s - loss: 0.1024 - val_loss: 0.1002
Epoch 94/200
90/90 - 19s - loss: 0.1006 - val_loss: 0.0977
Epoch 95/200
90/90 - 19s - loss: 0.0988 - val_loss: 0.0963
Epoch 96/200
90/90 - 19s - loss: 0.0972 - val_loss: 0.0936
Epoch 97/200
90/90 - 19s - loss: 0.0954 - val_loss: 0.0927
Epoch 98/200
90/90 - 19s - loss: 0.0940 - val_loss: 0.0905
Epoch 99/200
90/90 - 19s - loss: 0.0919 - val_loss: 0.0909
Epoch 100/200
90/90 - 19s - loss: 0.0905 - val_loss: 0.0868
Epoch 101/200
90/90 - 19s - loss: 0.0891 - val_loss: 0.0866
Epoch 102/200
90/90 - 19s - loss: 0.0876 - val_loss: 0.0856
Epoch 103/200
90/90 - 19s - loss: 0.0862 - val_loss: 0.0834
Epoch 104/200
90/90 - 19s - loss: 0.0846 - val_loss: 0.0820
Epoch 105/200
90/90 - 19s - loss: 0.0833 - val_loss: 0.0796
Epoch 106/200
90/90 - 19s - loss: 0.0820 - val_loss: 0.0791
Epoch 107/200
90/90 - 19s - loss: 0.0807 - val_loss: 0.0786
Epoch 108/200
90/90 - 19s - loss: 0.0792 - val_loss: 0.0766
Epoch 109/200
90/90 - 19s - loss: 0.0780 - val_loss: 0.0755
Epoch 110/200
90/90 - 19s - loss: 0.0767 - val_loss: 0.0737
Epoch 111/200
90/90 - 19s - loss: 0.0755 - val_loss: 0.0736
Epoch 112/200
90/90 - 19s - loss: 0.0744 - val_loss: 0.0718
Epoch 113/200
90/90 - 19s - loss: 0.0732 - val_loss: 0.0706
Epoch 114/200
90/90 - 19s - loss: 0.0721 - val_loss: 0.0697
Epoch 115/200
90/90 - 19s - loss: 0.0709 - val_loss: 0.0681
Epoch 116/200
90/90 - 19s - loss: 0.0699 - val_loss: 0.0683
Epoch 117/200
90/90 - 19s - loss: 0.0687 - val_loss: 0.0667
Epoch 118/200
90/90 - 19s - loss: 0.0678 - val_loss: 0.0645
Epoch 119/200
90/90 - 19s - loss: 0.0665 - val_loss: 0.0640
Epoch 120/200
90/90 - 19s - loss: 0.0656 - val_loss: 0.0632
Epoch 121/200
90/90 - 19s - loss: 0.0647 - val_loss: 0.0620
Epoch 122/200
90/90 - 19s - loss: 0.0638 - val_loss: 0.0617
Epoch 123/200
90/90 - 19s - loss: 0.0630 - val_loss: 0.0590
Epoch 124/200
90/90 - 19s - loss: 0.0619 - val_loss: 0.0595
Epoch 125/200
90/90 - 19s - loss: 0.0611 - val_loss: 0.0581
Epoch 126/200
90/90 - 19s - loss: 0.0603 - val_loss: 0.0581
Epoch 127/200
90/90 - 19s - loss: 0.0593 - val_loss: 0.0570
Epoch 128/200
90/90 - 19s - loss: 0.0584 - val_loss: 0.0560
Epoch 129/200
90/90 - 19s - loss: 0.0578 - val_loss: 0.0553
Epoch 130/200
90/90 - 19s - loss: 0.0569 - val_loss: 0.0550
Epoch 131/200
90/90 - 19s - loss: 0.0560 - val_loss: 0.0535
Epoch 132/200
90/90 - 19s - loss: 0.0555 - val_loss: 0.0528
Epoch 133/200
90/90 - 19s - loss: 0.0546 - val_loss: 0.0524
Epoch 134/200
90/90 - 19s - loss: 0.0540 - val_loss: 0.0514
Epoch 135/200
90/90 - 19s - loss: 0.0531 - val_loss: 0.0510
Epoch 136/200
90/90 - 19s - loss: 0.0524 - val_loss: 0.0499
Epoch 137/200
90/90 - 19s - loss: 0.0517 - val_loss: 0.0496
Epoch 138/200
90/90 - 19s - loss: 0.0510 - val_loss: 0.0500
Epoch 139/200
90/90 - 19s - loss: 0.0505 - val_loss: 0.0479
Epoch 140/200
90/90 - 19s - loss: 0.0497 - val_loss: 0.0480
Epoch 141/200
90/90 - 19s - loss: 0.0490 - val_loss: 0.0470
Epoch 142/200
90/90 - 19s - loss: 0.0485 - val_loss: 0.0462
Epoch 143/200
90/90 - 19s - loss: 0.0479 - val_loss: 0.0455
Epoch 144/200
90/90 - 19s - loss: 0.0471 - val_loss: 0.0453
Epoch 145/200
90/90 - 19s - loss: 0.0466 - val_loss: 0.0447
Epoch 146/200
90/90 - 19s - loss: 0.0459 - val_loss: 0.0435
Epoch 147/200
90/90 - 19s - loss: 0.0454 - val_loss: 0.0431
Epoch 148/200
90/90 - 19s - loss: 0.0448 - val_loss: 0.0427
Epoch 149/200
90/90 - 19s - loss: 0.0444 - val_loss: 0.0422
Epoch 150/200
90/90 - 19s - loss: 0.0437 - val_loss: 0.0424
Epoch 151/200
90/90 - 19s - loss: 0.0433 - val_loss: 0.0419
Epoch 152/200
90/90 - 19s - loss: 0.0428 - val_loss: 0.0410
Epoch 153/200
90/90 - 19s - loss: 0.0422 - val_loss: 0.0402
Epoch 154/200
90/90 - 19s - loss: 0.0418 - val_loss: 0.0399
Epoch 155/200
90/90 - 19s - loss: 0.0413 - val_loss: 0.0392
Epoch 156/200
90/90 - 19s - loss: 0.0407 - val_loss: 0.0387
Epoch 157/200
90/90 - 19s - loss: 0.0403 - val_loss: 0.0388
Epoch 158/200
90/90 - 19s - loss: 0.0399 - val_loss: 0.0378
Epoch 159/200
90/90 - 19s - loss: 0.0395 - val_loss: 0.0374
Epoch 160/200
90/90 - 19s - loss: 0.0390 - val_loss: 0.0374
Epoch 161/200
90/90 - 19s - loss: 0.0387 - val_loss: 0.0370
Epoch 162/200
90/90 - 19s - loss: 0.0382 - val_loss: 0.0367
Epoch 163/200
90/90 - 19s - loss: 0.0377 - val_loss: 0.0361
Epoch 164/200
90/90 - 19s - loss: 0.0374 - val_loss: 0.0355
Epoch 165/200
90/90 - 19s - loss: 0.0371 - val_loss: 0.0356
Epoch 166/200
90/90 - 19s - loss: 0.0366 - val_loss: 0.0349
Epoch 167/200
90/90 - 19s - loss: 0.0363 - val_loss: 0.0342
Epoch 168/200
90/90 - 19s - loss: 0.0360 - val_loss: 0.0343
Epoch 169/200
90/90 - 19s - loss: 0.0354 - val_loss: 0.0338
Epoch 170/200
90/90 - 19s - loss: 0.0351 - val_loss: 0.0338
Epoch 171/200
90/90 - 19s - loss: 0.0349 - val_loss: 0.0332
Epoch 172/200
90/90 - 19s - loss: 0.0345 - val_loss: 0.0329
Epoch 173/200
90/90 - 19s - loss: 0.0342 - val_loss: 0.0321
Epoch 174/200
90/90 - 19s - loss: 0.0339 - val_loss: 0.0323
Epoch 175/200
90/90 - 19s - loss: 0.0336 - val_loss: 0.0320
Epoch 176/200
90/90 - 19s - loss: 0.0331 - val_loss: 0.0318
Epoch 177/200
90/90 - 19s - loss: 0.0329 - val_loss: 0.0312
Epoch 178/200
90/90 - 19s - loss: 0.0325 - val_loss: 0.0309
Epoch 179/200
90/90 - 19s - loss: 0.0321 - val_loss: 0.0306
Epoch 180/200
90/90 - 19s - loss: 0.0319 - val_loss: 0.0305
Epoch 181/200
90/90 - 19s - loss: 0.0316 - val_loss: 0.0307
Epoch 182/200
90/90 - 19s - loss: 0.0313 - val_loss: 0.0297
Epoch 183/200
90/90 - 19s - loss: 0.0312 - val_loss: 0.0293
Epoch 184/200
90/90 - 19s - loss: 0.0307 - val_loss: 0.0293
Epoch 185/200
90/90 - 19s - loss: 0.0305 - val_loss: 0.0291
Epoch 186/200
90/90 - 19s - loss: 0.0302 - val_loss: 0.0288
Epoch 187/200
90/90 - 19s - loss: 0.0298 - val_loss: 0.0283
Epoch 188/200
90/90 - 19s - loss: 0.0296 - val_loss: 0.0280
Epoch 189/200
90/90 - 19s - loss: 0.0293 - val_loss: 0.0280
Epoch 190/200
90/90 - 19s - loss: 0.0292 - val_loss: 0.0277
Epoch 191/200
90/90 - 19s - loss: 0.0288 - val_loss: 0.0277
Epoch 192/200
90/90 - 19s - loss: 0.0287 - val_loss: 0.0275
Epoch 193/200
90/90 - 19s - loss: 0.0285 - val_loss: 0.0270
Epoch 194/200
90/90 - 19s - loss: 0.0282 - val_loss: 0.0270
Epoch 195/200
90/90 - 19s - loss: 0.0280 - val_loss: 0.0270
Epoch 196/200
90/90 - 19s - loss: 0.0278 - val_loss: 0.0267
Epoch 197/200
90/90 - 19s - loss: 0.0276 - val_loss: 0.0261
Epoch 198/200
90/90 - 19s - loss: 0.0273 - val_loss: 0.0260
Epoch 199/200
90/90 - 19s - loss: 0.0270 - val_loss: 0.0257
Epoch 200/200
90/90 - 19s - loss: 0.0268 - val_loss: 0.0257

<tensorflow.python.keras.callbacks.History at 0x7f4dfc555c18>

```
</div>
---
## Test the model on one movie

Feed it with the first 7 positions and then
predict the new positions.



```python
movie_index = 1004
track = noisy_movies[movie_index][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# And then compare the predictions
# to the ground truth
track2 = noisy_movies[movie_index][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, "Predictions !", fontsize=20, color="w")
    else:
        ax.text(1, 3, "Initial trajectory", fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, "Ground truth", fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[movie_index][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig("%i_animate.png" % (i + 1))

```


![png](/img/examples/vision/conv_lstm/conv_lstm_11_0.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_1.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_2.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_3.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_4.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_5.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_6.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_7.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_8.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_9.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_10.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_11.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_12.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_13.png)



![png](/img/examples/vision/conv_lstm/conv_lstm_11_14.png)


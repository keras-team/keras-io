# Timeseries anomaly detection using an Autoencoder

**Author:** [pavithrasv](https://github.com/pavithrasv)<br>
**Date created:** 2020/05/31<br>
**Last modified:** 2020/05/31<br>
**Description:** Detect anomalies in a timeseries using an Autoencoder.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_anomaly_detection.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/timeseries/timeseries_anomaly_detection.py)



---
## Introduction

This script demonstrates how you can use a reconstruction convolutional
autoencoder model to detect anomalies in timeseries data.


---
## Setup



```python
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import dates as md

```

---
## Load the data

We will use the [Numenta Anomaly Benchmark(NAB)](
https://www.kaggle.com/boltzmannbrain/nab) dataset. It provides artifical
timeseries data containing labeled anomalous periods of behavior. Data are
ordered, timestamped, single-valued metrics.

We will use the `art_daily_small_noise.csv` file for training and the
`art_daily_jumpsup.csv` file for testing. The simplicity of this dataset
allows us to demonstrate anomaly detection effectively.



```python
master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
df_small_noise_url = master_url_root + df_small_noise_url_suffix
df_small_noise = pd.read_csv(df_small_noise_url)

df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
df_daily_jumpsup = pd.read_csv(df_daily_jumpsup_url)

```

---
## Quick look at the data



```python
print(df_small_noise.head())

print(df_daily_jumpsup.head())

```

<div class="k-default-codeblock">
```
             timestamp      value
0  2014-04-01 00:00:00  18.324919
1  2014-04-01 00:05:00  21.970327
2  2014-04-01 00:10:00  18.624806
3  2014-04-01 00:15:00  21.953684
4  2014-04-01 00:20:00  21.909120
             timestamp      value
0  2014-04-01 00:00:00  19.761252
1  2014-04-01 00:05:00  20.500833
2  2014-04-01 00:10:00  19.961641
3  2014-04-01 00:15:00  21.490266
4  2014-04-01 00:20:00  20.187739

```
</div>
---
## Visualize the data



```python

def plot_dates_values(data):
    dates = data["timestamp"].to_list()
    values = data["value"].to_list()
    dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(dates, values)
    plt.show()


```

### Timeseries data without anomalies

We will use the following data for training.



```python
plot_dates_values(df_small_noise)

```

<div class="k-default-codeblock">
```
/usr/local/lib/python3.7/site-packages/pandas/plotting/_converter.py:129: FutureWarning: Using an implicitly registered datetime converter for a matplotlib plotting method. The converter was registered by pandas on import. Future versions of pandas will require you to explicitly register matplotlib converters.
```
</div>
    
<div class="k-default-codeblock">
```
To register the converters:
	>>> from pandas.plotting import register_matplotlib_converters
	>>> register_matplotlib_converters()
  warnings.warn(msg, FutureWarning)

```
</div>
![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_11_1.png)


### Timeseries data with anomalies

We will use the following data for testing and see if the sudden jump up in the
data is detected as an anomaly.



```python
plot_dates_values(df_daily_jumpsup)

```


![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_13_0.png)


---
## Prepare training data

Get data values from the training timeseries data file and normalize the
`value` data. We have a `value` for every 5 mins for 14 days.
*   24 * 60 / 5 = **288 timesteps per day**
*   288 * 14 = **4032 data points** in total



```python

def get_value_from_df(df):
    return df.value.to_list()


def normalize(values):
    mean = np.mean(values)
    values -= mean
    std = np.std(values)
    values /= std
    return values, mean, std


# Get the `value` column from the training dataframe.
training_value = get_value_from_df(df_small_noise)

# Normalize `value` and save the mean and std we get,
# for normalizing test data.
training_value, training_mean, training_std = normalize(training_value)
len(training_value)

```




<div class="k-default-codeblock">
```
4032

```
</div>
### Create sequences
Create sequences combining `TIME_STEPS` contiguous data values from the
training data.



```python
TIME_STEPS = 288


def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    # Convert 2D sequences into 3D as we will be feeding this into
    # a convolutional layer.
    return np.expand_dims(output, axis=2)


x_train = create_sequences(training_value)
print("Training input shape: ", x_train.shape)

```

<div class="k-default-codeblock">
```
Training input shape:  (3744, 288, 1)

```
</div>
---
## Build a model

We will build a convolutional reconstruction autoencoder model. The model will
take input of shape `(batch_size, sequence_length, num_features)` and return
output of the same shape. In this case, `sequence_length` is 288 and
`num_features` is 1.



```python
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

```

<div class="k-default-codeblock">
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, 144, 32)           256       
_________________________________________________________________
dropout (Dropout)            (None, 144, 32)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 72, 16)            3600      
_________________________________________________________________
conv1d_transpose (Conv1DTran (None, 144, 16)           1808      
_________________________________________________________________
dropout_1 (Dropout)          (None, 144, 16)           0         
_________________________________________________________________
conv1d_transpose_1 (Conv1DTr (None, 288, 32)           3616      
_________________________________________________________________
conv1d_transpose_2 (Conv1DTr (None, 288, 1)            225       
=================================================================
Total params: 9,505
Trainable params: 9,505
Non-trainable params: 0
_________________________________________________________________

```
</div>
---
## Train the model

Please note that we are using `x_train` as both the input and the target
since this is a reconstruction model.



```python
history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

```

<div class="k-default-codeblock">
```
Epoch 1/50
27/27 [==============================] - 1s 34ms/step - loss: 0.4540 - val_loss: 0.0622
Epoch 2/50
27/27 [==============================] - 1s 29ms/step - loss: 0.0748 - val_loss: 0.0417
Epoch 3/50
27/27 [==============================] - 1s 30ms/step - loss: 0.0557 - val_loss: 0.0362
Epoch 4/50
27/27 [==============================] - 1s 30ms/step - loss: 0.0482 - val_loss: 0.0313
Epoch 5/50
27/27 [==============================] - 1s 31ms/step - loss: 0.0425 - val_loss: 0.0281
Epoch 6/50
27/27 [==============================] - 1s 31ms/step - loss: 0.0378 - val_loss: 0.0245
Epoch 7/50
27/27 [==============================] - 1s 32ms/step - loss: 0.0341 - val_loss: 0.0229
Epoch 8/50
27/27 [==============================] - 1s 32ms/step - loss: 0.0312 - val_loss: 0.0222
Epoch 9/50
27/27 [==============================] - 1s 32ms/step - loss: 0.0288 - val_loss: 0.0203
Epoch 10/50
27/27 [==============================] - 1s 35ms/step - loss: 0.0269 - val_loss: 0.0199
Epoch 11/50
27/27 [==============================] - 1s 33ms/step - loss: 0.0253 - val_loss: 0.0205
Epoch 12/50
27/27 [==============================] - 1s 35ms/step - loss: 0.0239 - val_loss: 0.0188
Epoch 13/50
27/27 [==============================] - 1s 34ms/step - loss: 0.0228 - val_loss: 0.0197
Epoch 14/50
27/27 [==============================] - 1s 37ms/step - loss: 0.0218 - val_loss: 0.0187
Epoch 15/50
27/27 [==============================] - 1s 36ms/step - loss: 0.0208 - val_loss: 0.0193
Epoch 16/50
27/27 [==============================] - 1s 34ms/step - loss: 0.0200 - val_loss: 0.0188
Epoch 17/50
27/27 [==============================] - 1s 34ms/step - loss: 0.0193 - val_loss: 0.0175
Epoch 18/50
27/27 [==============================] - 1s 36ms/step - loss: 0.0186 - val_loss: 0.0177
Epoch 19/50
27/27 [==============================] - 1s 35ms/step - loss: 0.0179 - val_loss: 0.0178
Epoch 20/50
27/27 [==============================] - 1s 36ms/step - loss: 0.0172 - val_loss: 0.0181
Epoch 21/50
27/27 [==============================] - 1s 38ms/step - loss: 0.0167 - val_loss: 0.0174
Epoch 22/50
27/27 [==============================] - 1s 36ms/step - loss: 0.0162 - val_loss: 0.0161
Epoch 23/50
27/27 [==============================] - 1s 37ms/step - loss: 0.0156 - val_loss: 0.0172
Epoch 24/50
27/27 [==============================] - 1s 39ms/step - loss: 0.0151 - val_loss: 0.0151
Epoch 25/50
27/27 [==============================] - 1s 38ms/step - loss: 0.0145 - val_loss: 0.0149
Epoch 26/50
27/27 [==============================] - 1s 39ms/step - loss: 0.0140 - val_loss: 0.0150
Epoch 27/50
27/27 [==============================] - 1s 39ms/step - loss: 0.0135 - val_loss: 0.0138
Epoch 28/50
27/27 [==============================] - 1s 39ms/step - loss: 0.0130 - val_loss: 0.0129
Epoch 29/50
27/27 [==============================] - 1s 40ms/step - loss: 0.0124 - val_loss: 0.0121
Epoch 30/50
27/27 [==============================] - 1s 41ms/step - loss: 0.0119 - val_loss: 0.0113
Epoch 31/50
27/27 [==============================] - 1s 41ms/step - loss: 0.0115 - val_loss: 0.0110
Epoch 32/50
27/27 [==============================] - 1s 41ms/step - loss: 0.0111 - val_loss: 0.0107
Epoch 33/50
27/27 [==============================] - 1s 41ms/step - loss: 0.0108 - val_loss: 0.0098
Epoch 34/50
27/27 [==============================] - 1s 39ms/step - loss: 0.0105 - val_loss: 0.0102
Epoch 35/50
27/27 [==============================] - 1s 45ms/step - loss: 0.0101 - val_loss: 0.0096
Epoch 36/50
27/27 [==============================] - 1s 45ms/step - loss: 0.0098 - val_loss: 0.0092
Epoch 37/50
27/27 [==============================] - 1s 45ms/step - loss: 0.0096 - val_loss: 0.0090
Epoch 38/50
27/27 [==============================] - 1s 45ms/step - loss: 0.0093 - val_loss: 0.0085
Epoch 39/50
27/27 [==============================] - 1s 43ms/step - loss: 0.0091 - val_loss: 0.0079
Epoch 40/50
27/27 [==============================] - 1s 44ms/step - loss: 0.0088 - val_loss: 0.0084
Epoch 41/50
27/27 [==============================] - 1s 46ms/step - loss: 0.0086 - val_loss: 0.0078
Epoch 42/50
27/27 [==============================] - 1s 45ms/step - loss: 0.0084 - val_loss: 0.0079
Epoch 43/50
27/27 [==============================] - 1s 44ms/step - loss: 0.0082 - val_loss: 0.0073
Epoch 44/50
27/27 [==============================] - 1s 46ms/step - loss: 0.0080 - val_loss: 0.0073
Epoch 45/50
27/27 [==============================] - 1s 46ms/step - loss: 0.0078 - val_loss: 0.0072
Epoch 46/50
27/27 [==============================] - 1s 48ms/step - loss: 0.0076 - val_loss: 0.0072
Epoch 47/50
27/27 [==============================] - 1s 49ms/step - loss: 0.0075 - val_loss: 0.0070
Epoch 48/50
27/27 [==============================] - 1s 46ms/step - loss: 0.0073 - val_loss: 0.0067
Epoch 49/50
27/27 [==============================] - 1s 48ms/step - loss: 0.0071 - val_loss: 0.0068
Epoch 50/50
27/27 [==============================] - 1s 49ms/step - loss: 0.0070 - val_loss: 0.0061

```
</div>
Let's plot training and validation loss to see how the training went.



```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

```




<div class="k-default-codeblock">
```
<matplotlib.legend.Legend at 0x15cbff3d0>

```
</div>
![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_23_1.png)


---
## Detecting anomalies

We will detect anomalies by determining how well our model can reconstruct
the input data.


1.   Find MAE loss on training samples.
2.   Find max MAE loss value. This is the worst our model has performed trying
to reconstruct a sample. We will make this the `threshold` for anomaly
detection.
3.   If the reconstruction loss for a sample is greater than this `threshold`
value then we can infer that the model is seeing a pattern that it isn't
familiar with. We will label this sample as an `anomaly`.




```python
# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

```


![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_25_0.png)


<div class="k-default-codeblock">
```
Reconstruction error threshold:  0.06783195782739782

```
</div>
### Compare recontruction

Just for fun, let's see how our model has recontructed the first sample.
This is the 288 timesteps from day 1 of our training dataset.



```python
# Checking how the first sequence is learnt
plt.plot(x_train[0])
plt.show()
plt.plot(x_train_pred[0])
plt.show()

```


![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_27_0.png)



![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_27_1.png)


### Prepare test data



```python

def normalize_test(values, mean, std):
    values -= mean
    values /= std
    return values


test_value = get_value_from_df(df_daily_jumpsup)
test_value = normalize_test(test_value, training_mean, training_std)
plt.plot(test_value.tolist())
plt.show()

# Create sequences from test values.
x_test = create_sequences(test_value)
print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = (test_mae_loss > threshold).tolist()
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

```


![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_29_0.png)


<div class="k-default-codeblock">
```
Test input shape:  (3744, 288, 1)

```
</div>
![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_29_2.png)


<div class="k-default-codeblock">
```
Number of anomaly samples:  416
Indices of anomaly samples:  (array([ 216,  218,  777,  792,  793,  794,  973, 1657, 1943, 1944, 1945,
       1946, 2121, 2123, 2125, 2519, 2520, 2521, 2522, 2697, 2699, 2701,
       2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712,
       2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723,
       2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734,
       2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745,
       2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756,
       2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767,
       2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778,
       2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789,
       2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800,
       2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811,
       2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822,
       2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833,
       2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844,
       2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855,
       2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866,
       2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877,
       2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888,
       2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899,
       2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910,
       2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921,
       2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932,
       2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943,
       2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954,
       2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965,
       2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976,
       2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987,
       2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998,
       2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009,
       3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020,
       3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031,
       3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042,
       3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053,
       3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064,
       3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075,
       3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086,
       3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095]),)

```
</div>
---
## Plot anomalies

We now know the samples of the data which are anomalies. With this, we will
find the corresponding `timestamps` from the original test data. We will be
using the following method to do that:

Let's say time_steps = 3 and we have 10 training values. Our `x_train` will
look like this:

- 0, 1, 2
- 1, 2, 3
- 2, 3, 4
- 3, 4, 5
- 4, 5, 6
- 5, 6, 7
- 6, 7, 8
- 7, 8, 9

All except the initial and the final time_steps-1 data values, will appear in
`time_steps` number of samples. So, if we know that the samples
[(3, 4, 5), (4, 5, 6), (5, 6, 7)] are anomalies, we can say that the data point
5 is an anomaly.



```python
# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(test_value) - TIME_STEPS + 1):
    time_series = range(data_idx - TIME_STEPS + 1, data_idx)
    if all([anomalies[j] for j in time_series]):
        anomalous_data_indices.append(data_idx)

```

Let's overlay the anomalies on the original test data plot.



```python
df_subset = df_daily_jumpsup.iloc[anomalous_data_indices, :]
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=25)
ax = plt.gca()
xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
ax.xaxis.set_major_formatter(xfmt)

dates = df_daily_jumpsup["timestamp"].to_list()
dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
values = df_daily_jumpsup["value"].to_list()
plt.plot(dates, values, label="test data")

dates = df_subset["timestamp"].to_list()
dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
values = df_subset["value"].to_list()
plt.plot(dates, values, label="anomalies", color="r")

plt.legend()
plt.show()

```


![png](/img/examples/timeseries/timeseries_anomaly_detection/timeseries_anomaly_detection_33_0.png)


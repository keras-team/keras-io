"""
Title: Multivariate Time Series Forecasting using LSTM
Author: [Rohit Sahoo](https://www.linkedin.com/in/rohit-sahoo/)
Date created: 2021/07/17
Last modified: 2021/07/17
Description: Multivariate Time Series Forecasting on stock market data using LSTM.
"""

"""
## Introduction
Time-Series forecasting means predicting the future dependent variable (y) based on the
past independent variable (x). If the model predicts a dependent variable (y) based on
one independent variable (x), it is called univariate forecasting. Whereas, In
Multivariate forecasting, the model predicts a dependent variable (y) based on more than
one independent variable (x).

This Example implements a time series model for Google's stock market data. In this
example, Multivariate time series forecasting is performed by determining the opening
price of the stock using the historical opening, closing, highest, lowest and the
adjusted closing price. This example uses the LSTM (Long Short-Term Memory) model to
predict the opening price of the stock by taking the input shape defined by the window
length and these 5 features.

A univariable forecast model reduces this complexity to a minimum – a single factor and
ignores the other dimensions such as prediction of the opening price of the stock is
based only on the historical opening price. Whereas, A multivariate stock market
prediction model can consider the relationship between multiple variables. They offer a
more detailed abstraction of reality than univariate models. Multivariate models thus
tend to provide more accurate predictions than univariate models.
"""

"""
## Setup
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

"""
## Load the Data
"""

"""
This example uses Google's stock market dataset downloaded from [Yahoo
Finance](https://in.finance.yahoo.com/quote/GOOG/history?period1=1092960000&period2=1594944000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)


**Time-Frame Considered:** 16 Years of data starting from 2004/08/19 to 2020/07/17.

**Structure of Dataset**

1. Date - specifies trading date
2. Open - opening price
3. High - maximum price during the day
4. Low - minimum price during the day
5. Close - close price adjusted for splits
6. Adj Close - adjusted close price adjusted for both dividends and splits
7. Volume - the number of shares that changed hands during a given day

"""

df = pd.read_csv(
    "https://raw.githubusercontent.com/rohit-sahoo/Multivariate_"
    + "Timeseries_Forecasting_using_LSTM/master/GOOG.csv"
)

"""
Quick look at the Google Stock Market Dataset.
"""

df.head()

"""
Check the data type of each column in dataframe.
"""

df.dtypes

"""
The date column provided in the dataset is of object type, it has to be changed into
datetime format.
"""

df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

df.describe()  # concise summary of the dataframe

"""
**Visualizing the features**
"""

mpl.rcParams["figure.figsize"] = (10, 8)
mpl.rcParams["axes.grid"] = False
df.set_index("Date")[["Open", "High", "Low", "Close", "Adj Close"]].plot(subplots=True)

"""
Taking 5 features as the input to the time series - open, high, low, close, Adjusted close
"""

df_input = df[["Open", "High", "Low", "Close", "Adj Close",]]

"""
## Data Preprocessing
### 1. Standardization of data

**StandardScaler()**

LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be
normalized.

Since there are 5 features used in this example for prediction and have different scales,
they are Standardized to have a common scale while building the model.

StandardScaler() will normalize the features i.e. each column of X, INDIVIDUALLY, so that
each column/feature/variable will have mean = 0 and standard deviation = 1
"""

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_input)
data_scaled

"""
**Set the features and target for the model**

1. features = Open, High, Low, Close, Adj close
2. target = Open
"""

features = data_scaled
target = data_scaled[:, 0]  # Target Column - Open Price

"""
Split the data into training and testing.
"""

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=123, shuffle=False
)

print(x_train.shape)

"""
**Training Data**

1. Size: 3004
2. features: 5
"""

print(x_test.shape)

"""
**Testing Data**

1. Size: 1002
2. features: 5
"""

"""
### 2. TimeseriesGenerator

**TimeseriesGenerator()** is used to automatically transform both univariate and
multivariate time series data into samples, ready to train deep learning models.

In this example, Keras TimeseriesGenerator is used for preparing time series data for
modelling with deep learning methods.

Arguments that are passed to TimeseriesGenerator:
1. features: Passing the scaled multivariate data
2. target: Passing the scaled target column
3. length: It is the window_length
4. Sampling_rate: Period between successive individual timesteps within sequences.
5. batch_size: Number of time-series samples in each batch

To know more about TimeseriesGenerator, check the Keras documentation:
[TimeseriesGenerator](https://keras.io/api/preprocessing/timeseries/)
"""

# This code block is for the understanding the output of TimeseriesGenerator
print("Input Features\n", data_scaled[0:3])
print("\nTimeseriesGenerator")
print(TimeseriesGenerator(features, target, length=2, sampling_rate=1, batch_size=1)[0])

"""
From the above output, it can be observed that the opening price of 3rd day is made as
the target value for the input features of first two days.
"""

"""
## Build the model
"""

"""
### 1. Set the parameters
"""

win_length = 60  # window length
batch_size = 62
num_features = 5
train_generator = TimeseriesGenerator(
    x_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size
)
test_generator = TimeseriesGenerator(
    x_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size
)

"""
### 2. Build the model using LSTM
"""

"""
**Leaky ReLU**

Leaky ReLU function is an improved version of the ReLU activation function. As for the
ReLU activation function, the gradient is 0 for all the values of inputs that are less
than zero, which would deactivate the neurons in that region and may cause a dying ReLU
problem.

Leaky ReLU is defined to address this problem. Instead of defining the ReLU activation
function as 0 for negative values of inputs(x), we define it as an extremely small linear
component of x. Here is the formula for this activation function:

f(x)=max(0.01*x , x)

For reference, Relu formula: f(x) = max(0,x)

This function returns x if it receives any positive input, but for any negative value of
x, it returns a really small value which is 0.01 times x.

**Why LSTM?**

Recurrent neural networks are much more flexible and much better suited to time series
forecasting than the linear models usually applied.

Recurrent neural networks, of which LSTMs (“long short-term memory” units) are the most
powerful and well-known subset, are a type of artificial neural network designed to
recognize patterns in sequences of data, such as numerical times series data.
"""

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.LSTM(
        128, input_shape=(win_length, num_features), return_sequences=True
    )
)
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

"""
**summary of the model**
"""

model.summary()

"""
## Train the model
"""

model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=tf.optimizers.Adam(),
    metrics=[tf.metrics.MeanAbsoluteError()],
)

history = model.fit(
    train_generator, epochs=50, validation_data=test_generator, shuffle=False
)

"""
## Evaluate model on Test data
"""

"""
### 1. Visualize the training and validation loss
"""

plt.plot(history.history["mean_absolute_error"], label="Training MAE")
plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")
plt.legend()
plt.show()

"""
### 2. Test Loss and MAE
"""

test_loss, test_mae = model.evaluate(test_generator, verbose=0)
print("Test Loss:", test_loss)
print("Test MAE:", test_mae)

"""
## Predictions
"""

predictions = model.predict(test_generator)

print(predictions.shape[0])

"""
The shape of output is 942 and not 1002 when compared with x_test, since the first 60
days are used to determine the next day.
"""

# Consider only the 942 values after the first 60 values
print(x_test[:, 1:][win_length:])

"""
Concatenate the prediction dataframe with the x_test
"""

df_pred = pd.concat(
    [pd.DataFrame(predictions), pd.DataFrame(x_test[:, 1:][win_length:])], axis=1
)

df_pred

"""
**inverse_transform()** is used to scale back the data to the original representation.
"""

# To get the original values, the inverse_transform has to be performed.
in_trans = scaler.inverse_transform(df_pred)

print(in_trans)

"""
Take only last 942 rows to compare the actual opening and the predicted opening price,
since first 60 days are used to predict the upcoming day.
"""

df_final = df_input[predictions.shape[0] * -1 :]

# Add the Predicted Open price into the final dataframe
predicted_open = in_trans[:, 0].tolist()
df_final.insert(5, "Predicted Open", predicted_open)

df_final

"""
## Visualize the Predictions
"""

df_final[["Open", "Predicted Open"]].plot()

"""
Title: Sentiment analysis with LSTM on the IMDB dataset
Author:Madhur Jain
Date created: 2025/11/19
Last Modified: 2025/11/19
Description: A simple LSTM-based sentiment classifier trained on IMDB text reviews.
"""

"""
## Introduction

LSTM refers to Long short term memories, that is while predicting it not only keeps short term memory but also long term memory
LSTM uses sigmoid activation functions and tan-h activation functions:
    The Sigmoid fn. ranges the values from 0 to 1,
    tan-h function ranges the values from -1 to 1.
Doesn't let Gradient Descent of long term memory to vanish or short term memory to completely explode.
It contains 3 stages: 
    1st stage: Determines what % of long term memory is remembered- c/a Forget Gate
    2nd stage: Determines how we would update long-term memory- c/a Input Gate
    3rd stage: Updates short term memory and it is the output of the entire stage- c/a Output Gate

If you wanna know more deeply about it, I would recommend to watch Stanford Online: statistacl analysis with Python course lectures available on Youtube (for free)

"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers
from keras.models import Sequential

"""
## Load the dataset
get the kAGGLE.json from your kaggle account->settings->create new token
"""
kaggle_dictionary = json.load(open("kaggle.json"))  #converts json object to python dictionary
#Setup Kaggle collection as env vars
kaggle_dictionary.keys()

os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]

# unzip the dataset file
with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
  zip_ref.extractall()

#loading the dataset
data = pd.read_csv("/content/IMDB Dataset.csv")

data.shape

data.info()

data.head()

data["sentiment"].value_counts()

data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

data.head()

data["sentiment"].value_counts()


"""
## Splitting into Training and test set 
"""
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(train_data.shape)
print(test_data.shape)


"""
## Data Processing
"""
#Tokenize text data
# for text data one have to tokenize(convert words to integer in short) the data and stuff
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)

print(X_train)
print(X_test)

Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]

print(Y_train)
print(Y_test)


"""
## LSTM (Long Short Term Memory) Model
"""
# build the model

model = Sequential()  #sequential model
#add layers
model.add(Embedding(input_dim=5000, output_dim=128, input_shape=(200,)))
model.add(LSTM(128, dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.summary()

# compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
"""
## Training the Model
"""
model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)

"""
## Model Evaluation
"""
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

"""
### Predicting Values
"""
def predict_sentiment(review):
  # tokenize and pad the review
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment


#  examples

new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
#===================================================================================#
new_review = "This movie was not that good"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
#===================================================================================#
new_review = "Great movie but could have added a better action scene"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
#===================================================================================#
new_review = "Mid movie"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
# ==================================================================================#
new_review = "I laughing while shitting damn what a watch"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
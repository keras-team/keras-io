"""
Title: Sentiment analysis with LSTM on the IMDB dataset
Author: Madhur Jain
Date created: 2025/11/19
Last Modified: 2025/11/24 (Refactored)
Description: A simple LSTM-based sentiment classifier trained on IMDB text reviews,
             demonstrating the modern Keras TextVectorization layer.
"""

import os
import shutil # For removing the 'unsup' directory
import keras
import tensorflow as tf # Needed for tf.data.Dataset
import pandas as pd
from keras import layers
from keras.models import Sequential
from keras.layers import TextVectorization # Modern Keras text preprocessing

# Set the Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"


## Load the dataset 💾

# URL for the raw IMDB dataset (aclImdb_v1.tar.gz)
data_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Use keras.utils.get_file to download and extract the dataset
# This ensures portability across different environments (Colab, local machine, etc.)
dataset_path = keras.utils.get_file(
    "aclImdb_v1.tar.gz", data_url, untar=True, cache_dir=".", cache_subdir=""
)
main_dir = os.path.join(dataset_path, "aclImdb")
train_dir = os.path.join(main_dir, "train")
test_dir = os.path.join(main_dir, "test")

# The IMDB dataset includes an 'unsup' (unsupervised) directory in train that should be ignored
remove_dir = os.path.join(train_dir, "unsup")
if os.path.exists(remove_dir):
    shutil.rmtree(remove_dir)


# Helper function to load the data from the extracted files into a DataFrame
def load_data_from_dir(directory):
    """Loads text reviews and their labels from a directory."""
    reviews, sentiments = [], []
    for sentiment_type, sentiment_label in [("pos", 1), ("neg", 0)]:
        sentiment_dir = os.path.join(directory, sentiment_type)
        for fname in os.listdir(sentiment_dir):
            if fname.endswith(".txt"):
                # Use a standard, safe encoding
                with open(os.path.join(sentiment_dir, fname), encoding="utf-8") as f:
                    reviews.append(f.read())
                    sentiments.append(sentiment_label)
    return pd.DataFrame({"review": reviews, "sentiment": sentiments})

# Load dataframes directly
train_df = load_data_from_dir(train_dir)
test_df = load_data_from_dir(test_dir)

# The data is already split into 25k train and 25k test reviews
# Separate the features (X) and labels (Y)
X_train_text = train_df["review"]
Y_train = train_df["sentiment"]
X_test_text = test_df["review"]
Y_test = test_df["sentiment"]

print(f"Training samples: {len(X_train_text)}, Test samples: {len(X_test_text)}")
# 

---

## 🧠 Data Processing with TextVectorization

# Hyperparameters for the TextVectorization layer
max_features = 5000  # Only consider the top N words
sequence_length = 200 # Pad/truncate all sequences to a fixed length
embedding_dim = 128  # Size of the output vector for each word

# 1. Create the TextVectorization layer
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode="int", # Outputs integer indices
    output_sequence_length=sequence_length,
)

# 2. Adapt the layer to the training text
# Adapt() builds the vocabulary based on the training data.
# We convert the Pandas Series to a batched TensorFlow Dataset for efficient adaptation.
text_ds = tf.data.Dataset.from_tensor_slices(X_train_text.values).batch(128)
vectorize_layer.adapt(text_ds)

# Optional: Inspect the vocabulary
# print("Vocabulary size:", len(vectorize_layer.get_vocabulary()))
# print("Top 10 words in vocabulary:", vectorize_layer.get_vocabulary()[:10])

# The text and labels are now ready to be passed directly to model.fit

---

## 🏗️ LSTM Model (End-to-End)

# The TextVectorization layer is included directly in the Sequential model,
# making the model "end-to-end" (it accepts raw strings).

model = Sequential([
    # 1. Input: TextVectorization layer (accepts raw string, outputs integer sequences)
    vectorize_layer, 
    # 2. Embedding Layer: Maps integer indices to dense vectors
    layers.Embedding(input_dim=max_features, output_dim=embedding_dim, mask_zero=True),
    # 3. LSTM Layer: Recurrent layer for sequential processing
    layers.LSTM(128, dropout=0.2),
    # 4. Dense Output Layer: Binary classification with sigmoid activation
    layers.Dense(1, activation="sigmoid")
])

model.summary()
# 

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

---

## 🏃 Training and Evaluation

print("\n## Training the Model")
# The model can now be trained by passing the raw text and labels.
model.fit(
    X_train_text, # Raw text input
    Y_train,      # Integer labels
    epochs=5, 
    batch_size=64, 
    validation_split=0.2
)

print("\n## Model Evaluation")
# Note: For evaluation, use the raw text from the test set
loss, accuracy = model.evaluate(X_test_text, Y_test) 
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

---

## 🔮 Predicting Values (Simplified Inference)

def predict_sentiment(review):
    """Predicts sentiment for a raw text review using the end-to-end model."""
    # The model accepts a list/array of raw strings directly
    prediction = model.predict([review])
    
    # Sigmoid output is a probability
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    probability = prediction[0][0]
    return sentiment, probability

# Examples
print("\n### Predicting Values")
examples = [
    "This movie was fantastic. I loved it.",
    "This movie was not that good",
    "Great movie but could have added a better action scene",
    "Mid movie"
]

for review in examples:
    sentiment, prob = predict_sentiment(review)
    print(f"Review: '{review[:30]}...' -> Sentiment: {sentiment} ({prob:.2f})")

# Clean up the downloaded directory
if os.path.exists(main_dir):
    shutil.rmtree(main_dir)
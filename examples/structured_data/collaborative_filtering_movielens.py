"""
Title: Collaborative Filtering for Movie Recommendations
Author: Siddhartha Banerjee
Date created: 2020/05/24
Last modified: 2020/05/24
Description: Recommending movies using user and movie embeddings trained using Movielens small dataset.
"""
"""
## Introduction

This example looks at
[Collaborative filtering using Movielens dataset](https://www.kaggle.com/c/movielens-100k)
to recommend movies to users.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from pathlib import Path
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Dot,
    Add,
    Embedding,
    Activation,
    Lambda,
    Dense,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

"""
## First, load the data and apply some pre-processing
"""

# Download the actual data from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# use the ratings.csv file

data_directory = Path("/Users/siddban/Documents/ml-latest-small/")

ratings_file = data_directory / "ratings.csv"

df = pd.read_csv(ratings_file)

"""
First, need to perform some preprocessing to encode users and movies to specific indices.
This is done as the movieids and userids are non-sequential in nature.
"""

user_enc = LabelEncoder()
df["user"] = user_enc.fit_transform(df["userId"].values)
n_users = df["user"].nunique()
movie_enc = LabelEncoder()
df["movie"] = movie_enc.fit_transform(df["movieId"].values)
n_movies = df["movie"].nunique()
df["rating"] = df["rating"].values.astype(np.float32)
min_rating = min(df["rating"])
max_rating = max(df["rating"])
print(
    "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
        n_users, n_movies, min_rating, max_rating
    )
)

"""
## Prepare training and validation data
"""

X = df[["user", "movie"]].values
y = df["rating"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""
## Create the model

Let's fix some parameters in the architecture. We will use user embeddings and movie
embeddings, let's keep the dimensions as 50 for both.
"""
EMBEDDING_SIZE = 50


class EmbeddingLayer(layers.Layer):
    def __init__(self, n_items, n_factors):
        super(EmbeddingLayer, self).__init__()
        self.n_items = n_items
        self.n_factors = n_factors
        self.emb = Embedding(
            self.n_items,
            self.n_factors,
            embeddings_initializer="he_normal",
            embeddings_regularizer=l2(1e-6),
        )

    def call(self, x):
        x = self.emb(x)
        x = Reshape((self.n_factors,))(x)
        return x


class RecommenderNet(keras.Model):
    def __init__(self, n_users, n_movies, embedding_size, min_rating, max_rating):
        super(RecommenderNet, self).__init__(name="recnet")
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_size = embedding_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.user_embedding = Embedding(
            n_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=l2(1e-6),
        )
        self.user_bias = Embedding(n_users, 1)
        self.movie_embedding = Embedding(
            n_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=l2(1e-6),
        )
        self.movie_bias = Embedding(n_movies, 1)

    def call(self, inputs):
        u = self.user_embedding(inputs[:, 0])
        ub = self.user_bias(inputs[:, 0])
        m = self.movie_embedding(inputs[:, 1])
        mb = self.movie_bias(inputs[:, 1])
        x = Dot(axes=1)([u, m])
        x = Add()([x, ub, mb])
        x = Activation("sigmoid")(x)
        x = Lambda(lambda x: x * (self.max_rating - self.min_rating) + self.min_rating)(
            x
        )
        return x


inputs = layers.Input(shape=(2,))
recommendation_layer = RecommenderNet(
    n_users, n_movies, EMBEDDING_SIZE, min_rating, max_rating
)
outputs = recommendation_layer(inputs)
model = keras.Model(inputs=inputs, outputs=outputs)
opt = Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer=opt)
model.summary()

"""
## Train the model based on the data split
"""
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(X_test, y_test),
)

"""
## Plot training and test loss
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

"""
## Show top 10 movie recommendations to an user
"""
movie_df = pd.read_csv(data_directory / "movies.csv")

# Let us get an user and see the top recommendations.
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id].movieId.values
movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user)][
    "movieId"
]
movies_not_watched = list(set(movies_not_watched).intersection(set(movie_enc.classes_)))
movies_not_watched = movie_enc.transform(movies_not_watched).reshape(
    len(movies_not_watched), 1
)
user_encoder = user_enc.transform([user_id])
user_movie_array = np.hstack(
    ([[user_id]] * len(movies_not_watched), movies_not_watched)
)
ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = movie_enc.inverse_transform(top_ratings_indices)

print("Here are the top 10 movie recommendations for user: {}".format(user_id))
print("*****" * 10)
print("\n".join(movie_df[movie_df["movieId"].isin(recommended_movie_ids)].title.values))

"""
## Conclusions

As can be seen from the final results, we can recommend new movies to an user

- The validation loss dips until 3rd epoch
- The model shows a validation loss around 0.75
(although the lowest is 0.73 but we did not apply a callback here with Checkpointing) which is pretty good on this dataset.

This is how online video streaming companies provide you recommendations.
"""

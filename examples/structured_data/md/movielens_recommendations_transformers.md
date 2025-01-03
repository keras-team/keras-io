# A Transformer-based recommendation system

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2020/12/30<br>
**Last modified:** 2025/01/03<br>
**Description:** Rating rate prediction using the Behavior Sequence Transformer (BST) model on the Movielens.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/structured_data/ipynb/movielens_recommendations_transformers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/structured_data/movielens_recommendations_transformers.py)



---
## Introduction

This example demonstrates the [Behavior Sequence Transformer (BST)](https://arxiv.org/abs/1905.06874)
model, by Qiwei Chen et al., using the [Movielens dataset](https://grouplens.org/datasets/movielens/).
The BST model leverages the sequential behaviour of the users in watching and rating movies,
as well as user profile and movie features, to predict the rating of the user to a target movie.

More precisely, the BST model aims to predict the rating of a target movie by accepting
the following inputs:

1. A fixed-length *sequence* of `movie_ids` watched by a user.
2. A fixed-length *sequence* of the `ratings` for the movies watched by a user.
3. A *set* of user features, including `user_id`, `sex`, `occupation`, and `age_group`.
4. A *set* of `genres` for each movie in the input sequence and the target movie.
5. A `target_movie_id` for which to predict the rating.

This example modifies the original BST model in the following ways:

1. We incorporate the movie features (genres) into the processing of the embedding of each
movie of the input sequence and the target movie, rather than treating them as "other features"
outside the transformer layer.
2. We utilize the ratings of movies in the input sequence, along with the their positions
in the sequence, to update them before feeding them into the self-attention layer.


Note that this example should be run with TensorFlow 2.4 or higher.

---
## The dataset

We use the [1M version of the Movielens dataset](https://grouplens.org/datasets/movielens/1m/).
The dataset includes around 1 million ratings from 6000 users on 4000 movies,
along with some user features, movie genres. In addition, the timestamp of each user-movie
rating is provided, which allows creating sequences of movie ratings for each user,
as expected by the BST model.

---
## Setup


```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import math
from zipfile import ZipFile
from urllib.request import urlretrieve

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import StringLookup
```

---
## Prepare the data

### Download and prepare the DataFrames

First, let's download the movielens data.

The downloaded folder will contain three data files: `users.dat`, `movies.dat`,
and `ratings.dat`.


```python
urlretrieve("http://files.grouplens.org/datasets/movielens/ml-1m.zip", "movielens.zip")
ZipFile("movielens.zip", "r").extractall()
```

Then, we load the data into pandas DataFrames with their proper column names.


```python
users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    encoding="ISO-8859-1",
    engine="python",
)

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
    encoding="ISO-8859-1",
    engine="python",
)

movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    names=["movie_id", "title", "genres"],
    encoding="ISO-8859-1",
    engine="python",
)
```

Here, we do some simple data processing to fix the data types of the columns.


```python
users["user_id"] = users["user_id"].apply(lambda x: f"user_{x}")
users["age_group"] = users["age_group"].apply(lambda x: f"group_{x}")
users["occupation"] = users["occupation"].apply(lambda x: f"occupation_{x}")

movies["movie_id"] = movies["movie_id"].apply(lambda x: f"movie_{x}")

ratings["movie_id"] = ratings["movie_id"].apply(lambda x: f"movie_{x}")
ratings["user_id"] = ratings["user_id"].apply(lambda x: f"user_{x}")
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))
```

Each movie has multiple genres. We split them into separate columns in the `movies`
DataFrame.


```python
genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime"]
genres += ["Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical"]
genres += ["Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

for genre in genres:
    movies[genre] = movies["genres"].apply(
        lambda values: int(genre in values.split("|"))
    )

```

### Transform the movie ratings data into sequences

First, let's sort the the ratings data using the `unix_timestamp`, and then group the
`movie_id` values and the `rating` values by `user_id`.

The output DataFrame will have a record for each `user_id`, with two ordered lists
(sorted by rating datetime): the movies they have rated, and their ratings of these movies.


```python
ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

ratings_data = pd.DataFrame(
    data={
        "user_id": list(ratings_group.groups.keys()),
        "movie_ids": list(ratings_group.movie_id.apply(list)),
        "ratings": list(ratings_group.rating.apply(list)),
        "timestamps": list(ratings_group.unix_timestamp.apply(list)),
    }
)

```

Now, let's split the `movie_ids` list into a set of sequences of a fixed length.
We do the same for the `ratings`. Set the `sequence_length` variable to change the length
of the input sequence to the model. You can also change the `step_size` to control the
number of sequences to generate for each user.


```python
sequence_length = 4
step_size = 2


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

ratings_data.ratings = ratings_data.ratings.apply(
    lambda ids: create_sequences(ids, sequence_length, step_size)
)

del ratings_data["timestamps"]
```

After that, we process the output to have each sequence in a separate records in
the DataFrame. In addition, we join the user features with the ratings data.


```python
ratings_data_movies = ratings_data[["user_id", "movie_ids"]].explode(
    "movie_ids", ignore_index=True
)
ratings_data_rating = ratings_data[["ratings"]].explode("ratings", ignore_index=True)
ratings_data_transformed = pd.concat([ratings_data_movies, ratings_data_rating], axis=1)
ratings_data_transformed = ratings_data_transformed.join(
    users.set_index("user_id"), on="user_id"
)
ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
    lambda x: ",".join(x)
)
ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
    lambda x: ",".join([str(v) for v in x])
)

del ratings_data_transformed["zip_code"]

ratings_data_transformed.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
)
```

With `sequence_length` of 4 and `step_size` of 2, we end up with 498,623 sequences.

Finally, we split the data into training and testing splits, with 85% and 15% of
the instances, respectively, and store them to CSV files.


```python
random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
train_data = ratings_data_transformed[random_selection]
test_data = ratings_data_transformed[~random_selection]

train_data.to_csv("train_data.csv", index=False, sep="|", header=False)
test_data.to_csv("test_data.csv", index=False, sep="|", header=False)
```

---
## Define metadata


```python
CSV_HEADER = list(ratings_data_transformed.columns)

CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "user_id": list(users.user_id.unique()),
    "movie_id": list(movies.movie_id.unique()),
    "sex": list(users.sex.unique()),
    "age_group": list(users.age_group.unique()),
    "occupation": list(users.occupation.unique()),
}

USER_FEATURES = ["sex", "age_group", "occupation"]

MOVIE_FEATURES = ["genres"]
```

---
## Create `tf.data.Dataset` for training and evaluation


```python

def get_dataset_from_csv(csv_file_path, batch_size, shuffle=True):
    def process(features):
        movie_ids_string = features["sequence_movie_ids"]
        sequence_movie_ids = tf.strings.split(movie_ids_string, ",").to_tensor()

        # The last movie id in the sequence is the target movie.
        features["target_movie_id"] = sequence_movie_ids[:, -1]
        features["sequence_movie_ids"] = sequence_movie_ids[:, :-1]

        ratings_string = features["sequence_ratings"]
        sequence_ratings = tf.strings.to_number(
            tf.strings.split(ratings_string, ","), tf.dtypes.float32
        ).to_tensor()

        # The last rating in the sequence is the target for the model to predict.
        target = sequence_ratings[:, -1]
        features["sequence_ratings"] = sequence_ratings[:, :-1]

        return dict(features), target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        field_delim="|",
        shuffle=shuffle,
    ).map(process)

    return dataset

```

---
## Create model inputs


```python

def create_model_inputs():
    return {
        "user_id": keras.Input(name="user_id", shape=(1,), dtype="string"),
        "sequence_movie_ids": keras.Input(
            name="sequence_movie_ids", shape=(sequence_length - 1,), dtype="string"
        ),
        "target_movie_id": keras.Input(
            name="target_movie_id", shape=(1,), dtype="string"
        ),
        "sequence_ratings": keras.Input(
            name="sequence_ratings", shape=(sequence_length - 1,), dtype=tf.float32
        ),
        "sex": keras.Input(name="sex", shape=(1,), dtype="string"),
        "age_group": keras.Input(name="age_group", shape=(1,), dtype="string"),
        "occupation": keras.Input(name="occupation", shape=(1,), dtype="string"),
    }

```

---
## Encode input features

The `encode_input_features` method works as follows:

1. Each categorical user feature is encoded using `layers.Embedding`, with embedding
dimension equals to the square root of the vocabulary size of the feature.
The embeddings of these features are concatenated to form a single input tensor.

2. Each movie in the movie sequence and the target movie is encoded `layers.Embedding`,
where the dimension size is the square root of the number of movies.

3. A multi-hot genres vector for each movie is concatenated with its embedding vector,
and processed using a non-linear `layers.Dense` to output a vector of the same movie
embedding dimensions.

4. A positional embedding is added to each movie embedding in the sequence, and then
multiplied by its rating from the ratings sequence.

5. The target movie embedding is concatenated to the sequence movie embeddings, producing
a tensor with the shape of `[batch size, sequence length, embedding size]`, as expected
by the attention layer for the transformer architecture.

6. The method returns a tuple of two elements:  `encoded_transformer_features` and
`encoded_other_features`.


```python

def encode_input_features(
    inputs,
    include_user_id=True,
    include_user_features=True,
    include_movie_features=True,
):
    encoded_transformer_features = []
    encoded_other_features = []

    other_feature_names = []
    if include_user_id:
        other_feature_names.append("user_id")
    if include_user_features:
        other_feature_names.extend(USER_FEATURES)

    ## Encode user features
    for feature_name in other_feature_names:
        # Convert the string input values into integer indices.
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        idx = StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)(
            inputs[feature_name]
        )
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f"{feature_name}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder(idx))

    ## Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    ## Create a movie embedding encoder
    movie_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY["movie_id"]
    movie_embedding_dims = int(math.sqrt(len(movie_vocabulary)))
    # Create a lookup to convert string values to integer indices.
    movie_index_lookup = StringLookup(
        vocabulary=movie_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name="movie_index_lookup",
    )
    # Create an embedding layer with the specified dimensions.
    movie_embedding_encoder = layers.Embedding(
        input_dim=len(movie_vocabulary),
        output_dim=movie_embedding_dims,
        name=f"movie_embedding",
    )
    # Create a vector lookup for movie genres.
    genre_vectors = movies[genres].to_numpy()
    movie_genres_lookup = layers.Embedding(
        input_dim=genre_vectors.shape[0],
        output_dim=genre_vectors.shape[1],
        embeddings_initializer=keras.initializers.Constant(genre_vectors),
        trainable=False,
        name="genres_vector",
    )
    # Create a processing layer for genres.
    movie_embedding_processor = layers.Dense(
        units=movie_embedding_dims,
        activation="relu",
        name="process_movie_embedding_with_genres",
    )

    ## Define a function to encode a given movie id.
    def encode_movie(movie_id):
        # Convert the string input values into integer indices.
        movie_idx = movie_index_lookup(movie_id)
        movie_embedding = movie_embedding_encoder(movie_idx)
        encoded_movie = movie_embedding
        if include_movie_features:
            movie_genres_vector = movie_genres_lookup(movie_idx)
            encoded_movie = movie_embedding_processor(
                layers.concatenate([movie_embedding, movie_genres_vector])
            )
        return encoded_movie

    ## Encoding target_movie_id
    target_movie_id = inputs["target_movie_id"]
    encoded_target_movie = encode_movie(target_movie_id)

    ## Encoding sequence movie_ids.
    sequence_movies_ids = inputs["sequence_movie_ids"]
    encoded_sequence_movies = encode_movie(sequence_movies_ids)
    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=sequence_length,
        output_dim=movie_embedding_dims,
        name="position_embedding",
    )
    positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # Retrieve sequence ratings to incorporate them into the encoding of the movie.
    sequence_ratings = inputs["sequence_ratings"]
    sequence_ratings = keras.ops.expand_dims(sequence_ratings, -1)
    # Add the positional encoding to the movie encodings and multiply them by rating.
    encoded_sequence_movies_with_poistion_and_rating = layers.Multiply()(
        [(encoded_sequence_movies + encodded_positions), sequence_ratings]
    )

    # Construct the transformer inputs.
    for i in range(sequence_length - 1):
        feature = encoded_sequence_movies_with_poistion_and_rating[:, i, ...]
        feature = keras.ops.expand_dims(feature, 1)
        encoded_transformer_features.append(feature)
    encoded_transformer_features.append(encoded_target_movie)

    encoded_transformer_features = layers.concatenate(
        encoded_transformer_features, axis=1
    )

    return encoded_transformer_features, encoded_other_features

```

---
## Create a BST model


```python
include_user_id = False
include_user_features = False
include_movie_features = False

hidden_units = [256, 128]
dropout_rate = 0.1
num_heads = 3


def create_model():
    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(
        inputs, include_user_id, include_user_features, include_movie_features
    )

    # Create a multi-headed attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other features.
    if other_features is not None:
        features = layers.concatenate(
            [features, layers.Reshape([other_features.shape[-1]])(other_features)]
        )

    # Fully-connected layers.
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = create_model()
```

---
## Run training and evaluation experiment


```python
# Compile the model.
model.compile(
    optimizer=keras.optimizers.Adagrad(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()],
)

# Read the training data.
train_dataset = get_dataset_from_csv("train_data.csv", batch_size=265, shuffle=True)

# Fit the model with the training data.
model.fit(train_dataset, epochs=2)

# Read the test data.
test_dataset = get_dataset_from_csv("test_data.csv", batch_size=265)

# Evaluate the model on the test data.
_, rmse = model.evaluate(test_dataset, verbose=0)
print(f"Test MAE: {round(rmse, 3)}")
```

<div class="k-default-codeblock">
```
Epoch 1/2

```
</div>
    
<div class="k-default-codeblock">
```
  1/Unknown  4s 4s/step - loss: 12.3858 - mean_absolute_error: 3.1926


  2/Unknown  4s 239ms/step - loss: 11.8118 - mean_absolute_error: 3.1138


  3/Unknown  4s 243ms/step - loss: 11.2310 - mean_absolute_error: 3.0252


  4/Unknown  4s 243ms/step - loss: 10.7471 - mean_absolute_error: 2.9454


  5/Unknown  5s 241ms/step - loss: 10.3358 - mean_absolute_error: 2.8758


  6/Unknown  5s 244ms/step - loss: 9.9552 - mean_absolute_error: 2.8099 


  7/Unknown  5s 239ms/step - loss: 9.6164 - mean_absolute_error: 2.7506


  8/Unknown  5s 240ms/step - loss: 9.3172 - mean_absolute_error: 2.6978


  9/Unknown  6s 237ms/step - loss: 9.0496 - mean_absolute_error: 2.6500


 10/Unknown  6s 237ms/step - loss: 8.8066 - mean_absolute_error: 2.6060


 11/Unknown  6s 235ms/step - loss: 8.5807 - mean_absolute_error: 2.5641


 12/Unknown  6s 235ms/step - loss: 8.3674 - mean_absolute_error: 2.5238


 13/Unknown  6s 233ms/step - loss: 8.1696 - mean_absolute_error: 2.4860


 14/Unknown  7s 231ms/step - loss: 7.9838 - mean_absolute_error: 2.4502


 15/Unknown  7s 229ms/step - loss: 7.8095 - mean_absolute_error: 2.4163


 16/Unknown  7s 229ms/step - loss: 7.6458 - mean_absolute_error: 2.3842


 17/Unknown  7s 227ms/step - loss: 7.4916 - mean_absolute_error: 2.3536


 18/Unknown  7s 225ms/step - loss: 7.3462 - mean_absolute_error: 2.3245


 19/Unknown  8s 223ms/step - loss: 7.2084 - mean_absolute_error: 2.2967


 20/Unknown  8s 223ms/step - loss: 7.0775 - mean_absolute_error: 2.2700


 21/Unknown  8s 222ms/step - loss: 6.9525 - mean_absolute_error: 2.2442


 22/Unknown  8s 221ms/step - loss: 6.8334 - mean_absolute_error: 2.2195


 23/Unknown  8s 220ms/step - loss: 6.7195 - mean_absolute_error: 2.1956


 24/Unknown  9s 220ms/step - loss: 6.6108 - mean_absolute_error: 2.1727


 25/Unknown  9s 218ms/step - loss: 6.5073 - mean_absolute_error: 2.1508


 26/Unknown  9s 218ms/step - loss: 6.4083 - mean_absolute_error: 2.1298


 27/Unknown  9s 217ms/step - loss: 6.3134 - mean_absolute_error: 2.1094


 28/Unknown  9s 217ms/step - loss: 6.2224 - mean_absolute_error: 2.0898


 29/Unknown  10s 216ms/step - loss: 6.1350 - mean_absolute_error: 2.0709


 30/Unknown  10s 216ms/step - loss: 6.0512 - mean_absolute_error: 2.0527


 31/Unknown  10s 215ms/step - loss: 5.9709 - mean_absolute_error: 2.0352


 32/Unknown  10s 215ms/step - loss: 5.8938 - mean_absolute_error: 2.0184


 33/Unknown  11s 215ms/step - loss: 5.8197 - mean_absolute_error: 2.0022


 34/Unknown  11s 214ms/step - loss: 5.7486 - mean_absolute_error: 1.9866


 35/Unknown  11s 214ms/step - loss: 5.6801 - mean_absolute_error: 1.9716


 36/Unknown  11s 215ms/step - loss: 5.6138 - mean_absolute_error: 1.9569


 37/Unknown  11s 215ms/step - loss: 5.5499 - mean_absolute_error: 1.9428


 38/Unknown  12s 214ms/step - loss: 5.4882 - mean_absolute_error: 1.9291


 39/Unknown  12s 214ms/step - loss: 5.4286 - mean_absolute_error: 1.9159


 40/Unknown  12s 214ms/step - loss: 5.3711 - mean_absolute_error: 1.9032


 41/Unknown  12s 214ms/step - loss: 5.3154 - mean_absolute_error: 1.8907


 42/Unknown  12s 214ms/step - loss: 5.2615 - mean_absolute_error: 1.8787


 43/Unknown  13s 214ms/step - loss: 5.2092 - mean_absolute_error: 1.8670


 44/Unknown  13s 214ms/step - loss: 5.1587 - mean_absolute_error: 1.8557


 45/Unknown  13s 213ms/step - loss: 5.1096 - mean_absolute_error: 1.8446


 46/Unknown  13s 214ms/step - loss: 5.0619 - mean_absolute_error: 1.8339


 47/Unknown  13s 214ms/step - loss: 5.0156 - mean_absolute_error: 1.8234


 48/Unknown  14s 214ms/step - loss: 4.9705 - mean_absolute_error: 1.8132


 49/Unknown  14s 214ms/step - loss: 4.9267 - mean_absolute_error: 1.8033


 50/Unknown  14s 214ms/step - loss: 4.8842 - mean_absolute_error: 1.7936


 51/Unknown  14s 214ms/step - loss: 4.8430 - mean_absolute_error: 1.7842


 52/Unknown  15s 214ms/step - loss: 4.8028 - mean_absolute_error: 1.7751


 53/Unknown  15s 213ms/step - loss: 4.7637 - mean_absolute_error: 1.7662


 54/Unknown  15s 214ms/step - loss: 4.7256 - mean_absolute_error: 1.7574


 55/Unknown  15s 213ms/step - loss: 4.6885 - mean_absolute_error: 1.7489


 56/Unknown  15s 214ms/step - loss: 4.6523 - mean_absolute_error: 1.7407


 57/Unknown  16s 213ms/step - loss: 4.6171 - mean_absolute_error: 1.7326


 58/Unknown  16s 213ms/step - loss: 4.5826 - mean_absolute_error: 1.7246


 59/Unknown  16s 213ms/step - loss: 4.5489 - mean_absolute_error: 1.7169


 60/Unknown  16s 213ms/step - loss: 4.5160 - mean_absolute_error: 1.7093


 61/Unknown  16s 213ms/step - loss: 4.4839 - mean_absolute_error: 1.7019


 62/Unknown  17s 213ms/step - loss: 4.4527 - mean_absolute_error: 1.6946


 63/Unknown  17s 213ms/step - loss: 4.4220 - mean_absolute_error: 1.6876


 64/Unknown  17s 213ms/step - loss: 4.3921 - mean_absolute_error: 1.6806


 65/Unknown  17s 213ms/step - loss: 4.3629 - mean_absolute_error: 1.6738


 66/Unknown  17s 213ms/step - loss: 4.3342 - mean_absolute_error: 1.6672


 67/Unknown  18s 213ms/step - loss: 4.3062 - mean_absolute_error: 1.6607


 68/Unknown  18s 213ms/step - loss: 4.2788 - mean_absolute_error: 1.6543


 69/Unknown  18s 213ms/step - loss: 4.2521 - mean_absolute_error: 1.6480


 70/Unknown  18s 214ms/step - loss: 4.2258 - mean_absolute_error: 1.6419


 71/Unknown  19s 214ms/step - loss: 4.2001 - mean_absolute_error: 1.6359


 72/Unknown  19s 214ms/step - loss: 4.1749 - mean_absolute_error: 1.6300


 73/Unknown  19s 214ms/step - loss: 4.1503 - mean_absolute_error: 1.6243


 74/Unknown  19s 214ms/step - loss: 4.1260 - mean_absolute_error: 1.6186


 75/Unknown  19s 214ms/step - loss: 4.1023 - mean_absolute_error: 1.6130


 76/Unknown  20s 214ms/step - loss: 4.0790 - mean_absolute_error: 1.6075


 77/Unknown  20s 214ms/step - loss: 4.0562 - mean_absolute_error: 1.6022


 78/Unknown  20s 215ms/step - loss: 4.0338 - mean_absolute_error: 1.5969


 79/Unknown  20s 215ms/step - loss: 4.0119 - mean_absolute_error: 1.5918


 80/Unknown  21s 215ms/step - loss: 3.9904 - mean_absolute_error: 1.5867


 81/Unknown  21s 216ms/step - loss: 3.9692 - mean_absolute_error: 1.5817


 82/Unknown  21s 217ms/step - loss: 3.9485 - mean_absolute_error: 1.5768


 83/Unknown  21s 217ms/step - loss: 3.9281 - mean_absolute_error: 1.5720


 84/Unknown  22s 218ms/step - loss: 3.9082 - mean_absolute_error: 1.5673


 85/Unknown  22s 218ms/step - loss: 3.8886 - mean_absolute_error: 1.5627


 86/Unknown  22s 219ms/step - loss: 3.8694 - mean_absolute_error: 1.5582


 87/Unknown  22s 219ms/step - loss: 3.8505 - mean_absolute_error: 1.5537


 88/Unknown  23s 219ms/step - loss: 3.8319 - mean_absolute_error: 1.5493


 89/Unknown  23s 219ms/step - loss: 3.8137 - mean_absolute_error: 1.5450


 90/Unknown  23s 219ms/step - loss: 3.7958 - mean_absolute_error: 1.5408


 91/Unknown  23s 219ms/step - loss: 3.7782 - mean_absolute_error: 1.5366


 92/Unknown  24s 219ms/step - loss: 3.7610 - mean_absolute_error: 1.5325


 93/Unknown  24s 219ms/step - loss: 3.7440 - mean_absolute_error: 1.5285


 94/Unknown  24s 219ms/step - loss: 3.7273 - mean_absolute_error: 1.5245


 95/Unknown  24s 219ms/step - loss: 3.7108 - mean_absolute_error: 1.5206


 96/Unknown  24s 219ms/step - loss: 3.6947 - mean_absolute_error: 1.5168


 97/Unknown  25s 219ms/step - loss: 3.6787 - mean_absolute_error: 1.5130


 98/Unknown  25s 219ms/step - loss: 3.6631 - mean_absolute_error: 1.5093


 99/Unknown  25s 220ms/step - loss: 3.6477 - mean_absolute_error: 1.5057


100/Unknown  25s 220ms/step - loss: 3.6326 - mean_absolute_error: 1.5021


101/Unknown  26s 220ms/step - loss: 3.6176 - mean_absolute_error: 1.4985


102/Unknown  26s 220ms/step - loss: 3.6030 - mean_absolute_error: 1.4950


103/Unknown  26s 220ms/step - loss: 3.5885 - mean_absolute_error: 1.4916


104/Unknown  26s 220ms/step - loss: 3.5743 - mean_absolute_error: 1.4882


105/Unknown  27s 220ms/step - loss: 3.5603 - mean_absolute_error: 1.4849


106/Unknown  27s 220ms/step - loss: 3.5464 - mean_absolute_error: 1.4816


107/Unknown  27s 220ms/step - loss: 3.5328 - mean_absolute_error: 1.4783


108/Unknown  27s 220ms/step - loss: 3.5194 - mean_absolute_error: 1.4751


109/Unknown  27s 220ms/step - loss: 3.5061 - mean_absolute_error: 1.4720


110/Unknown  28s 220ms/step - loss: 3.4931 - mean_absolute_error: 1.4689


111/Unknown  28s 220ms/step - loss: 3.4802 - mean_absolute_error: 1.4658


112/Unknown  28s 220ms/step - loss: 3.4675 - mean_absolute_error: 1.4628


113/Unknown  28s 220ms/step - loss: 3.4550 - mean_absolute_error: 1.4598


114/Unknown  28s 220ms/step - loss: 3.4427 - mean_absolute_error: 1.4568


115/Unknown  29s 220ms/step - loss: 3.4305 - mean_absolute_error: 1.4539


116/Unknown  29s 220ms/step - loss: 3.4185 - mean_absolute_error: 1.4511


117/Unknown  29s 220ms/step - loss: 3.4067 - mean_absolute_error: 1.4482


118/Unknown  29s 220ms/step - loss: 3.3950 - mean_absolute_error: 1.4455


119/Unknown  30s 220ms/step - loss: 3.3835 - mean_absolute_error: 1.4427


120/Unknown  30s 220ms/step - loss: 3.3721 - mean_absolute_error: 1.4400


121/Unknown  30s 220ms/step - loss: 3.3608 - mean_absolute_error: 1.4373


122/Unknown  30s 220ms/step - loss: 3.3498 - mean_absolute_error: 1.4346


123/Unknown  30s 220ms/step - loss: 3.3388 - mean_absolute_error: 1.4320


124/Unknown  31s 220ms/step - loss: 3.3280 - mean_absolute_error: 1.4294


125/Unknown  31s 220ms/step - loss: 3.3173 - mean_absolute_error: 1.4269


126/Unknown  31s 220ms/step - loss: 3.3068 - mean_absolute_error: 1.4243


127/Unknown  31s 220ms/step - loss: 3.2964 - mean_absolute_error: 1.4219


128/Unknown  32s 220ms/step - loss: 3.2861 - mean_absolute_error: 1.4194


129/Unknown  32s 221ms/step - loss: 3.2759 - mean_absolute_error: 1.4170


130/Unknown  32s 221ms/step - loss: 3.2659 - mean_absolute_error: 1.4145


131/Unknown  32s 221ms/step - loss: 3.2560 - mean_absolute_error: 1.4122


132/Unknown  33s 222ms/step - loss: 3.2462 - mean_absolute_error: 1.4098


133/Unknown  33s 222ms/step - loss: 3.2365 - mean_absolute_error: 1.4075


134/Unknown  33s 222ms/step - loss: 3.2269 - mean_absolute_error: 1.4052


135/Unknown  33s 222ms/step - loss: 3.2174 - mean_absolute_error: 1.4029


136/Unknown  34s 223ms/step - loss: 3.2081 - mean_absolute_error: 1.4006


137/Unknown  34s 223ms/step - loss: 3.1988 - mean_absolute_error: 1.3984


138/Unknown  34s 223ms/step - loss: 3.1897 - mean_absolute_error: 1.3962


139/Unknown  34s 223ms/step - loss: 3.1807 - mean_absolute_error: 1.3940


140/Unknown  35s 223ms/step - loss: 3.1717 - mean_absolute_error: 1.3919


141/Unknown  35s 223ms/step - loss: 3.1629 - mean_absolute_error: 1.3897


142/Unknown  35s 224ms/step - loss: 3.1541 - mean_absolute_error: 1.3876


143/Unknown  35s 224ms/step - loss: 3.1455 - mean_absolute_error: 1.3855


144/Unknown  36s 224ms/step - loss: 3.1369 - mean_absolute_error: 1.3835


145/Unknown  36s 224ms/step - loss: 3.1284 - mean_absolute_error: 1.3814


146/Unknown  36s 225ms/step - loss: 3.1201 - mean_absolute_error: 1.3794


147/Unknown  36s 225ms/step - loss: 3.1118 - mean_absolute_error: 1.3774


148/Unknown  37s 225ms/step - loss: 3.1035 - mean_absolute_error: 1.3754


149/Unknown  37s 225ms/step - loss: 3.0954 - mean_absolute_error: 1.3734


150/Unknown  37s 226ms/step - loss: 3.0874 - mean_absolute_error: 1.3715


151/Unknown  38s 226ms/step - loss: 3.0794 - mean_absolute_error: 1.3696


152/Unknown  38s 226ms/step - loss: 3.0716 - mean_absolute_error: 1.3676


153/Unknown  38s 226ms/step - loss: 3.0638 - mean_absolute_error: 1.3658


154/Unknown  38s 226ms/step - loss: 3.0561 - mean_absolute_error: 1.3639


155/Unknown  39s 227ms/step - loss: 3.0484 - mean_absolute_error: 1.3620


156/Unknown  39s 227ms/step - loss: 3.0409 - mean_absolute_error: 1.3602


157/Unknown  39s 227ms/step - loss: 3.0334 - mean_absolute_error: 1.3584


158/Unknown  39s 227ms/step - loss: 3.0259 - mean_absolute_error: 1.3566


159/Unknown  40s 227ms/step - loss: 3.0186 - mean_absolute_error: 1.3548


160/Unknown  40s 228ms/step - loss: 3.0113 - mean_absolute_error: 1.3530


161/Unknown  40s 228ms/step - loss: 3.0041 - mean_absolute_error: 1.3512


162/Unknown  40s 228ms/step - loss: 2.9969 - mean_absolute_error: 1.3495


163/Unknown  41s 228ms/step - loss: 2.9899 - mean_absolute_error: 1.3478


164/Unknown  41s 229ms/step - loss: 2.9829 - mean_absolute_error: 1.3461


165/Unknown  41s 229ms/step - loss: 2.9759 - mean_absolute_error: 1.3444


166/Unknown  42s 230ms/step - loss: 2.9690 - mean_absolute_error: 1.3427


167/Unknown  42s 230ms/step - loss: 2.9622 - mean_absolute_error: 1.3410


168/Unknown  42s 230ms/step - loss: 2.9555 - mean_absolute_error: 1.3394


169/Unknown  42s 231ms/step - loss: 2.9488 - mean_absolute_error: 1.3377


170/Unknown  43s 231ms/step - loss: 2.9422 - mean_absolute_error: 1.3361


171/Unknown  43s 231ms/step - loss: 2.9356 - mean_absolute_error: 1.3345


172/Unknown  43s 231ms/step - loss: 2.9291 - mean_absolute_error: 1.3329


173/Unknown  43s 231ms/step - loss: 2.9226 - mean_absolute_error: 1.3313


174/Unknown  44s 231ms/step - loss: 2.9163 - mean_absolute_error: 1.3298


175/Unknown  44s 232ms/step - loss: 2.9099 - mean_absolute_error: 1.3282


176/Unknown  44s 232ms/step - loss: 2.9037 - mean_absolute_error: 1.3267


177/Unknown  44s 232ms/step - loss: 2.8975 - mean_absolute_error: 1.3252


178/Unknown  45s 232ms/step - loss: 2.8913 - mean_absolute_error: 1.3237


179/Unknown  45s 232ms/step - loss: 2.8852 - mean_absolute_error: 1.3222


180/Unknown  45s 232ms/step - loss: 2.8791 - mean_absolute_error: 1.3207


181/Unknown  45s 232ms/step - loss: 2.8732 - mean_absolute_error: 1.3192


182/Unknown  46s 232ms/step - loss: 2.8672 - mean_absolute_error: 1.3178


183/Unknown  46s 233ms/step - loss: 2.8613 - mean_absolute_error: 1.3163


184/Unknown  46s 233ms/step - loss: 2.8555 - mean_absolute_error: 1.3149


185/Unknown  47s 233ms/step - loss: 2.8497 - mean_absolute_error: 1.3135


186/Unknown  47s 233ms/step - loss: 2.8439 - mean_absolute_error: 1.3120


187/Unknown  47s 233ms/step - loss: 2.8382 - mean_absolute_error: 1.3106


188/Unknown  47s 234ms/step - loss: 2.8326 - mean_absolute_error: 1.3093


189/Unknown  48s 234ms/step - loss: 2.8270 - mean_absolute_error: 1.3079


190/Unknown  48s 234ms/step - loss: 2.8214 - mean_absolute_error: 1.3065


191/Unknown  48s 234ms/step - loss: 2.8159 - mean_absolute_error: 1.3052


192/Unknown  48s 235ms/step - loss: 2.8105 - mean_absolute_error: 1.3038


193/Unknown  49s 235ms/step - loss: 2.8051 - mean_absolute_error: 1.3025


194/Unknown  49s 235ms/step - loss: 2.7997 - mean_absolute_error: 1.3012


195/Unknown  49s 236ms/step - loss: 2.7944 - mean_absolute_error: 1.2999


196/Unknown  50s 236ms/step - loss: 2.7891 - mean_absolute_error: 1.2986


197/Unknown  50s 236ms/step - loss: 2.7839 - mean_absolute_error: 1.2973


198/Unknown  50s 236ms/step - loss: 2.7787 - mean_absolute_error: 1.2960


199/Unknown  50s 236ms/step - loss: 2.7735 - mean_absolute_error: 1.2947


200/Unknown  51s 237ms/step - loss: 2.7684 - mean_absolute_error: 1.2934


201/Unknown  51s 237ms/step - loss: 2.7633 - mean_absolute_error: 1.2922


202/Unknown  51s 237ms/step - loss: 2.7583 - mean_absolute_error: 1.2909


203/Unknown  52s 237ms/step - loss: 2.7533 - mean_absolute_error: 1.2897


204/Unknown  52s 237ms/step - loss: 2.7483 - mean_absolute_error: 1.2885


205/Unknown  52s 237ms/step - loss: 2.7434 - mean_absolute_error: 1.2873


206/Unknown  52s 237ms/step - loss: 2.7385 - mean_absolute_error: 1.2861


207/Unknown  53s 238ms/step - loss: 2.7336 - mean_absolute_error: 1.2849


208/Unknown  53s 238ms/step - loss: 2.7288 - mean_absolute_error: 1.2837


209/Unknown  53s 238ms/step - loss: 2.7240 - mean_absolute_error: 1.2825


210/Unknown  53s 238ms/step - loss: 2.7193 - mean_absolute_error: 1.2813


211/Unknown  54s 238ms/step - loss: 2.7146 - mean_absolute_error: 1.2801


212/Unknown  54s 238ms/step - loss: 2.7099 - mean_absolute_error: 1.2790


213/Unknown  54s 238ms/step - loss: 2.7052 - mean_absolute_error: 1.2778


214/Unknown  54s 238ms/step - loss: 2.7006 - mean_absolute_error: 1.2767


215/Unknown  55s 239ms/step - loss: 2.6961 - mean_absolute_error: 1.2756


216/Unknown  55s 239ms/step - loss: 2.6915 - mean_absolute_error: 1.2744


217/Unknown  55s 239ms/step - loss: 2.6870 - mean_absolute_error: 1.2733


218/Unknown  56s 239ms/step - loss: 2.6826 - mean_absolute_error: 1.2722


219/Unknown  56s 239ms/step - loss: 2.6781 - mean_absolute_error: 1.2711


220/Unknown  56s 239ms/step - loss: 2.6737 - mean_absolute_error: 1.2700


221/Unknown  56s 240ms/step - loss: 2.6694 - mean_absolute_error: 1.2689


222/Unknown  57s 240ms/step - loss: 2.6650 - mean_absolute_error: 1.2678


223/Unknown  57s 240ms/step - loss: 2.6607 - mean_absolute_error: 1.2668


224/Unknown  57s 241ms/step - loss: 2.6564 - mean_absolute_error: 1.2657


225/Unknown  58s 241ms/step - loss: 2.6522 - mean_absolute_error: 1.2647


226/Unknown  58s 241ms/step - loss: 2.6480 - mean_absolute_error: 1.2636


227/Unknown  58s 242ms/step - loss: 2.6438 - mean_absolute_error: 1.2626


228/Unknown  59s 242ms/step - loss: 2.6396 - mean_absolute_error: 1.2615


229/Unknown  59s 242ms/step - loss: 2.6355 - mean_absolute_error: 1.2605


230/Unknown  59s 243ms/step - loss: 2.6314 - mean_absolute_error: 1.2595


231/Unknown  59s 243ms/step - loss: 2.6273 - mean_absolute_error: 1.2585


232/Unknown  60s 243ms/step - loss: 2.6232 - mean_absolute_error: 1.2575


233/Unknown  60s 243ms/step - loss: 2.6192 - mean_absolute_error: 1.2565


234/Unknown  60s 243ms/step - loss: 2.6152 - mean_absolute_error: 1.2555


235/Unknown  61s 243ms/step - loss: 2.6113 - mean_absolute_error: 1.2545


236/Unknown  61s 243ms/step - loss: 2.6073 - mean_absolute_error: 1.2535


237/Unknown  61s 243ms/step - loss: 2.6034 - mean_absolute_error: 1.2525


238/Unknown  61s 243ms/step - loss: 2.5995 - mean_absolute_error: 1.2515


239/Unknown  62s 243ms/step - loss: 2.5956 - mean_absolute_error: 1.2506


240/Unknown  62s 244ms/step - loss: 2.5918 - mean_absolute_error: 1.2496


241/Unknown  62s 244ms/step - loss: 2.5880 - mean_absolute_error: 1.2487


242/Unknown  62s 244ms/step - loss: 2.5842 - mean_absolute_error: 1.2477


243/Unknown  63s 244ms/step - loss: 2.5804 - mean_absolute_error: 1.2468


244/Unknown  63s 244ms/step - loss: 2.5767 - mean_absolute_error: 1.2458


245/Unknown  63s 244ms/step - loss: 2.5730 - mean_absolute_error: 1.2449


246/Unknown  64s 244ms/step - loss: 2.5693 - mean_absolute_error: 1.2440


247/Unknown  64s 244ms/step - loss: 2.5656 - mean_absolute_error: 1.2431


248/Unknown  64s 245ms/step - loss: 2.5620 - mean_absolute_error: 1.2422


249/Unknown  64s 245ms/step - loss: 2.5584 - mean_absolute_error: 1.2413


250/Unknown  65s 245ms/step - loss: 2.5548 - mean_absolute_error: 1.2404


251/Unknown  65s 246ms/step - loss: 2.5512 - mean_absolute_error: 1.2395


252/Unknown  65s 246ms/step - loss: 2.5476 - mean_absolute_error: 1.2386


253/Unknown  66s 246ms/step - loss: 2.5441 - mean_absolute_error: 1.2377


254/Unknown  66s 246ms/step - loss: 2.5406 - mean_absolute_error: 1.2368


255/Unknown  66s 246ms/step - loss: 2.5371 - mean_absolute_error: 1.2359


256/Unknown  66s 246ms/step - loss: 2.5336 - mean_absolute_error: 1.2351


257/Unknown  67s 246ms/step - loss: 2.5302 - mean_absolute_error: 1.2342


258/Unknown  67s 246ms/step - loss: 2.5267 - mean_absolute_error: 1.2333


259/Unknown  67s 247ms/step - loss: 2.5233 - mean_absolute_error: 1.2325


260/Unknown  68s 247ms/step - loss: 2.5199 - mean_absolute_error: 1.2316


261/Unknown  68s 247ms/step - loss: 2.5166 - mean_absolute_error: 1.2308


262/Unknown  68s 247ms/step - loss: 2.5132 - mean_absolute_error: 1.2299


263/Unknown  68s 247ms/step - loss: 2.5099 - mean_absolute_error: 1.2291


264/Unknown  69s 247ms/step - loss: 2.5066 - mean_absolute_error: 1.2283


265/Unknown  69s 247ms/step - loss: 2.5033 - mean_absolute_error: 1.2274


266/Unknown  69s 247ms/step - loss: 2.5000 - mean_absolute_error: 1.2266


267/Unknown  69s 247ms/step - loss: 2.4968 - mean_absolute_error: 1.2258


268/Unknown  70s 248ms/step - loss: 2.4935 - mean_absolute_error: 1.2250


269/Unknown  70s 248ms/step - loss: 2.4903 - mean_absolute_error: 1.2242


270/Unknown  70s 248ms/step - loss: 2.4871 - mean_absolute_error: 1.2233


271/Unknown  71s 248ms/step - loss: 2.4839 - mean_absolute_error: 1.2225


272/Unknown  71s 248ms/step - loss: 2.4808 - mean_absolute_error: 1.2217


273/Unknown  71s 248ms/step - loss: 2.4776 - mean_absolute_error: 1.2210


274/Unknown  71s 248ms/step - loss: 2.4745 - mean_absolute_error: 1.2202


275/Unknown  72s 248ms/step - loss: 2.4714 - mean_absolute_error: 1.2194


276/Unknown  72s 248ms/step - loss: 2.4683 - mean_absolute_error: 1.2186


277/Unknown  72s 248ms/step - loss: 2.4652 - mean_absolute_error: 1.2178


278/Unknown  72s 248ms/step - loss: 2.4622 - mean_absolute_error: 1.2170


279/Unknown  73s 248ms/step - loss: 2.4591 - mean_absolute_error: 1.2163


280/Unknown  73s 249ms/step - loss: 2.4561 - mean_absolute_error: 1.2155


281/Unknown  73s 249ms/step - loss: 2.4531 - mean_absolute_error: 1.2147


282/Unknown  74s 249ms/step - loss: 2.4501 - mean_absolute_error: 1.2140


283/Unknown  74s 249ms/step - loss: 2.4471 - mean_absolute_error: 1.2132


284/Unknown  74s 249ms/step - loss: 2.4442 - mean_absolute_error: 1.2125


285/Unknown  74s 249ms/step - loss: 2.4412 - mean_absolute_error: 1.2117


286/Unknown  75s 250ms/step - loss: 2.4383 - mean_absolute_error: 1.2110


287/Unknown  75s 250ms/step - loss: 2.4354 - mean_absolute_error: 1.2103


288/Unknown  75s 250ms/step - loss: 2.4325 - mean_absolute_error: 1.2095


289/Unknown  76s 251ms/step - loss: 2.4296 - mean_absolute_error: 1.2088


290/Unknown  76s 251ms/step - loss: 2.4267 - mean_absolute_error: 1.2081


291/Unknown  76s 251ms/step - loss: 2.4239 - mean_absolute_error: 1.2073


292/Unknown  77s 251ms/step - loss: 2.4210 - mean_absolute_error: 1.2066


293/Unknown  77s 251ms/step - loss: 2.4182 - mean_absolute_error: 1.2059


294/Unknown  77s 251ms/step - loss: 2.4154 - mean_absolute_error: 1.2052


295/Unknown  78s 251ms/step - loss: 2.4126 - mean_absolute_error: 1.2045


296/Unknown  78s 251ms/step - loss: 2.4098 - mean_absolute_error: 1.2038


297/Unknown  78s 251ms/step - loss: 2.4071 - mean_absolute_error: 1.2031


298/Unknown  78s 251ms/step - loss: 2.4043 - mean_absolute_error: 1.2024


299/Unknown  79s 251ms/step - loss: 2.4016 - mean_absolute_error: 1.2017


300/Unknown  79s 251ms/step - loss: 2.3989 - mean_absolute_error: 1.2010


301/Unknown  79s 251ms/step - loss: 2.3962 - mean_absolute_error: 1.2003


302/Unknown  79s 252ms/step - loss: 2.3935 - mean_absolute_error: 1.1996


303/Unknown  80s 252ms/step - loss: 2.3908 - mean_absolute_error: 1.1989


304/Unknown  80s 252ms/step - loss: 2.3882 - mean_absolute_error: 1.1983


305/Unknown  80s 252ms/step - loss: 2.3855 - mean_absolute_error: 1.1976


306/Unknown  80s 252ms/step - loss: 2.3829 - mean_absolute_error: 1.1969


307/Unknown  81s 252ms/step - loss: 2.3803 - mean_absolute_error: 1.1962


308/Unknown  81s 252ms/step - loss: 2.3777 - mean_absolute_error: 1.1956


309/Unknown  81s 252ms/step - loss: 2.3751 - mean_absolute_error: 1.1949


310/Unknown  81s 252ms/step - loss: 2.3725 - mean_absolute_error: 1.1943


311/Unknown  82s 252ms/step - loss: 2.3700 - mean_absolute_error: 1.1936


312/Unknown  82s 252ms/step - loss: 2.3674 - mean_absolute_error: 1.1930


313/Unknown  82s 252ms/step - loss: 2.3649 - mean_absolute_error: 1.1923


314/Unknown  83s 252ms/step - loss: 2.3624 - mean_absolute_error: 1.1917


315/Unknown  83s 252ms/step - loss: 2.3599 - mean_absolute_error: 1.1910


316/Unknown  83s 252ms/step - loss: 2.3574 - mean_absolute_error: 1.1904


317/Unknown  83s 252ms/step - loss: 2.3549 - mean_absolute_error: 1.1898


318/Unknown  84s 252ms/step - loss: 2.3524 - mean_absolute_error: 1.1891


319/Unknown  84s 253ms/step - loss: 2.3500 - mean_absolute_error: 1.1885


320/Unknown  84s 253ms/step - loss: 2.3475 - mean_absolute_error: 1.1879


321/Unknown  85s 253ms/step - loss: 2.3451 - mean_absolute_error: 1.1873


322/Unknown  85s 253ms/step - loss: 2.3427 - mean_absolute_error: 1.1866


323/Unknown  85s 253ms/step - loss: 2.3403 - mean_absolute_error: 1.1860


324/Unknown  86s 253ms/step - loss: 2.3379 - mean_absolute_error: 1.1854


325/Unknown  86s 254ms/step - loss: 2.3355 - mean_absolute_error: 1.1848


326/Unknown  86s 254ms/step - loss: 2.3331 - mean_absolute_error: 1.1842


327/Unknown  86s 254ms/step - loss: 2.3308 - mean_absolute_error: 1.1836


328/Unknown  87s 254ms/step - loss: 2.3284 - mean_absolute_error: 1.1830


329/Unknown  87s 254ms/step - loss: 2.3261 - mean_absolute_error: 1.1824


330/Unknown  87s 254ms/step - loss: 2.3238 - mean_absolute_error: 1.1818


331/Unknown  88s 254ms/step - loss: 2.3214 - mean_absolute_error: 1.1812


332/Unknown  88s 254ms/step - loss: 2.3191 - mean_absolute_error: 1.1806


333/Unknown  88s 254ms/step - loss: 2.3168 - mean_absolute_error: 1.1800


334/Unknown  88s 255ms/step - loss: 2.3146 - mean_absolute_error: 1.1794


335/Unknown  89s 255ms/step - loss: 2.3123 - mean_absolute_error: 1.1789


336/Unknown  89s 255ms/step - loss: 2.3100 - mean_absolute_error: 1.1783


337/Unknown  89s 255ms/step - loss: 2.3078 - mean_absolute_error: 1.1777


338/Unknown  90s 255ms/step - loss: 2.3056 - mean_absolute_error: 1.1771


339/Unknown  90s 255ms/step - loss: 2.3033 - mean_absolute_error: 1.1766


340/Unknown  90s 255ms/step - loss: 2.3011 - mean_absolute_error: 1.1760


341/Unknown  90s 255ms/step - loss: 2.2989 - mean_absolute_error: 1.1754


342/Unknown  91s 256ms/step - loss: 2.2967 - mean_absolute_error: 1.1749


343/Unknown  91s 256ms/step - loss: 2.2945 - mean_absolute_error: 1.1743


344/Unknown  91s 256ms/step - loss: 2.2924 - mean_absolute_error: 1.1738


345/Unknown  92s 256ms/step - loss: 2.2902 - mean_absolute_error: 1.1732


346/Unknown  92s 256ms/step - loss: 2.2881 - mean_absolute_error: 1.1726


347/Unknown  92s 256ms/step - loss: 2.2859 - mean_absolute_error: 1.1721


348/Unknown  93s 256ms/step - loss: 2.2838 - mean_absolute_error: 1.1715


349/Unknown  93s 257ms/step - loss: 2.2817 - mean_absolute_error: 1.1710


350/Unknown  93s 257ms/step - loss: 2.2795 - mean_absolute_error: 1.1705


351/Unknown  94s 257ms/step - loss: 2.2774 - mean_absolute_error: 1.1699


352/Unknown  94s 257ms/step - loss: 2.2753 - mean_absolute_error: 1.1694


353/Unknown  94s 257ms/step - loss: 2.2733 - mean_absolute_error: 1.1688


354/Unknown  95s 258ms/step - loss: 2.2712 - mean_absolute_error: 1.1683


355/Unknown  95s 258ms/step - loss: 2.2691 - mean_absolute_error: 1.1678


356/Unknown  95s 258ms/step - loss: 2.2671 - mean_absolute_error: 1.1673


357/Unknown  95s 258ms/step - loss: 2.2650 - mean_absolute_error: 1.1667


358/Unknown  96s 258ms/step - loss: 2.2630 - mean_absolute_error: 1.1662


359/Unknown  96s 258ms/step - loss: 2.2610 - mean_absolute_error: 1.1657


360/Unknown  96s 258ms/step - loss: 2.2590 - mean_absolute_error: 1.1652


361/Unknown  97s 258ms/step - loss: 2.2570 - mean_absolute_error: 1.1647


362/Unknown  97s 258ms/step - loss: 2.2550 - mean_absolute_error: 1.1641


363/Unknown  97s 258ms/step - loss: 2.2530 - mean_absolute_error: 1.1636


364/Unknown  97s 258ms/step - loss: 2.2510 - mean_absolute_error: 1.1631


365/Unknown  98s 258ms/step - loss: 2.2490 - mean_absolute_error: 1.1626


366/Unknown  98s 258ms/step - loss: 2.2471 - mean_absolute_error: 1.1621


367/Unknown  98s 258ms/step - loss: 2.2451 - mean_absolute_error: 1.1616


368/Unknown  98s 258ms/step - loss: 2.2432 - mean_absolute_error: 1.1611


369/Unknown  99s 258ms/step - loss: 2.2412 - mean_absolute_error: 1.1606


370/Unknown  99s 258ms/step - loss: 2.2393 - mean_absolute_error: 1.1601


371/Unknown  99s 258ms/step - loss: 2.2374 - mean_absolute_error: 1.1596


372/Unknown  99s 258ms/step - loss: 2.2355 - mean_absolute_error: 1.1591


373/Unknown  100s 258ms/step - loss: 2.2336 - mean_absolute_error: 1.1587


374/Unknown  100s 258ms/step - loss: 2.2317 - mean_absolute_error: 1.1582


375/Unknown  100s 259ms/step - loss: 2.2298 - mean_absolute_error: 1.1577


376/Unknown  101s 259ms/step - loss: 2.2279 - mean_absolute_error: 1.1572


377/Unknown  101s 259ms/step - loss: 2.2261 - mean_absolute_error: 1.1567


378/Unknown  101s 259ms/step - loss: 2.2242 - mean_absolute_error: 1.1562


379/Unknown  102s 259ms/step - loss: 2.2223 - mean_absolute_error: 1.1558


380/Unknown  102s 259ms/step - loss: 2.2205 - mean_absolute_error: 1.1553


381/Unknown  102s 259ms/step - loss: 2.2187 - mean_absolute_error: 1.1548


382/Unknown  102s 259ms/step - loss: 2.2168 - mean_absolute_error: 1.1543


383/Unknown  103s 259ms/step - loss: 2.2150 - mean_absolute_error: 1.1539


384/Unknown  103s 259ms/step - loss: 2.2132 - mean_absolute_error: 1.1534


385/Unknown  103s 259ms/step - loss: 2.2114 - mean_absolute_error: 1.1529


386/Unknown  104s 259ms/step - loss: 2.2096 - mean_absolute_error: 1.1525


387/Unknown  104s 259ms/step - loss: 2.2078 - mean_absolute_error: 1.1520


388/Unknown  104s 259ms/step - loss: 2.2060 - mean_absolute_error: 1.1516


389/Unknown  104s 259ms/step - loss: 2.2043 - mean_absolute_error: 1.1511


390/Unknown  105s 259ms/step - loss: 2.2025 - mean_absolute_error: 1.1506


391/Unknown  105s 259ms/step - loss: 2.2007 - mean_absolute_error: 1.1502


392/Unknown  105s 259ms/step - loss: 2.1990 - mean_absolute_error: 1.1497


393/Unknown  105s 259ms/step - loss: 2.1972 - mean_absolute_error: 1.1493


394/Unknown  106s 259ms/step - loss: 2.1955 - mean_absolute_error: 1.1488


395/Unknown  106s 259ms/step - loss: 2.1937 - mean_absolute_error: 1.1484


396/Unknown  106s 259ms/step - loss: 2.1920 - mean_absolute_error: 1.1479


397/Unknown  106s 260ms/step - loss: 2.1903 - mean_absolute_error: 1.1475


398/Unknown  107s 260ms/step - loss: 2.1886 - mean_absolute_error: 1.1471


399/Unknown  107s 260ms/step - loss: 2.1869 - mean_absolute_error: 1.1466


400/Unknown  107s 260ms/step - loss: 2.1852 - mean_absolute_error: 1.1462


401/Unknown  107s 260ms/step - loss: 2.1835 - mean_absolute_error: 1.1457


402/Unknown  108s 260ms/step - loss: 2.1818 - mean_absolute_error: 1.1453


403/Unknown  108s 260ms/step - loss: 2.1801 - mean_absolute_error: 1.1449


404/Unknown  108s 260ms/step - loss: 2.1784 - mean_absolute_error: 1.1444


405/Unknown  109s 260ms/step - loss: 2.1767 - mean_absolute_error: 1.1440


406/Unknown  109s 260ms/step - loss: 2.1751 - mean_absolute_error: 1.1436


407/Unknown  109s 261ms/step - loss: 2.1734 - mean_absolute_error: 1.1431


408/Unknown  110s 261ms/step - loss: 2.1718 - mean_absolute_error: 1.1427


409/Unknown  110s 261ms/step - loss: 2.1701 - mean_absolute_error: 1.1423


410/Unknown  110s 261ms/step - loss: 2.1685 - mean_absolute_error: 1.1419


411/Unknown  111s 261ms/step - loss: 2.1668 - mean_absolute_error: 1.1414


412/Unknown  111s 261ms/step - loss: 2.1652 - mean_absolute_error: 1.1410


413/Unknown  111s 261ms/step - loss: 2.1636 - mean_absolute_error: 1.1406


414/Unknown  111s 261ms/step - loss: 2.1620 - mean_absolute_error: 1.1402


415/Unknown  112s 261ms/step - loss: 2.1604 - mean_absolute_error: 1.1398


416/Unknown  112s 261ms/step - loss: 2.1588 - mean_absolute_error: 1.1393


417/Unknown  112s 261ms/step - loss: 2.1572 - mean_absolute_error: 1.1389


418/Unknown  113s 261ms/step - loss: 2.1556 - mean_absolute_error: 1.1385


419/Unknown  113s 261ms/step - loss: 2.1540 - mean_absolute_error: 1.1381


420/Unknown  113s 261ms/step - loss: 2.1524 - mean_absolute_error: 1.1377


421/Unknown  113s 262ms/step - loss: 2.1509 - mean_absolute_error: 1.1373


422/Unknown  114s 262ms/step - loss: 2.1493 - mean_absolute_error: 1.1369


423/Unknown  114s 262ms/step - loss: 2.1477 - mean_absolute_error: 1.1365


424/Unknown  114s 262ms/step - loss: 2.1462 - mean_absolute_error: 1.1361


425/Unknown  115s 262ms/step - loss: 2.1446 - mean_absolute_error: 1.1357


426/Unknown  115s 262ms/step - loss: 2.1431 - mean_absolute_error: 1.1353


427/Unknown  115s 262ms/step - loss: 2.1416 - mean_absolute_error: 1.1349


428/Unknown  116s 262ms/step - loss: 2.1400 - mean_absolute_error: 1.1345


429/Unknown  116s 262ms/step - loss: 2.1385 - mean_absolute_error: 1.1341


430/Unknown  116s 263ms/step - loss: 2.1370 - mean_absolute_error: 1.1337


431/Unknown  117s 263ms/step - loss: 2.1355 - mean_absolute_error: 1.1333


432/Unknown  117s 263ms/step - loss: 2.1340 - mean_absolute_error: 1.1329


433/Unknown  117s 263ms/step - loss: 2.1324 - mean_absolute_error: 1.1325


434/Unknown  118s 263ms/step - loss: 2.1310 - mean_absolute_error: 1.1321


435/Unknown  118s 263ms/step - loss: 2.1295 - mean_absolute_error: 1.1317


436/Unknown  118s 264ms/step - loss: 2.1280 - mean_absolute_error: 1.1313


437/Unknown  119s 264ms/step - loss: 2.1265 - mean_absolute_error: 1.1309


438/Unknown  119s 264ms/step - loss: 2.1250 - mean_absolute_error: 1.1306


439/Unknown  119s 264ms/step - loss: 2.1235 - mean_absolute_error: 1.1302


440/Unknown  119s 264ms/step - loss: 2.1221 - mean_absolute_error: 1.1298


441/Unknown  120s 264ms/step - loss: 2.1206 - mean_absolute_error: 1.1294


442/Unknown  120s 264ms/step - loss: 2.1192 - mean_absolute_error: 1.1290


443/Unknown  120s 264ms/step - loss: 2.1177 - mean_absolute_error: 1.1286


444/Unknown  121s 264ms/step - loss: 2.1163 - mean_absolute_error: 1.1283


445/Unknown  121s 264ms/step - loss: 2.1148 - mean_absolute_error: 1.1279


446/Unknown  121s 264ms/step - loss: 2.1134 - mean_absolute_error: 1.1275


447/Unknown  121s 264ms/step - loss: 2.1119 - mean_absolute_error: 1.1271


448/Unknown  122s 264ms/step - loss: 2.1105 - mean_absolute_error: 1.1268


449/Unknown  122s 264ms/step - loss: 2.1091 - mean_absolute_error: 1.1264


450/Unknown  122s 264ms/step - loss: 2.1077 - mean_absolute_error: 1.1260


451/Unknown  123s 264ms/step - loss: 2.1063 - mean_absolute_error: 1.1257


452/Unknown  123s 265ms/step - loss: 2.1049 - mean_absolute_error: 1.1253


453/Unknown  123s 265ms/step - loss: 2.1035 - mean_absolute_error: 1.1249


454/Unknown  124s 265ms/step - loss: 2.1021 - mean_absolute_error: 1.1246


455/Unknown  124s 265ms/step - loss: 2.1007 - mean_absolute_error: 1.1242


456/Unknown  124s 265ms/step - loss: 2.0993 - mean_absolute_error: 1.1238


457/Unknown  124s 265ms/step - loss: 2.0979 - mean_absolute_error: 1.1235


458/Unknown  125s 265ms/step - loss: 2.0965 - mean_absolute_error: 1.1231


459/Unknown  125s 265ms/step - loss: 2.0952 - mean_absolute_error: 1.1228


460/Unknown  125s 265ms/step - loss: 2.0938 - mean_absolute_error: 1.1224


461/Unknown  126s 265ms/step - loss: 2.0924 - mean_absolute_error: 1.1220


462/Unknown  126s 265ms/step - loss: 2.0911 - mean_absolute_error: 1.1217


463/Unknown  126s 265ms/step - loss: 2.0897 - mean_absolute_error: 1.1213


464/Unknown  126s 265ms/step - loss: 2.0884 - mean_absolute_error: 1.1210


465/Unknown  127s 265ms/step - loss: 2.0870 - mean_absolute_error: 1.1206


466/Unknown  127s 265ms/step - loss: 2.0857 - mean_absolute_error: 1.1203


467/Unknown  127s 265ms/step - loss: 2.0843 - mean_absolute_error: 1.1199


468/Unknown  128s 266ms/step - loss: 2.0830 - mean_absolute_error: 1.1196


469/Unknown  128s 266ms/step - loss: 2.0817 - mean_absolute_error: 1.1192


470/Unknown  128s 266ms/step - loss: 2.0803 - mean_absolute_error: 1.1189


471/Unknown  129s 266ms/step - loss: 2.0790 - mean_absolute_error: 1.1185


472/Unknown  129s 266ms/step - loss: 2.0777 - mean_absolute_error: 1.1182


473/Unknown  129s 266ms/step - loss: 2.0764 - mean_absolute_error: 1.1178


474/Unknown  130s 266ms/step - loss: 2.0751 - mean_absolute_error: 1.1175


475/Unknown  130s 266ms/step - loss: 2.0738 - mean_absolute_error: 1.1172


476/Unknown  130s 266ms/step - loss: 2.0725 - mean_absolute_error: 1.1168


477/Unknown  130s 266ms/step - loss: 2.0712 - mean_absolute_error: 1.1165


478/Unknown  131s 266ms/step - loss: 2.0699 - mean_absolute_error: 1.1161


479/Unknown  131s 266ms/step - loss: 2.0686 - mean_absolute_error: 1.1158


480/Unknown  131s 266ms/step - loss: 2.0673 - mean_absolute_error: 1.1155


481/Unknown  131s 266ms/step - loss: 2.0661 - mean_absolute_error: 1.1151


482/Unknown  132s 266ms/step - loss: 2.0648 - mean_absolute_error: 1.1148


483/Unknown  132s 266ms/step - loss: 2.0635 - mean_absolute_error: 1.1145


484/Unknown  132s 266ms/step - loss: 2.0623 - mean_absolute_error: 1.1141


485/Unknown  133s 267ms/step - loss: 2.0610 - mean_absolute_error: 1.1138


486/Unknown  133s 267ms/step - loss: 2.0597 - mean_absolute_error: 1.1135


487/Unknown  133s 267ms/step - loss: 2.0585 - mean_absolute_error: 1.1131


488/Unknown  134s 267ms/step - loss: 2.0572 - mean_absolute_error: 1.1128


489/Unknown  134s 267ms/step - loss: 2.0560 - mean_absolute_error: 1.1125


490/Unknown  134s 267ms/step - loss: 2.0548 - mean_absolute_error: 1.1122


491/Unknown  134s 267ms/step - loss: 2.0535 - mean_absolute_error: 1.1118


492/Unknown  135s 267ms/step - loss: 2.0523 - mean_absolute_error: 1.1115


493/Unknown  135s 267ms/step - loss: 2.0511 - mean_absolute_error: 1.1112


494/Unknown  135s 267ms/step - loss: 2.0498 - mean_absolute_error: 1.1109


495/Unknown  136s 267ms/step - loss: 2.0486 - mean_absolute_error: 1.1106


496/Unknown  136s 268ms/step - loss: 2.0474 - mean_absolute_error: 1.1102


497/Unknown  136s 268ms/step - loss: 2.0462 - mean_absolute_error: 1.1099


498/Unknown  137s 268ms/step - loss: 2.0450 - mean_absolute_error: 1.1096


499/Unknown  137s 268ms/step - loss: 2.0438 - mean_absolute_error: 1.1093


500/Unknown  137s 268ms/step - loss: 2.0426 - mean_absolute_error: 1.1090


501/Unknown  138s 268ms/step - loss: 2.0414 - mean_absolute_error: 1.1086


502/Unknown  138s 268ms/step - loss: 2.0402 - mean_absolute_error: 1.1083


503/Unknown  138s 268ms/step - loss: 2.0390 - mean_absolute_error: 1.1080


504/Unknown  139s 268ms/step - loss: 2.0378 - mean_absolute_error: 1.1077


505/Unknown  139s 269ms/step - loss: 2.0366 - mean_absolute_error: 1.1074


506/Unknown  139s 269ms/step - loss: 2.0354 - mean_absolute_error: 1.1071


507/Unknown  140s 269ms/step - loss: 2.0343 - mean_absolute_error: 1.1068


508/Unknown  140s 269ms/step - loss: 2.0331 - mean_absolute_error: 1.1065


509/Unknown  140s 269ms/step - loss: 2.0319 - mean_absolute_error: 1.1062


510/Unknown  141s 269ms/step - loss: 2.0308 - mean_absolute_error: 1.1059


511/Unknown  141s 269ms/step - loss: 2.0296 - mean_absolute_error: 1.1055


512/Unknown  141s 269ms/step - loss: 2.0285 - mean_absolute_error: 1.1052


513/Unknown  142s 270ms/step - loss: 2.0273 - mean_absolute_error: 1.1049


514/Unknown  142s 270ms/step - loss: 2.0262 - mean_absolute_error: 1.1046


515/Unknown  142s 270ms/step - loss: 2.0250 - mean_absolute_error: 1.1043


516/Unknown  143s 270ms/step - loss: 2.0239 - mean_absolute_error: 1.1040


517/Unknown  143s 270ms/step - loss: 2.0227 - mean_absolute_error: 1.1037


518/Unknown  143s 270ms/step - loss: 2.0216 - mean_absolute_error: 1.1034


519/Unknown  144s 270ms/step - loss: 2.0205 - mean_absolute_error: 1.1031


520/Unknown  144s 270ms/step - loss: 2.0193 - mean_absolute_error: 1.1028


521/Unknown  144s 270ms/step - loss: 2.0182 - mean_absolute_error: 1.1025


522/Unknown  144s 270ms/step - loss: 2.0171 - mean_absolute_error: 1.1022


523/Unknown  145s 270ms/step - loss: 2.0160 - mean_absolute_error: 1.1019


524/Unknown  145s 270ms/step - loss: 2.0149 - mean_absolute_error: 1.1017


525/Unknown  145s 271ms/step - loss: 2.0137 - mean_absolute_error: 1.1014


526/Unknown  146s 271ms/step - loss: 2.0126 - mean_absolute_error: 1.1011


527/Unknown  146s 271ms/step - loss: 2.0115 - mean_absolute_error: 1.1008


528/Unknown  146s 271ms/step - loss: 2.0104 - mean_absolute_error: 1.1005


529/Unknown  147s 271ms/step - loss: 2.0093 - mean_absolute_error: 1.1002


530/Unknown  147s 271ms/step - loss: 2.0082 - mean_absolute_error: 1.0999


531/Unknown  147s 271ms/step - loss: 2.0072 - mean_absolute_error: 1.0996


532/Unknown  147s 271ms/step - loss: 2.0061 - mean_absolute_error: 1.0993


533/Unknown  148s 271ms/step - loss: 2.0050 - mean_absolute_error: 1.0990


534/Unknown  148s 271ms/step - loss: 2.0039 - mean_absolute_error: 1.0988


535/Unknown  148s 271ms/step - loss: 2.0028 - mean_absolute_error: 1.0985


536/Unknown  149s 271ms/step - loss: 2.0018 - mean_absolute_error: 1.0982


537/Unknown  149s 271ms/step - loss: 2.0007 - mean_absolute_error: 1.0979


538/Unknown  149s 271ms/step - loss: 1.9996 - mean_absolute_error: 1.0976


539/Unknown  150s 271ms/step - loss: 1.9986 - mean_absolute_error: 1.0973


540/Unknown  150s 271ms/step - loss: 1.9975 - mean_absolute_error: 1.0971


541/Unknown  150s 271ms/step - loss: 1.9964 - mean_absolute_error: 1.0968


542/Unknown  150s 271ms/step - loss: 1.9954 - mean_absolute_error: 1.0965


543/Unknown  151s 271ms/step - loss: 1.9943 - mean_absolute_error: 1.0962


544/Unknown  151s 271ms/step - loss: 1.9933 - mean_absolute_error: 1.0960


545/Unknown  151s 272ms/step - loss: 1.9922 - mean_absolute_error: 1.0957


546/Unknown  152s 272ms/step - loss: 1.9912 - mean_absolute_error: 1.0954


547/Unknown  152s 272ms/step - loss: 1.9901 - mean_absolute_error: 1.0951


548/Unknown  152s 272ms/step - loss: 1.9891 - mean_absolute_error: 1.0948


549/Unknown  153s 272ms/step - loss: 1.9881 - mean_absolute_error: 1.0946


550/Unknown  153s 272ms/step - loss: 1.9870 - mean_absolute_error: 1.0943


551/Unknown  154s 273ms/step - loss: 1.9860 - mean_absolute_error: 1.0940


552/Unknown  154s 273ms/step - loss: 1.9850 - mean_absolute_error: 1.0938


553/Unknown  154s 273ms/step - loss: 1.9840 - mean_absolute_error: 1.0935


554/Unknown  154s 273ms/step - loss: 1.9829 - mean_absolute_error: 1.0932


555/Unknown  155s 273ms/step - loss: 1.9819 - mean_absolute_error: 1.0930


556/Unknown  155s 273ms/step - loss: 1.9809 - mean_absolute_error: 1.0927


557/Unknown  155s 273ms/step - loss: 1.9799 - mean_absolute_error: 1.0924


558/Unknown  156s 273ms/step - loss: 1.9789 - mean_absolute_error: 1.0921


559/Unknown  156s 273ms/step - loss: 1.9779 - mean_absolute_error: 1.0919


560/Unknown  156s 273ms/step - loss: 1.9769 - mean_absolute_error: 1.0916


561/Unknown  157s 273ms/step - loss: 1.9759 - mean_absolute_error: 1.0914


562/Unknown  157s 273ms/step - loss: 1.9749 - mean_absolute_error: 1.0911


563/Unknown  157s 273ms/step - loss: 1.9739 - mean_absolute_error: 1.0908


564/Unknown  158s 273ms/step - loss: 1.9729 - mean_absolute_error: 1.0906


565/Unknown  158s 274ms/step - loss: 1.9719 - mean_absolute_error: 1.0903


566/Unknown  158s 274ms/step - loss: 1.9709 - mean_absolute_error: 1.0900


567/Unknown  159s 274ms/step - loss: 1.9700 - mean_absolute_error: 1.0898


568/Unknown  159s 274ms/step - loss: 1.9690 - mean_absolute_error: 1.0895


569/Unknown  159s 274ms/step - loss: 1.9680 - mean_absolute_error: 1.0893


570/Unknown  160s 274ms/step - loss: 1.9670 - mean_absolute_error: 1.0890


571/Unknown  160s 274ms/step - loss: 1.9661 - mean_absolute_error: 1.0887


572/Unknown  160s 274ms/step - loss: 1.9651 - mean_absolute_error: 1.0885


573/Unknown  161s 274ms/step - loss: 1.9641 - mean_absolute_error: 1.0882


574/Unknown  161s 274ms/step - loss: 1.9632 - mean_absolute_error: 1.0880


575/Unknown  161s 275ms/step - loss: 1.9622 - mean_absolute_error: 1.0877


576/Unknown  162s 275ms/step - loss: 1.9612 - mean_absolute_error: 1.0875


577/Unknown  162s 275ms/step - loss: 1.9603 - mean_absolute_error: 1.0872


578/Unknown  162s 275ms/step - loss: 1.9593 - mean_absolute_error: 1.0870


579/Unknown  163s 275ms/step - loss: 1.9584 - mean_absolute_error: 1.0867


580/Unknown  163s 275ms/step - loss: 1.9574 - mean_absolute_error: 1.0865


581/Unknown  163s 275ms/step - loss: 1.9565 - mean_absolute_error: 1.0862


582/Unknown  163s 275ms/step - loss: 1.9556 - mean_absolute_error: 1.0860


583/Unknown  164s 275ms/step - loss: 1.9546 - mean_absolute_error: 1.0857


584/Unknown  164s 275ms/step - loss: 1.9537 - mean_absolute_error: 1.0855


585/Unknown  164s 275ms/step - loss: 1.9527 - mean_absolute_error: 1.0852


586/Unknown  165s 275ms/step - loss: 1.9518 - mean_absolute_error: 1.0850


587/Unknown  165s 275ms/step - loss: 1.9509 - mean_absolute_error: 1.0847


588/Unknown  165s 275ms/step - loss: 1.9499 - mean_absolute_error: 1.0845


589/Unknown  166s 275ms/step - loss: 1.9490 - mean_absolute_error: 1.0842


590/Unknown  166s 276ms/step - loss: 1.9481 - mean_absolute_error: 1.0840


591/Unknown  166s 276ms/step - loss: 1.9472 - mean_absolute_error: 1.0837


592/Unknown  167s 276ms/step - loss: 1.9463 - mean_absolute_error: 1.0835


593/Unknown  167s 276ms/step - loss: 1.9453 - mean_absolute_error: 1.0832


594/Unknown  167s 276ms/step - loss: 1.9444 - mean_absolute_error: 1.0830


595/Unknown  168s 276ms/step - loss: 1.9435 - mean_absolute_error: 1.0828


596/Unknown  168s 276ms/step - loss: 1.9426 - mean_absolute_error: 1.0825


597/Unknown  168s 276ms/step - loss: 1.9417 - mean_absolute_error: 1.0823


598/Unknown  169s 276ms/step - loss: 1.9408 - mean_absolute_error: 1.0820


599/Unknown  169s 276ms/step - loss: 1.9399 - mean_absolute_error: 1.0818


600/Unknown  169s 277ms/step - loss: 1.9390 - mean_absolute_error: 1.0816


601/Unknown  170s 277ms/step - loss: 1.9381 - mean_absolute_error: 1.0813


602/Unknown  170s 277ms/step - loss: 1.9372 - mean_absolute_error: 1.0811


603/Unknown  171s 277ms/step - loss: 1.9363 - mean_absolute_error: 1.0808


604/Unknown  171s 277ms/step - loss: 1.9354 - mean_absolute_error: 1.0806


605/Unknown  171s 277ms/step - loss: 1.9345 - mean_absolute_error: 1.0804


606/Unknown  171s 277ms/step - loss: 1.9337 - mean_absolute_error: 1.0801


607/Unknown  172s 278ms/step - loss: 1.9328 - mean_absolute_error: 1.0799


608/Unknown  172s 278ms/step - loss: 1.9319 - mean_absolute_error: 1.0797


609/Unknown  172s 278ms/step - loss: 1.9310 - mean_absolute_error: 1.0794


610/Unknown  173s 278ms/step - loss: 1.9301 - mean_absolute_error: 1.0792


611/Unknown  173s 278ms/step - loss: 1.9293 - mean_absolute_error: 1.0790


612/Unknown  173s 278ms/step - loss: 1.9284 - mean_absolute_error: 1.0787


613/Unknown  174s 278ms/step - loss: 1.9275 - mean_absolute_error: 1.0785


614/Unknown  174s 278ms/step - loss: 1.9267 - mean_absolute_error: 1.0783


615/Unknown  174s 278ms/step - loss: 1.9258 - mean_absolute_error: 1.0780


616/Unknown  175s 278ms/step - loss: 1.9249 - mean_absolute_error: 1.0778


617/Unknown  175s 278ms/step - loss: 1.9241 - mean_absolute_error: 1.0776


618/Unknown  175s 278ms/step - loss: 1.9232 - mean_absolute_error: 1.0774


619/Unknown  176s 278ms/step - loss: 1.9224 - mean_absolute_error: 1.0771


620/Unknown  176s 278ms/step - loss: 1.9215 - mean_absolute_error: 1.0769


621/Unknown  176s 278ms/step - loss: 1.9207 - mean_absolute_error: 1.0767


622/Unknown  176s 278ms/step - loss: 1.9198 - mean_absolute_error: 1.0764


623/Unknown  177s 278ms/step - loss: 1.9190 - mean_absolute_error: 1.0762


624/Unknown  177s 278ms/step - loss: 1.9181 - mean_absolute_error: 1.0760


625/Unknown  177s 278ms/step - loss: 1.9173 - mean_absolute_error: 1.0758


626/Unknown  178s 278ms/step - loss: 1.9165 - mean_absolute_error: 1.0755


627/Unknown  178s 278ms/step - loss: 1.9156 - mean_absolute_error: 1.0753


628/Unknown  178s 278ms/step - loss: 1.9148 - mean_absolute_error: 1.0751


629/Unknown  178s 278ms/step - loss: 1.9139 - mean_absolute_error: 1.0749


630/Unknown  179s 278ms/step - loss: 1.9131 - mean_absolute_error: 1.0747


631/Unknown  179s 278ms/step - loss: 1.9123 - mean_absolute_error: 1.0744


632/Unknown  179s 278ms/step - loss: 1.9115 - mean_absolute_error: 1.0742


633/Unknown  180s 278ms/step - loss: 1.9106 - mean_absolute_error: 1.0740


634/Unknown  180s 278ms/step - loss: 1.9098 - mean_absolute_error: 1.0738


635/Unknown  180s 278ms/step - loss: 1.9090 - mean_absolute_error: 1.0736


636/Unknown  181s 279ms/step - loss: 1.9082 - mean_absolute_error: 1.0733


637/Unknown  181s 279ms/step - loss: 1.9074 - mean_absolute_error: 1.0731


638/Unknown  181s 279ms/step - loss: 1.9065 - mean_absolute_error: 1.0729


639/Unknown  182s 279ms/step - loss: 1.9057 - mean_absolute_error: 1.0727


640/Unknown  182s 279ms/step - loss: 1.9049 - mean_absolute_error: 1.0725


641/Unknown  182s 279ms/step - loss: 1.9041 - mean_absolute_error: 1.0722


642/Unknown  183s 279ms/step - loss: 1.9033 - mean_absolute_error: 1.0720


643/Unknown  183s 279ms/step - loss: 1.9025 - mean_absolute_error: 1.0718


644/Unknown  183s 279ms/step - loss: 1.9017 - mean_absolute_error: 1.0716


645/Unknown  183s 279ms/step - loss: 1.9009 - mean_absolute_error: 1.0714


646/Unknown  184s 279ms/step - loss: 1.9001 - mean_absolute_error: 1.0712


647/Unknown  184s 279ms/step - loss: 1.8993 - mean_absolute_error: 1.0710


648/Unknown  184s 279ms/step - loss: 1.8985 - mean_absolute_error: 1.0707


649/Unknown  185s 280ms/step - loss: 1.8977 - mean_absolute_error: 1.0705


650/Unknown  185s 280ms/step - loss: 1.8969 - mean_absolute_error: 1.0703


651/Unknown  185s 280ms/step - loss: 1.8961 - mean_absolute_error: 1.0701


652/Unknown  186s 280ms/step - loss: 1.8953 - mean_absolute_error: 1.0699


653/Unknown  186s 280ms/step - loss: 1.8946 - mean_absolute_error: 1.0697


654/Unknown  186s 280ms/step - loss: 1.8938 - mean_absolute_error: 1.0695


655/Unknown  187s 280ms/step - loss: 1.8930 - mean_absolute_error: 1.0693


656/Unknown  187s 280ms/step - loss: 1.8922 - mean_absolute_error: 1.0691


657/Unknown  187s 280ms/step - loss: 1.8914 - mean_absolute_error: 1.0689


658/Unknown  188s 280ms/step - loss: 1.8907 - mean_absolute_error: 1.0686


659/Unknown  188s 281ms/step - loss: 1.8899 - mean_absolute_error: 1.0684


660/Unknown  189s 281ms/step - loss: 1.8891 - mean_absolute_error: 1.0682


661/Unknown  189s 281ms/step - loss: 1.8883 - mean_absolute_error: 1.0680


662/Unknown  189s 281ms/step - loss: 1.8876 - mean_absolute_error: 1.0678


663/Unknown  190s 281ms/step - loss: 1.8868 - mean_absolute_error: 1.0676


664/Unknown  190s 281ms/step - loss: 1.8860 - mean_absolute_error: 1.0674


665/Unknown  190s 281ms/step - loss: 1.8853 - mean_absolute_error: 1.0672


666/Unknown  191s 281ms/step - loss: 1.8845 - mean_absolute_error: 1.0670


667/Unknown  191s 281ms/step - loss: 1.8838 - mean_absolute_error: 1.0668


668/Unknown  191s 281ms/step - loss: 1.8830 - mean_absolute_error: 1.0666


669/Unknown  192s 281ms/step - loss: 1.8822 - mean_absolute_error: 1.0664


670/Unknown  192s 282ms/step - loss: 1.8815 - mean_absolute_error: 1.0662


671/Unknown  192s 282ms/step - loss: 1.8807 - mean_absolute_error: 1.0660


672/Unknown  193s 282ms/step - loss: 1.8800 - mean_absolute_error: 1.0658


673/Unknown  193s 282ms/step - loss: 1.8792 - mean_absolute_error: 1.0656


674/Unknown  193s 282ms/step - loss: 1.8785 - mean_absolute_error: 1.0654


675/Unknown  194s 282ms/step - loss: 1.8777 - mean_absolute_error: 1.0652


676/Unknown  194s 282ms/step - loss: 1.8770 - mean_absolute_error: 1.0650


677/Unknown  194s 282ms/step - loss: 1.8762 - mean_absolute_error: 1.0648


678/Unknown  195s 282ms/step - loss: 1.8755 - mean_absolute_error: 1.0646


679/Unknown  195s 282ms/step - loss: 1.8748 - mean_absolute_error: 1.0644


680/Unknown  195s 283ms/step - loss: 1.8740 - mean_absolute_error: 1.0642


681/Unknown  196s 283ms/step - loss: 1.8733 - mean_absolute_error: 1.0640


682/Unknown  196s 283ms/step - loss: 1.8726 - mean_absolute_error: 1.0638


683/Unknown  196s 283ms/step - loss: 1.8718 - mean_absolute_error: 1.0636


684/Unknown  197s 283ms/step - loss: 1.8711 - mean_absolute_error: 1.0634


685/Unknown  197s 283ms/step - loss: 1.8704 - mean_absolute_error: 1.0632


686/Unknown  197s 283ms/step - loss: 1.8697 - mean_absolute_error: 1.0630


687/Unknown  198s 283ms/step - loss: 1.8689 - mean_absolute_error: 1.0628


688/Unknown  198s 283ms/step - loss: 1.8682 - mean_absolute_error: 1.0626


689/Unknown  198s 283ms/step - loss: 1.8675 - mean_absolute_error: 1.0624


690/Unknown  199s 283ms/step - loss: 1.8668 - mean_absolute_error: 1.0622


691/Unknown  199s 283ms/step - loss: 1.8660 - mean_absolute_error: 1.0620


692/Unknown  199s 283ms/step - loss: 1.8653 - mean_absolute_error: 1.0618


693/Unknown  200s 283ms/step - loss: 1.8646 - mean_absolute_error: 1.0617


694/Unknown  200s 283ms/step - loss: 1.8639 - mean_absolute_error: 1.0615


695/Unknown  200s 284ms/step - loss: 1.8632 - mean_absolute_error: 1.0613


696/Unknown  201s 284ms/step - loss: 1.8625 - mean_absolute_error: 1.0611


697/Unknown  201s 284ms/step - loss: 1.8618 - mean_absolute_error: 1.0609


698/Unknown  201s 284ms/step - loss: 1.8611 - mean_absolute_error: 1.0607


699/Unknown  202s 284ms/step - loss: 1.8604 - mean_absolute_error: 1.0605


700/Unknown  202s 284ms/step - loss: 1.8597 - mean_absolute_error: 1.0603


701/Unknown  202s 284ms/step - loss: 1.8590 - mean_absolute_error: 1.0601


702/Unknown  203s 284ms/step - loss: 1.8583 - mean_absolute_error: 1.0599


703/Unknown  203s 284ms/step - loss: 1.8576 - mean_absolute_error: 1.0598


704/Unknown  204s 284ms/step - loss: 1.8569 - mean_absolute_error: 1.0596


705/Unknown  204s 284ms/step - loss: 1.8562 - mean_absolute_error: 1.0594


706/Unknown  204s 285ms/step - loss: 1.8555 - mean_absolute_error: 1.0592


707/Unknown  205s 285ms/step - loss: 1.8548 - mean_absolute_error: 1.0590


708/Unknown  205s 285ms/step - loss: 1.8541 - mean_absolute_error: 1.0588


709/Unknown  205s 285ms/step - loss: 1.8534 - mean_absolute_error: 1.0586


710/Unknown  206s 285ms/step - loss: 1.8527 - mean_absolute_error: 1.0585


711/Unknown  206s 285ms/step - loss: 1.8520 - mean_absolute_error: 1.0583


712/Unknown  206s 285ms/step - loss: 1.8514 - mean_absolute_error: 1.0581


713/Unknown  206s 285ms/step - loss: 1.8507 - mean_absolute_error: 1.0579


714/Unknown  207s 285ms/step - loss: 1.8500 - mean_absolute_error: 1.0577


715/Unknown  207s 285ms/step - loss: 1.8493 - mean_absolute_error: 1.0575


716/Unknown  207s 285ms/step - loss: 1.8487 - mean_absolute_error: 1.0574


717/Unknown  208s 285ms/step - loss: 1.8480 - mean_absolute_error: 1.0572


718/Unknown  208s 285ms/step - loss: 1.8473 - mean_absolute_error: 1.0570


719/Unknown  208s 285ms/step - loss: 1.8466 - mean_absolute_error: 1.0568


720/Unknown  209s 285ms/step - loss: 1.8460 - mean_absolute_error: 1.0566


721/Unknown  209s 285ms/step - loss: 1.8453 - mean_absolute_error: 1.0565


722/Unknown  209s 285ms/step - loss: 1.8446 - mean_absolute_error: 1.0563


723/Unknown  210s 285ms/step - loss: 1.8440 - mean_absolute_error: 1.0561


724/Unknown  210s 286ms/step - loss: 1.8433 - mean_absolute_error: 1.0559


725/Unknown  211s 286ms/step - loss: 1.8426 - mean_absolute_error: 1.0557


726/Unknown  211s 286ms/step - loss: 1.8420 - mean_absolute_error: 1.0556


727/Unknown  211s 286ms/step - loss: 1.8413 - mean_absolute_error: 1.0554


728/Unknown  212s 286ms/step - loss: 1.8407 - mean_absolute_error: 1.0552


729/Unknown  212s 286ms/step - loss: 1.8400 - mean_absolute_error: 1.0550


730/Unknown  212s 286ms/step - loss: 1.8394 - mean_absolute_error: 1.0549


731/Unknown  213s 286ms/step - loss: 1.8387 - mean_absolute_error: 1.0547


732/Unknown  213s 286ms/step - loss: 1.8380 - mean_absolute_error: 1.0545


733/Unknown  213s 286ms/step - loss: 1.8374 - mean_absolute_error: 1.0543


734/Unknown  214s 286ms/step - loss: 1.8367 - mean_absolute_error: 1.0541


735/Unknown  214s 286ms/step - loss: 1.8361 - mean_absolute_error: 1.0540


736/Unknown  214s 286ms/step - loss: 1.8355 - mean_absolute_error: 1.0538


737/Unknown  214s 286ms/step - loss: 1.8348 - mean_absolute_error: 1.0536


738/Unknown  215s 286ms/step - loss: 1.8342 - mean_absolute_error: 1.0535


739/Unknown  215s 286ms/step - loss: 1.8335 - mean_absolute_error: 1.0533


740/Unknown  215s 286ms/step - loss: 1.8329 - mean_absolute_error: 1.0531


741/Unknown  216s 286ms/step - loss: 1.8322 - mean_absolute_error: 1.0529


742/Unknown  216s 286ms/step - loss: 1.8316 - mean_absolute_error: 1.0528


743/Unknown  216s 286ms/step - loss: 1.8310 - mean_absolute_error: 1.0526


744/Unknown  216s 286ms/step - loss: 1.8303 - mean_absolute_error: 1.0524


745/Unknown  217s 286ms/step - loss: 1.8297 - mean_absolute_error: 1.0523


746/Unknown  217s 286ms/step - loss: 1.8291 - mean_absolute_error: 1.0521


747/Unknown  217s 286ms/step - loss: 1.8284 - mean_absolute_error: 1.0519


748/Unknown  218s 286ms/step - loss: 1.8278 - mean_absolute_error: 1.0517


749/Unknown  218s 286ms/step - loss: 1.8272 - mean_absolute_error: 1.0516


750/Unknown  218s 286ms/step - loss: 1.8266 - mean_absolute_error: 1.0514


751/Unknown  219s 286ms/step - loss: 1.8259 - mean_absolute_error: 1.0512


752/Unknown  219s 287ms/step - loss: 1.8253 - mean_absolute_error: 1.0511


753/Unknown  219s 287ms/step - loss: 1.8247 - mean_absolute_error: 1.0509


754/Unknown  219s 287ms/step - loss: 1.8241 - mean_absolute_error: 1.0507


755/Unknown  220s 287ms/step - loss: 1.8234 - mean_absolute_error: 1.0506


756/Unknown  220s 287ms/step - loss: 1.8228 - mean_absolute_error: 1.0504


757/Unknown  220s 287ms/step - loss: 1.8222 - mean_absolute_error: 1.0502


758/Unknown  221s 287ms/step - loss: 1.8216 - mean_absolute_error: 1.0501


759/Unknown  221s 287ms/step - loss: 1.8210 - mean_absolute_error: 1.0499


760/Unknown  221s 287ms/step - loss: 1.8204 - mean_absolute_error: 1.0497


761/Unknown  222s 287ms/step - loss: 1.8198 - mean_absolute_error: 1.0496


762/Unknown  222s 287ms/step - loss: 1.8192 - mean_absolute_error: 1.0494


763/Unknown  222s 287ms/step - loss: 1.8185 - mean_absolute_error: 1.0492


764/Unknown  223s 287ms/step - loss: 1.8179 - mean_absolute_error: 1.0491


765/Unknown  223s 287ms/step - loss: 1.8173 - mean_absolute_error: 1.0489


766/Unknown  223s 287ms/step - loss: 1.8167 - mean_absolute_error: 1.0487


767/Unknown  224s 287ms/step - loss: 1.8161 - mean_absolute_error: 1.0486


768/Unknown  224s 287ms/step - loss: 1.8155 - mean_absolute_error: 1.0484


769/Unknown  225s 288ms/step - loss: 1.8149 - mean_absolute_error: 1.0483


770/Unknown  225s 288ms/step - loss: 1.8143 - mean_absolute_error: 1.0481


771/Unknown  225s 288ms/step - loss: 1.8137 - mean_absolute_error: 1.0479


772/Unknown  226s 288ms/step - loss: 1.8131 - mean_absolute_error: 1.0478


773/Unknown  226s 288ms/step - loss: 1.8125 - mean_absolute_error: 1.0476


774/Unknown  226s 288ms/step - loss: 1.8119 - mean_absolute_error: 1.0474


775/Unknown  227s 288ms/step - loss: 1.8113 - mean_absolute_error: 1.0473


776/Unknown  227s 288ms/step - loss: 1.8107 - mean_absolute_error: 1.0471


777/Unknown  227s 288ms/step - loss: 1.8101 - mean_absolute_error: 1.0470


778/Unknown  228s 288ms/step - loss: 1.8096 - mean_absolute_error: 1.0468


779/Unknown  228s 288ms/step - loss: 1.8090 - mean_absolute_error: 1.0466


780/Unknown  228s 289ms/step - loss: 1.8084 - mean_absolute_error: 1.0465


781/Unknown  229s 289ms/step - loss: 1.8078 - mean_absolute_error: 1.0463


782/Unknown  229s 289ms/step - loss: 1.8072 - mean_absolute_error: 1.0462


783/Unknown  229s 289ms/step - loss: 1.8066 - mean_absolute_error: 1.0460


784/Unknown  230s 289ms/step - loss: 1.8060 - mean_absolute_error: 1.0459


785/Unknown  230s 289ms/step - loss: 1.8055 - mean_absolute_error: 1.0457


786/Unknown  231s 289ms/step - loss: 1.8049 - mean_absolute_error: 1.0455


787/Unknown  231s 289ms/step - loss: 1.8043 - mean_absolute_error: 1.0454


788/Unknown  231s 289ms/step - loss: 1.8037 - mean_absolute_error: 1.0452


789/Unknown  232s 289ms/step - loss: 1.8031 - mean_absolute_error: 1.0451


790/Unknown  232s 289ms/step - loss: 1.8026 - mean_absolute_error: 1.0449


791/Unknown  232s 289ms/step - loss: 1.8020 - mean_absolute_error: 1.0448


792/Unknown  233s 289ms/step - loss: 1.8014 - mean_absolute_error: 1.0446


793/Unknown  233s 290ms/step - loss: 1.8008 - mean_absolute_error: 1.0444


794/Unknown  233s 290ms/step - loss: 1.8003 - mean_absolute_error: 1.0443


795/Unknown  234s 290ms/step - loss: 1.7997 - mean_absolute_error: 1.0441


796/Unknown  234s 290ms/step - loss: 1.7991 - mean_absolute_error: 1.0440


797/Unknown  234s 290ms/step - loss: 1.7986 - mean_absolute_error: 1.0438


798/Unknown  235s 290ms/step - loss: 1.7980 - mean_absolute_error: 1.0437


799/Unknown  235s 290ms/step - loss: 1.7974 - mean_absolute_error: 1.0435


800/Unknown  235s 290ms/step - loss: 1.7969 - mean_absolute_error: 1.0434


801/Unknown  236s 290ms/step - loss: 1.7963 - mean_absolute_error: 1.0432


802/Unknown  236s 290ms/step - loss: 1.7957 - mean_absolute_error: 1.0431


803/Unknown  236s 290ms/step - loss: 1.7952 - mean_absolute_error: 1.0429


804/Unknown  237s 290ms/step - loss: 1.7946 - mean_absolute_error: 1.0427


805/Unknown  237s 290ms/step - loss: 1.7940 - mean_absolute_error: 1.0426


806/Unknown  238s 291ms/step - loss: 1.7935 - mean_absolute_error: 1.0424


807/Unknown  238s 291ms/step - loss: 1.7929 - mean_absolute_error: 1.0423


808/Unknown  238s 291ms/step - loss: 1.7924 - mean_absolute_error: 1.0421


809/Unknown  239s 291ms/step - loss: 1.7918 - mean_absolute_error: 1.0420


810/Unknown  239s 291ms/step - loss: 1.7913 - mean_absolute_error: 1.0418


811/Unknown  239s 291ms/step - loss: 1.7907 - mean_absolute_error: 1.0417


812/Unknown  240s 291ms/step - loss: 1.7902 - mean_absolute_error: 1.0415


813/Unknown  240s 291ms/step - loss: 1.7896 - mean_absolute_error: 1.0414


814/Unknown  241s 292ms/step - loss: 1.7891 - mean_absolute_error: 1.0412


815/Unknown  241s 292ms/step - loss: 1.7885 - mean_absolute_error: 1.0411


816/Unknown  241s 292ms/step - loss: 1.7880 - mean_absolute_error: 1.0409


817/Unknown  242s 292ms/step - loss: 1.7874 - mean_absolute_error: 1.0408


818/Unknown  242s 292ms/step - loss: 1.7869 - mean_absolute_error: 1.0406


819/Unknown  243s 292ms/step - loss: 1.7863 - mean_absolute_error: 1.0405


820/Unknown  243s 292ms/step - loss: 1.7858 - mean_absolute_error: 1.0404


821/Unknown  243s 292ms/step - loss: 1.7852 - mean_absolute_error: 1.0402


822/Unknown  244s 292ms/step - loss: 1.7847 - mean_absolute_error: 1.0401


823/Unknown  244s 292ms/step - loss: 1.7842 - mean_absolute_error: 1.0399


824/Unknown  244s 292ms/step - loss: 1.7836 - mean_absolute_error: 1.0398


825/Unknown  245s 293ms/step - loss: 1.7831 - mean_absolute_error: 1.0396


826/Unknown  245s 293ms/step - loss: 1.7825 - mean_absolute_error: 1.0395


827/Unknown  245s 293ms/step - loss: 1.7820 - mean_absolute_error: 1.0393


828/Unknown  246s 293ms/step - loss: 1.7815 - mean_absolute_error: 1.0392


829/Unknown  246s 293ms/step - loss: 1.7809 - mean_absolute_error: 1.0390


830/Unknown  246s 293ms/step - loss: 1.7804 - mean_absolute_error: 1.0389


831/Unknown  247s 293ms/step - loss: 1.7799 - mean_absolute_error: 1.0387


832/Unknown  247s 293ms/step - loss: 1.7793 - mean_absolute_error: 1.0386


833/Unknown  247s 293ms/step - loss: 1.7788 - mean_absolute_error: 1.0385


834/Unknown  247s 293ms/step - loss: 1.7783 - mean_absolute_error: 1.0383


835/Unknown  248s 293ms/step - loss: 1.7778 - mean_absolute_error: 1.0382


836/Unknown  248s 293ms/step - loss: 1.7772 - mean_absolute_error: 1.0380


837/Unknown  248s 293ms/step - loss: 1.7767 - mean_absolute_error: 1.0379


838/Unknown  249s 293ms/step - loss: 1.7762 - mean_absolute_error: 1.0377


839/Unknown  249s 293ms/step - loss: 1.7756 - mean_absolute_error: 1.0376


840/Unknown  249s 293ms/step - loss: 1.7751 - mean_absolute_error: 1.0375


841/Unknown  250s 293ms/step - loss: 1.7746 - mean_absolute_error: 1.0373


842/Unknown  250s 293ms/step - loss: 1.7741 - mean_absolute_error: 1.0372


843/Unknown  250s 293ms/step - loss: 1.7736 - mean_absolute_error: 1.0370


844/Unknown  251s 293ms/step - loss: 1.7730 - mean_absolute_error: 1.0369


845/Unknown  251s 293ms/step - loss: 1.7725 - mean_absolute_error: 1.0368


846/Unknown  252s 293ms/step - loss: 1.7720 - mean_absolute_error: 1.0366


847/Unknown  252s 293ms/step - loss: 1.7715 - mean_absolute_error: 1.0365


848/Unknown  252s 294ms/step - loss: 1.7710 - mean_absolute_error: 1.0363


849/Unknown  253s 294ms/step - loss: 1.7705 - mean_absolute_error: 1.0362


850/Unknown  253s 294ms/step - loss: 1.7699 - mean_absolute_error: 1.0360


851/Unknown  253s 294ms/step - loss: 1.7694 - mean_absolute_error: 1.0359


852/Unknown  254s 294ms/step - loss: 1.7689 - mean_absolute_error: 1.0358


853/Unknown  254s 294ms/step - loss: 1.7684 - mean_absolute_error: 1.0356


854/Unknown  254s 294ms/step - loss: 1.7679 - mean_absolute_error: 1.0355


855/Unknown  255s 294ms/step - loss: 1.7674 - mean_absolute_error: 1.0354


856/Unknown  255s 294ms/step - loss: 1.7669 - mean_absolute_error: 1.0352


857/Unknown  255s 294ms/step - loss: 1.7664 - mean_absolute_error: 1.0351


858/Unknown  256s 294ms/step - loss: 1.7659 - mean_absolute_error: 1.0349


859/Unknown  256s 294ms/step - loss: 1.7654 - mean_absolute_error: 1.0348


860/Unknown  256s 294ms/step - loss: 1.7649 - mean_absolute_error: 1.0347


861/Unknown  257s 294ms/step - loss: 1.7644 - mean_absolute_error: 1.0345


862/Unknown  257s 294ms/step - loss: 1.7639 - mean_absolute_error: 1.0344


863/Unknown  257s 294ms/step - loss: 1.7634 - mean_absolute_error: 1.0343


864/Unknown  258s 294ms/step - loss: 1.7629 - mean_absolute_error: 1.0341


865/Unknown  258s 294ms/step - loss: 1.7624 - mean_absolute_error: 1.0340


866/Unknown  258s 294ms/step - loss: 1.7619 - mean_absolute_error: 1.0338


867/Unknown  259s 294ms/step - loss: 1.7614 - mean_absolute_error: 1.0337


868/Unknown  259s 295ms/step - loss: 1.7609 - mean_absolute_error: 1.0336


869/Unknown  259s 295ms/step - loss: 1.7604 - mean_absolute_error: 1.0334


870/Unknown  260s 295ms/step - loss: 1.7599 - mean_absolute_error: 1.0333


871/Unknown  260s 295ms/step - loss: 1.7594 - mean_absolute_error: 1.0332


872/Unknown  261s 295ms/step - loss: 1.7589 - mean_absolute_error: 1.0330


873/Unknown  261s 295ms/step - loss: 1.7584 - mean_absolute_error: 1.0329


874/Unknown  261s 295ms/step - loss: 1.7579 - mean_absolute_error: 1.0328


875/Unknown  262s 295ms/step - loss: 1.7574 - mean_absolute_error: 1.0326


876/Unknown  262s 295ms/step - loss: 1.7569 - mean_absolute_error: 1.0325


877/Unknown  262s 295ms/step - loss: 1.7564 - mean_absolute_error: 1.0324


878/Unknown  263s 295ms/step - loss: 1.7559 - mean_absolute_error: 1.0322


879/Unknown  263s 295ms/step - loss: 1.7555 - mean_absolute_error: 1.0321


880/Unknown  263s 296ms/step - loss: 1.7550 - mean_absolute_error: 1.0320


881/Unknown  264s 296ms/step - loss: 1.7545 - mean_absolute_error: 1.0318


882/Unknown  264s 296ms/step - loss: 1.7540 - mean_absolute_error: 1.0317


883/Unknown  265s 296ms/step - loss: 1.7535 - mean_absolute_error: 1.0316


884/Unknown  265s 296ms/step - loss: 1.7530 - mean_absolute_error: 1.0314


885/Unknown  265s 296ms/step - loss: 1.7526 - mean_absolute_error: 1.0313


886/Unknown  266s 296ms/step - loss: 1.7521 - mean_absolute_error: 1.0312


887/Unknown  266s 296ms/step - loss: 1.7516 - mean_absolute_error: 1.0311


888/Unknown  266s 296ms/step - loss: 1.7511 - mean_absolute_error: 1.0309


889/Unknown  267s 296ms/step - loss: 1.7506 - mean_absolute_error: 1.0308


890/Unknown  267s 296ms/step - loss: 1.7502 - mean_absolute_error: 1.0307


891/Unknown  268s 296ms/step - loss: 1.7497 - mean_absolute_error: 1.0305


892/Unknown  268s 297ms/step - loss: 1.7492 - mean_absolute_error: 1.0304


893/Unknown  268s 297ms/step - loss: 1.7487 - mean_absolute_error: 1.0303


894/Unknown  269s 297ms/step - loss: 1.7483 - mean_absolute_error: 1.0301


895/Unknown  269s 297ms/step - loss: 1.7478 - mean_absolute_error: 1.0300


896/Unknown  269s 297ms/step - loss: 1.7473 - mean_absolute_error: 1.0299


897/Unknown  270s 297ms/step - loss: 1.7469 - mean_absolute_error: 1.0298


898/Unknown  270s 297ms/step - loss: 1.7464 - mean_absolute_error: 1.0296


899/Unknown  270s 297ms/step - loss: 1.7459 - mean_absolute_error: 1.0295


900/Unknown  271s 297ms/step - loss: 1.7455 - mean_absolute_error: 1.0294


901/Unknown  271s 297ms/step - loss: 1.7450 - mean_absolute_error: 1.0292


902/Unknown  272s 297ms/step - loss: 1.7445 - mean_absolute_error: 1.0291


903/Unknown  272s 297ms/step - loss: 1.7441 - mean_absolute_error: 1.0290


904/Unknown  272s 297ms/step - loss: 1.7436 - mean_absolute_error: 1.0289


905/Unknown  273s 297ms/step - loss: 1.7431 - mean_absolute_error: 1.0287


906/Unknown  273s 298ms/step - loss: 1.7427 - mean_absolute_error: 1.0286


907/Unknown  273s 298ms/step - loss: 1.7422 - mean_absolute_error: 1.0285


908/Unknown  274s 298ms/step - loss: 1.7417 - mean_absolute_error: 1.0284


909/Unknown  274s 298ms/step - loss: 1.7413 - mean_absolute_error: 1.0282


910/Unknown  275s 298ms/step - loss: 1.7408 - mean_absolute_error: 1.0281


911/Unknown  275s 298ms/step - loss: 1.7404 - mean_absolute_error: 1.0280


912/Unknown  276s 298ms/step - loss: 1.7399 - mean_absolute_error: 1.0279


913/Unknown  276s 299ms/step - loss: 1.7394 - mean_absolute_error: 1.0277


914/Unknown  276s 299ms/step - loss: 1.7390 - mean_absolute_error: 1.0276


915/Unknown  277s 299ms/step - loss: 1.7385 - mean_absolute_error: 1.0275


916/Unknown  277s 299ms/step - loss: 1.7381 - mean_absolute_error: 1.0274


917/Unknown  277s 299ms/step - loss: 1.7376 - mean_absolute_error: 1.0272


918/Unknown  278s 299ms/step - loss: 1.7372 - mean_absolute_error: 1.0271


919/Unknown  278s 299ms/step - loss: 1.7367 - mean_absolute_error: 1.0270


920/Unknown  278s 299ms/step - loss: 1.7362 - mean_absolute_error: 1.0269


921/Unknown  279s 299ms/step - loss: 1.7358 - mean_absolute_error: 1.0267


922/Unknown  279s 299ms/step - loss: 1.7353 - mean_absolute_error: 1.0266


923/Unknown  279s 299ms/step - loss: 1.7349 - mean_absolute_error: 1.0265


924/Unknown  279s 299ms/step - loss: 1.7344 - mean_absolute_error: 1.0264


925/Unknown  280s 299ms/step - loss: 1.7340 - mean_absolute_error: 1.0262


926/Unknown  280s 299ms/step - loss: 1.7335 - mean_absolute_error: 1.0261


927/Unknown  280s 299ms/step - loss: 1.7331 - mean_absolute_error: 1.0260


928/Unknown  281s 299ms/step - loss: 1.7326 - mean_absolute_error: 1.0259


929/Unknown  281s 299ms/step - loss: 1.7322 - mean_absolute_error: 1.0257


930/Unknown  281s 299ms/step - loss: 1.7317 - mean_absolute_error: 1.0256


931/Unknown  282s 299ms/step - loss: 1.7313 - mean_absolute_error: 1.0255


932/Unknown  282s 299ms/step - loss: 1.7309 - mean_absolute_error: 1.0254


933/Unknown  283s 299ms/step - loss: 1.7304 - mean_absolute_error: 1.0253


934/Unknown  283s 299ms/step - loss: 1.7300 - mean_absolute_error: 1.0251


935/Unknown  283s 299ms/step - loss: 1.7295 - mean_absolute_error: 1.0250


936/Unknown  284s 299ms/step - loss: 1.7291 - mean_absolute_error: 1.0249


937/Unknown  284s 300ms/step - loss: 1.7286 - mean_absolute_error: 1.0248


938/Unknown  284s 300ms/step - loss: 1.7282 - mean_absolute_error: 1.0247


939/Unknown  285s 300ms/step - loss: 1.7278 - mean_absolute_error: 1.0245


940/Unknown  285s 300ms/step - loss: 1.7273 - mean_absolute_error: 1.0244


941/Unknown  286s 300ms/step - loss: 1.7269 - mean_absolute_error: 1.0243


942/Unknown  286s 300ms/step - loss: 1.7264 - mean_absolute_error: 1.0242


943/Unknown  286s 300ms/step - loss: 1.7260 - mean_absolute_error: 1.0240


944/Unknown  287s 300ms/step - loss: 1.7256 - mean_absolute_error: 1.0239


945/Unknown  287s 300ms/step - loss: 1.7251 - mean_absolute_error: 1.0238


946/Unknown  287s 300ms/step - loss: 1.7247 - mean_absolute_error: 1.0237


947/Unknown  288s 300ms/step - loss: 1.7243 - mean_absolute_error: 1.0236


948/Unknown  288s 300ms/step - loss: 1.7238 - mean_absolute_error: 1.0235


949/Unknown  288s 300ms/step - loss: 1.7234 - mean_absolute_error: 1.0233


950/Unknown  289s 300ms/step - loss: 1.7230 - mean_absolute_error: 1.0232


951/Unknown  289s 300ms/step - loss: 1.7225 - mean_absolute_error: 1.0231


952/Unknown  289s 300ms/step - loss: 1.7221 - mean_absolute_error: 1.0230


953/Unknown  289s 300ms/step - loss: 1.7217 - mean_absolute_error: 1.0229


954/Unknown  290s 300ms/step - loss: 1.7212 - mean_absolute_error: 1.0227


955/Unknown  290s 300ms/step - loss: 1.7208 - mean_absolute_error: 1.0226


956/Unknown  290s 300ms/step - loss: 1.7204 - mean_absolute_error: 1.0225


957/Unknown  291s 300ms/step - loss: 1.7199 - mean_absolute_error: 1.0224


958/Unknown  291s 300ms/step - loss: 1.7195 - mean_absolute_error: 1.0223


959/Unknown  291s 300ms/step - loss: 1.7191 - mean_absolute_error: 1.0221


960/Unknown  292s 300ms/step - loss: 1.7187 - mean_absolute_error: 1.0220


961/Unknown  292s 300ms/step - loss: 1.7182 - mean_absolute_error: 1.0219


962/Unknown  292s 300ms/step - loss: 1.7178 - mean_absolute_error: 1.0218


963/Unknown  293s 300ms/step - loss: 1.7174 - mean_absolute_error: 1.0217


964/Unknown  293s 300ms/step - loss: 1.7170 - mean_absolute_error: 1.0216


965/Unknown  293s 300ms/step - loss: 1.7165 - mean_absolute_error: 1.0214


966/Unknown  294s 300ms/step - loss: 1.7161 - mean_absolute_error: 1.0213


967/Unknown  294s 301ms/step - loss: 1.7157 - mean_absolute_error: 1.0212


968/Unknown  294s 301ms/step - loss: 1.7153 - mean_absolute_error: 1.0211


969/Unknown  295s 301ms/step - loss: 1.7149 - mean_absolute_error: 1.0210


970/Unknown  295s 301ms/step - loss: 1.7144 - mean_absolute_error: 1.0209


971/Unknown  295s 301ms/step - loss: 1.7140 - mean_absolute_error: 1.0207


972/Unknown  296s 301ms/step - loss: 1.7136 - mean_absolute_error: 1.0206


973/Unknown  296s 301ms/step - loss: 1.7132 - mean_absolute_error: 1.0205


974/Unknown  296s 301ms/step - loss: 1.7128 - mean_absolute_error: 1.0204


975/Unknown  297s 301ms/step - loss: 1.7123 - mean_absolute_error: 1.0203


976/Unknown  297s 301ms/step - loss: 1.7119 - mean_absolute_error: 1.0202


977/Unknown  297s 301ms/step - loss: 1.7115 - mean_absolute_error: 1.0201


978/Unknown  298s 301ms/step - loss: 1.7111 - mean_absolute_error: 1.0199


979/Unknown  298s 301ms/step - loss: 1.7107 - mean_absolute_error: 1.0198


980/Unknown  298s 301ms/step - loss: 1.7103 - mean_absolute_error: 1.0197


981/Unknown  299s 301ms/step - loss: 1.7099 - mean_absolute_error: 1.0196


982/Unknown  299s 301ms/step - loss: 1.7094 - mean_absolute_error: 1.0195


983/Unknown  299s 301ms/step - loss: 1.7090 - mean_absolute_error: 1.0194


984/Unknown  300s 301ms/step - loss: 1.7086 - mean_absolute_error: 1.0193


985/Unknown  300s 301ms/step - loss: 1.7082 - mean_absolute_error: 1.0191


986/Unknown  301s 301ms/step - loss: 1.7078 - mean_absolute_error: 1.0190


987/Unknown  301s 302ms/step - loss: 1.7074 - mean_absolute_error: 1.0189


988/Unknown  301s 302ms/step - loss: 1.7070 - mean_absolute_error: 1.0188


989/Unknown  302s 302ms/step - loss: 1.7066 - mean_absolute_error: 1.0187


990/Unknown  302s 302ms/step - loss: 1.7062 - mean_absolute_error: 1.0186


991/Unknown  302s 302ms/step - loss: 1.7058 - mean_absolute_error: 1.0185


992/Unknown  303s 302ms/step - loss: 1.7054 - mean_absolute_error: 1.0184


993/Unknown  303s 302ms/step - loss: 1.7050 - mean_absolute_error: 1.0182


994/Unknown  303s 302ms/step - loss: 1.7046 - mean_absolute_error: 1.0181


995/Unknown  304s 302ms/step - loss: 1.7042 - mean_absolute_error: 1.0180


996/Unknown  304s 302ms/step - loss: 1.7037 - mean_absolute_error: 1.0179


997/Unknown  305s 302ms/step - loss: 1.7033 - mean_absolute_error: 1.0178


998/Unknown  305s 302ms/step - loss: 1.7029 - mean_absolute_error: 1.0177


999/Unknown  305s 302ms/step - loss: 1.7025 - mean_absolute_error: 1.0176


```
</div>
   1000/Unknown  306s 302ms/step - loss: 1.7021 - mean_absolute_error: 1.0175

<div class="k-default-codeblock">
```

```
</div>
   1001/Unknown  306s 302ms/step - loss: 1.7017 - mean_absolute_error: 1.0173

<div class="k-default-codeblock">
```

```
</div>
   1002/Unknown  306s 302ms/step - loss: 1.7013 - mean_absolute_error: 1.0172

<div class="k-default-codeblock">
```

```
</div>
   1003/Unknown  307s 303ms/step - loss: 1.7009 - mean_absolute_error: 1.0171

<div class="k-default-codeblock">
```

```
</div>
   1004/Unknown  307s 303ms/step - loss: 1.7005 - mean_absolute_error: 1.0170

<div class="k-default-codeblock">
```

```
</div>
   1005/Unknown  307s 303ms/step - loss: 1.7002 - mean_absolute_error: 1.0169

<div class="k-default-codeblock">
```

```
</div>
   1006/Unknown  308s 303ms/step - loss: 1.6998 - mean_absolute_error: 1.0168

<div class="k-default-codeblock">
```

```
</div>
   1007/Unknown  308s 303ms/step - loss: 1.6994 - mean_absolute_error: 1.0167

<div class="k-default-codeblock">
```

```
</div>
   1008/Unknown  308s 303ms/step - loss: 1.6990 - mean_absolute_error: 1.0166

<div class="k-default-codeblock">
```

```
</div>
   1009/Unknown  309s 303ms/step - loss: 1.6986 - mean_absolute_error: 1.0165

<div class="k-default-codeblock">
```

```
</div>
   1010/Unknown  309s 303ms/step - loss: 1.6982 - mean_absolute_error: 1.0164

<div class="k-default-codeblock">
```

```
</div>
   1011/Unknown  309s 303ms/step - loss: 1.6978 - mean_absolute_error: 1.0162

<div class="k-default-codeblock">
```

```
</div>
   1012/Unknown  310s 303ms/step - loss: 1.6974 - mean_absolute_error: 1.0161

<div class="k-default-codeblock">
```

```
</div>
   1013/Unknown  310s 303ms/step - loss: 1.6970 - mean_absolute_error: 1.0160

<div class="k-default-codeblock">
```

```
</div>
   1014/Unknown  311s 303ms/step - loss: 1.6966 - mean_absolute_error: 1.0159

<div class="k-default-codeblock">
```

```
</div>
   1015/Unknown  311s 303ms/step - loss: 1.6962 - mean_absolute_error: 1.0158

<div class="k-default-codeblock">
```

```
</div>
   1016/Unknown  311s 303ms/step - loss: 1.6958 - mean_absolute_error: 1.0157

<div class="k-default-codeblock">
```

```
</div>
   1017/Unknown  312s 303ms/step - loss: 1.6954 - mean_absolute_error: 1.0156

<div class="k-default-codeblock">
```

```
</div>
   1018/Unknown  312s 303ms/step - loss: 1.6950 - mean_absolute_error: 1.0155

<div class="k-default-codeblock">
```

```
</div>
   1019/Unknown  312s 303ms/step - loss: 1.6946 - mean_absolute_error: 1.0154

<div class="k-default-codeblock">
```

```
</div>
   1020/Unknown  313s 303ms/step - loss: 1.6943 - mean_absolute_error: 1.0153

<div class="k-default-codeblock">
```

```
</div>
   1021/Unknown  313s 303ms/step - loss: 1.6939 - mean_absolute_error: 1.0152

<div class="k-default-codeblock">
```

```
</div>
   1022/Unknown  314s 304ms/step - loss: 1.6935 - mean_absolute_error: 1.0151

<div class="k-default-codeblock">
```

```
</div>
   1023/Unknown  314s 304ms/step - loss: 1.6931 - mean_absolute_error: 1.0149

<div class="k-default-codeblock">
```

```
</div>
   1024/Unknown  314s 304ms/step - loss: 1.6927 - mean_absolute_error: 1.0148

<div class="k-default-codeblock">
```

```
</div>
   1025/Unknown  315s 304ms/step - loss: 1.6923 - mean_absolute_error: 1.0147

<div class="k-default-codeblock">
```

```
</div>
   1026/Unknown  315s 304ms/step - loss: 1.6919 - mean_absolute_error: 1.0146

<div class="k-default-codeblock">
```

```
</div>
   1027/Unknown  315s 304ms/step - loss: 1.6915 - mean_absolute_error: 1.0145

<div class="k-default-codeblock">
```

```
</div>
   1028/Unknown  316s 304ms/step - loss: 1.6912 - mean_absolute_error: 1.0144

<div class="k-default-codeblock">
```

```
</div>
   1029/Unknown  316s 304ms/step - loss: 1.6908 - mean_absolute_error: 1.0143

<div class="k-default-codeblock">
```

```
</div>
   1030/Unknown  316s 304ms/step - loss: 1.6904 - mean_absolute_error: 1.0142

<div class="k-default-codeblock">
```

```
</div>
   1031/Unknown  317s 304ms/step - loss: 1.6900 - mean_absolute_error: 1.0141

<div class="k-default-codeblock">
```

```
</div>
   1032/Unknown  317s 304ms/step - loss: 1.6896 - mean_absolute_error: 1.0140

<div class="k-default-codeblock">
```

```
</div>
   1033/Unknown  317s 304ms/step - loss: 1.6892 - mean_absolute_error: 1.0139

<div class="k-default-codeblock">
```

```
</div>
   1034/Unknown  318s 304ms/step - loss: 1.6889 - mean_absolute_error: 1.0138

<div class="k-default-codeblock">
```

```
</div>
   1035/Unknown  318s 304ms/step - loss: 1.6885 - mean_absolute_error: 1.0137

<div class="k-default-codeblock">
```

```
</div>
   1036/Unknown  319s 304ms/step - loss: 1.6881 - mean_absolute_error: 1.0136

<div class="k-default-codeblock">
```

```
</div>
   1037/Unknown  319s 304ms/step - loss: 1.6877 - mean_absolute_error: 1.0135

<div class="k-default-codeblock">
```

```
</div>
   1038/Unknown  319s 304ms/step - loss: 1.6873 - mean_absolute_error: 1.0133

<div class="k-default-codeblock">
```

```
</div>
   1039/Unknown  320s 304ms/step - loss: 1.6870 - mean_absolute_error: 1.0132

<div class="k-default-codeblock">
```

```
</div>
   1040/Unknown  320s 304ms/step - loss: 1.6866 - mean_absolute_error: 1.0131

<div class="k-default-codeblock">
```

```
</div>
   1041/Unknown  320s 304ms/step - loss: 1.6862 - mean_absolute_error: 1.0130

<div class="k-default-codeblock">
```

```
</div>
   1042/Unknown  321s 304ms/step - loss: 1.6858 - mean_absolute_error: 1.0129

<div class="k-default-codeblock">
```

```
</div>
   1043/Unknown  321s 305ms/step - loss: 1.6855 - mean_absolute_error: 1.0128

<div class="k-default-codeblock">
```

```
</div>
   1044/Unknown  321s 305ms/step - loss: 1.6851 - mean_absolute_error: 1.0127

<div class="k-default-codeblock">
```

```
</div>
   1045/Unknown  322s 305ms/step - loss: 1.6847 - mean_absolute_error: 1.0126

<div class="k-default-codeblock">
```

```
</div>
   1046/Unknown  322s 305ms/step - loss: 1.6843 - mean_absolute_error: 1.0125

<div class="k-default-codeblock">
```

```
</div>
   1047/Unknown  322s 305ms/step - loss: 1.6840 - mean_absolute_error: 1.0124

<div class="k-default-codeblock">
```

```
</div>
   1048/Unknown  323s 305ms/step - loss: 1.6836 - mean_absolute_error: 1.0123

<div class="k-default-codeblock">
```

```
</div>
   1049/Unknown  323s 305ms/step - loss: 1.6832 - mean_absolute_error: 1.0122

<div class="k-default-codeblock">
```

```
</div>
   1050/Unknown  323s 305ms/step - loss: 1.6828 - mean_absolute_error: 1.0121

<div class="k-default-codeblock">
```

```
</div>
   1051/Unknown  324s 305ms/step - loss: 1.6825 - mean_absolute_error: 1.0120

<div class="k-default-codeblock">
```

```
</div>
   1052/Unknown  324s 305ms/step - loss: 1.6821 - mean_absolute_error: 1.0119

<div class="k-default-codeblock">
```

```
</div>
   1053/Unknown  325s 305ms/step - loss: 1.6817 - mean_absolute_error: 1.0118

<div class="k-default-codeblock">
```

```
</div>
   1054/Unknown  325s 305ms/step - loss: 1.6813 - mean_absolute_error: 1.0117

<div class="k-default-codeblock">
```

```
</div>
   1055/Unknown  325s 305ms/step - loss: 1.6810 - mean_absolute_error: 1.0116

<div class="k-default-codeblock">
```

```
</div>
   1056/Unknown  326s 305ms/step - loss: 1.6806 - mean_absolute_error: 1.0115

<div class="k-default-codeblock">
```

```
</div>
   1057/Unknown  326s 305ms/step - loss: 1.6802 - mean_absolute_error: 1.0114

<div class="k-default-codeblock">
```

```
</div>
   1058/Unknown  326s 305ms/step - loss: 1.6799 - mean_absolute_error: 1.0113

<div class="k-default-codeblock">
```

```
</div>
   1059/Unknown  327s 305ms/step - loss: 1.6795 - mean_absolute_error: 1.0112

<div class="k-default-codeblock">
```

```
</div>
   1060/Unknown  327s 305ms/step - loss: 1.6791 - mean_absolute_error: 1.0111

<div class="k-default-codeblock">
```

```
</div>
   1061/Unknown  327s 306ms/step - loss: 1.6788 - mean_absolute_error: 1.0110

<div class="k-default-codeblock">
```

```
</div>
   1062/Unknown  328s 306ms/step - loss: 1.6784 - mean_absolute_error: 1.0109

<div class="k-default-codeblock">
```

```
</div>
   1063/Unknown  328s 306ms/step - loss: 1.6780 - mean_absolute_error: 1.0107

<div class="k-default-codeblock">
```

```
</div>
   1064/Unknown  329s 306ms/step - loss: 1.6777 - mean_absolute_error: 1.0106

<div class="k-default-codeblock">
```

```
</div>
   1065/Unknown  329s 306ms/step - loss: 1.6773 - mean_absolute_error: 1.0105

<div class="k-default-codeblock">
```

```
</div>
   1066/Unknown  329s 306ms/step - loss: 1.6769 - mean_absolute_error: 1.0104

<div class="k-default-codeblock">
```

```
</div>
   1067/Unknown  330s 306ms/step - loss: 1.6766 - mean_absolute_error: 1.0103

<div class="k-default-codeblock">
```

```
</div>
   1068/Unknown  330s 306ms/step - loss: 1.6762 - mean_absolute_error: 1.0102

<div class="k-default-codeblock">
```

```
</div>
   1069/Unknown  330s 306ms/step - loss: 1.6759 - mean_absolute_error: 1.0101

<div class="k-default-codeblock">
```

```
</div>
   1070/Unknown  331s 306ms/step - loss: 1.6755 - mean_absolute_error: 1.0100

<div class="k-default-codeblock">
```

```
</div>
   1071/Unknown  331s 306ms/step - loss: 1.6751 - mean_absolute_error: 1.0099

<div class="k-default-codeblock">
```

```
</div>
   1072/Unknown  331s 306ms/step - loss: 1.6748 - mean_absolute_error: 1.0098

<div class="k-default-codeblock">
```

```
</div>
   1073/Unknown  332s 306ms/step - loss: 1.6744 - mean_absolute_error: 1.0097

<div class="k-default-codeblock">
```

```
</div>
   1074/Unknown  332s 306ms/step - loss: 1.6740 - mean_absolute_error: 1.0096

<div class="k-default-codeblock">
```

```
</div>
   1075/Unknown  332s 306ms/step - loss: 1.6737 - mean_absolute_error: 1.0095

<div class="k-default-codeblock">
```

```
</div>
   1076/Unknown  333s 306ms/step - loss: 1.6733 - mean_absolute_error: 1.0094

<div class="k-default-codeblock">
```

```
</div>
   1077/Unknown  333s 306ms/step - loss: 1.6730 - mean_absolute_error: 1.0093

<div class="k-default-codeblock">
```

```
</div>
   1078/Unknown  333s 306ms/step - loss: 1.6726 - mean_absolute_error: 1.0092

<div class="k-default-codeblock">
```

```
</div>
   1079/Unknown  334s 306ms/step - loss: 1.6723 - mean_absolute_error: 1.0091

<div class="k-default-codeblock">
```

```
</div>
   1080/Unknown  334s 306ms/step - loss: 1.6719 - mean_absolute_error: 1.0090

<div class="k-default-codeblock">
```

```
</div>
   1081/Unknown  334s 306ms/step - loss: 1.6715 - mean_absolute_error: 1.0089

<div class="k-default-codeblock">
```

```
</div>
   1082/Unknown  335s 306ms/step - loss: 1.6712 - mean_absolute_error: 1.0088

<div class="k-default-codeblock">
```

```
</div>
   1083/Unknown  335s 306ms/step - loss: 1.6708 - mean_absolute_error: 1.0087

<div class="k-default-codeblock">
```

```
</div>
   1084/Unknown  335s 306ms/step - loss: 1.6705 - mean_absolute_error: 1.0086

<div class="k-default-codeblock">
```

```
</div>
   1085/Unknown  336s 306ms/step - loss: 1.6701 - mean_absolute_error: 1.0085

<div class="k-default-codeblock">
```

```
</div>
   1086/Unknown  336s 306ms/step - loss: 1.6698 - mean_absolute_error: 1.0084

<div class="k-default-codeblock">
```

```
</div>
   1087/Unknown  337s 307ms/step - loss: 1.6694 - mean_absolute_error: 1.0083

<div class="k-default-codeblock">
```

```
</div>
   1088/Unknown  337s 307ms/step - loss: 1.6691 - mean_absolute_error: 1.0082

<div class="k-default-codeblock">
```

```
</div>
   1089/Unknown  337s 307ms/step - loss: 1.6687 - mean_absolute_error: 1.0081

<div class="k-default-codeblock">
```

```
</div>
   1090/Unknown  338s 307ms/step - loss: 1.6684 - mean_absolute_error: 1.0080

<div class="k-default-codeblock">
```

```
</div>
   1091/Unknown  338s 307ms/step - loss: 1.6680 - mean_absolute_error: 1.0079

<div class="k-default-codeblock">
```

```
</div>
   1092/Unknown  338s 307ms/step - loss: 1.6677 - mean_absolute_error: 1.0078

<div class="k-default-codeblock">
```

```
</div>
   1093/Unknown  339s 307ms/step - loss: 1.6673 - mean_absolute_error: 1.0077

<div class="k-default-codeblock">
```

```
</div>
   1094/Unknown  339s 307ms/step - loss: 1.6670 - mean_absolute_error: 1.0076

<div class="k-default-codeblock">
```

```
</div>
   1095/Unknown  339s 307ms/step - loss: 1.6666 - mean_absolute_error: 1.0076

<div class="k-default-codeblock">
```

```
</div>
   1096/Unknown  340s 307ms/step - loss: 1.6663 - mean_absolute_error: 1.0075

<div class="k-default-codeblock">
```

```
</div>
   1097/Unknown  340s 307ms/step - loss: 1.6659 - mean_absolute_error: 1.0074

<div class="k-default-codeblock">
```

```
</div>
   1098/Unknown  340s 307ms/step - loss: 1.6656 - mean_absolute_error: 1.0073

<div class="k-default-codeblock">
```

```
</div>
   1099/Unknown  341s 307ms/step - loss: 1.6652 - mean_absolute_error: 1.0072

<div class="k-default-codeblock">
```

```
</div>
   1100/Unknown  341s 307ms/step - loss: 1.6649 - mean_absolute_error: 1.0071

<div class="k-default-codeblock">
```

```
</div>
   1101/Unknown  342s 307ms/step - loss: 1.6645 - mean_absolute_error: 1.0070

<div class="k-default-codeblock">
```

```
</div>
   1102/Unknown  342s 307ms/step - loss: 1.6642 - mean_absolute_error: 1.0069

<div class="k-default-codeblock">
```

```
</div>
   1103/Unknown  342s 307ms/step - loss: 1.6638 - mean_absolute_error: 1.0068

<div class="k-default-codeblock">
```

```
</div>
   1104/Unknown  343s 307ms/step - loss: 1.6635 - mean_absolute_error: 1.0067

<div class="k-default-codeblock">
```

```
</div>
   1105/Unknown  343s 307ms/step - loss: 1.6631 - mean_absolute_error: 1.0066

<div class="k-default-codeblock">
```

```
</div>
   1106/Unknown  343s 307ms/step - loss: 1.6628 - mean_absolute_error: 1.0065

<div class="k-default-codeblock">
```

```
</div>
   1107/Unknown  344s 307ms/step - loss: 1.6625 - mean_absolute_error: 1.0064

<div class="k-default-codeblock">
```

```
</div>
   1108/Unknown  344s 307ms/step - loss: 1.6621 - mean_absolute_error: 1.0063

<div class="k-default-codeblock">
```

```
</div>
   1109/Unknown  344s 307ms/step - loss: 1.6618 - mean_absolute_error: 1.0062

<div class="k-default-codeblock">
```

```
</div>
   1110/Unknown  345s 307ms/step - loss: 1.6614 - mean_absolute_error: 1.0061

<div class="k-default-codeblock">
```

```
</div>
   1111/Unknown  345s 307ms/step - loss: 1.6611 - mean_absolute_error: 1.0060

<div class="k-default-codeblock">
```

```
</div>
   1112/Unknown  345s 308ms/step - loss: 1.6607 - mean_absolute_error: 1.0059

<div class="k-default-codeblock">
```

```
</div>
   1113/Unknown  346s 308ms/step - loss: 1.6604 - mean_absolute_error: 1.0058

<div class="k-default-codeblock">
```

```
</div>
   1114/Unknown  346s 308ms/step - loss: 1.6601 - mean_absolute_error: 1.0057

<div class="k-default-codeblock">
```

```
</div>
   1115/Unknown  346s 308ms/step - loss: 1.6597 - mean_absolute_error: 1.0056

<div class="k-default-codeblock">
```

```
</div>
   1116/Unknown  347s 308ms/step - loss: 1.6594 - mean_absolute_error: 1.0055

<div class="k-default-codeblock">
```

```
</div>
   1117/Unknown  347s 308ms/step - loss: 1.6590 - mean_absolute_error: 1.0054

<div class="k-default-codeblock">
```

```
</div>
   1118/Unknown  347s 308ms/step - loss: 1.6587 - mean_absolute_error: 1.0053

<div class="k-default-codeblock">
```

```
</div>
   1119/Unknown  348s 308ms/step - loss: 1.6584 - mean_absolute_error: 1.0052

<div class="k-default-codeblock">
```

```
</div>
   1120/Unknown  348s 308ms/step - loss: 1.6580 - mean_absolute_error: 1.0051

<div class="k-default-codeblock">
```

```
</div>
   1121/Unknown  349s 308ms/step - loss: 1.6577 - mean_absolute_error: 1.0050

<div class="k-default-codeblock">
```

```
</div>
   1122/Unknown  349s 308ms/step - loss: 1.6573 - mean_absolute_error: 1.0049

<div class="k-default-codeblock">
```

```
</div>
   1123/Unknown  349s 308ms/step - loss: 1.6570 - mean_absolute_error: 1.0048

<div class="k-default-codeblock">
```

```
</div>
   1124/Unknown  350s 308ms/step - loss: 1.6567 - mean_absolute_error: 1.0048

<div class="k-default-codeblock">
```

```
</div>
   1125/Unknown  350s 308ms/step - loss: 1.6563 - mean_absolute_error: 1.0047

<div class="k-default-codeblock">
```

```
</div>
   1126/Unknown  350s 308ms/step - loss: 1.6560 - mean_absolute_error: 1.0046

<div class="k-default-codeblock">
```

```
</div>
   1127/Unknown  351s 308ms/step - loss: 1.6557 - mean_absolute_error: 1.0045

<div class="k-default-codeblock">
```

```
</div>
   1128/Unknown  351s 308ms/step - loss: 1.6553 - mean_absolute_error: 1.0044

<div class="k-default-codeblock">
```

```
</div>
   1129/Unknown  352s 308ms/step - loss: 1.6550 - mean_absolute_error: 1.0043

<div class="k-default-codeblock">
```

```
</div>
   1130/Unknown  352s 308ms/step - loss: 1.6546 - mean_absolute_error: 1.0042

<div class="k-default-codeblock">
```

```
</div>
   1131/Unknown  352s 308ms/step - loss: 1.6543 - mean_absolute_error: 1.0041

<div class="k-default-codeblock">
```

```
</div>
   1132/Unknown  352s 308ms/step - loss: 1.6540 - mean_absolute_error: 1.0040

<div class="k-default-codeblock">
```

```
</div>
   1133/Unknown  353s 308ms/step - loss: 1.6536 - mean_absolute_error: 1.0039

<div class="k-default-codeblock">
```

```
</div>
   1134/Unknown  353s 308ms/step - loss: 1.6533 - mean_absolute_error: 1.0038

<div class="k-default-codeblock">
```

```
</div>
   1135/Unknown  353s 308ms/step - loss: 1.6530 - mean_absolute_error: 1.0037

<div class="k-default-codeblock">
```

```
</div>
   1136/Unknown  354s 308ms/step - loss: 1.6527 - mean_absolute_error: 1.0036

<div class="k-default-codeblock">
```

```
</div>
   1137/Unknown  354s 308ms/step - loss: 1.6523 - mean_absolute_error: 1.0035

<div class="k-default-codeblock">
```

```
</div>
   1138/Unknown  354s 308ms/step - loss: 1.6520 - mean_absolute_error: 1.0034

<div class="k-default-codeblock">
```

```
</div>
   1139/Unknown  355s 308ms/step - loss: 1.6517 - mean_absolute_error: 1.0033

<div class="k-default-codeblock">
```

```
</div>
   1140/Unknown  355s 308ms/step - loss: 1.6513 - mean_absolute_error: 1.0032

<div class="k-default-codeblock">
```

```
</div>
   1141/Unknown  355s 309ms/step - loss: 1.6510 - mean_absolute_error: 1.0032

<div class="k-default-codeblock">
```

```
</div>
   1142/Unknown  356s 309ms/step - loss: 1.6507 - mean_absolute_error: 1.0031

<div class="k-default-codeblock">
```

```
</div>
   1143/Unknown  356s 309ms/step - loss: 1.6503 - mean_absolute_error: 1.0030

<div class="k-default-codeblock">
```

```
</div>
   1144/Unknown  356s 309ms/step - loss: 1.6500 - mean_absolute_error: 1.0029

<div class="k-default-codeblock">
```

```
</div>
   1145/Unknown  357s 309ms/step - loss: 1.6497 - mean_absolute_error: 1.0028

<div class="k-default-codeblock">
```

```
</div>
   1146/Unknown  357s 309ms/step - loss: 1.6494 - mean_absolute_error: 1.0027

<div class="k-default-codeblock">
```

```
</div>
   1147/Unknown  357s 309ms/step - loss: 1.6490 - mean_absolute_error: 1.0026

<div class="k-default-codeblock">
```

```
</div>
   1148/Unknown  358s 309ms/step - loss: 1.6487 - mean_absolute_error: 1.0025

<div class="k-default-codeblock">
```

```
</div>
   1149/Unknown  358s 309ms/step - loss: 1.6484 - mean_absolute_error: 1.0024

<div class="k-default-codeblock">
```

```
</div>
   1150/Unknown  358s 309ms/step - loss: 1.6481 - mean_absolute_error: 1.0023

<div class="k-default-codeblock">
```

```
</div>
   1151/Unknown  359s 309ms/step - loss: 1.6477 - mean_absolute_error: 1.0022

<div class="k-default-codeblock">
```

```
</div>
   1152/Unknown  359s 309ms/step - loss: 1.6474 - mean_absolute_error: 1.0021

<div class="k-default-codeblock">
```

```
</div>
   1153/Unknown  359s 309ms/step - loss: 1.6471 - mean_absolute_error: 1.0020

<div class="k-default-codeblock">
```

```
</div>
   1154/Unknown  360s 309ms/step - loss: 1.6468 - mean_absolute_error: 1.0019

<div class="k-default-codeblock">
```

```
</div>
   1155/Unknown  360s 309ms/step - loss: 1.6464 - mean_absolute_error: 1.0019

<div class="k-default-codeblock">
```

```
</div>
   1156/Unknown  361s 309ms/step - loss: 1.6461 - mean_absolute_error: 1.0018

<div class="k-default-codeblock">
```

```
</div>
   1157/Unknown  361s 309ms/step - loss: 1.6458 - mean_absolute_error: 1.0017

<div class="k-default-codeblock">
```

```
</div>
   1158/Unknown  361s 309ms/step - loss: 1.6455 - mean_absolute_error: 1.0016

<div class="k-default-codeblock">
```

```
</div>
   1159/Unknown  362s 309ms/step - loss: 1.6451 - mean_absolute_error: 1.0015

<div class="k-default-codeblock">
```

```
</div>
   1160/Unknown  362s 309ms/step - loss: 1.6448 - mean_absolute_error: 1.0014

<div class="k-default-codeblock">
```

```
</div>
   1161/Unknown  362s 309ms/step - loss: 1.6445 - mean_absolute_error: 1.0013

<div class="k-default-codeblock">
```

```
</div>
   1162/Unknown  363s 309ms/step - loss: 1.6442 - mean_absolute_error: 1.0012

<div class="k-default-codeblock">
```

```
</div>
   1163/Unknown  363s 309ms/step - loss: 1.6439 - mean_absolute_error: 1.0011

<div class="k-default-codeblock">
```

```
</div>
   1164/Unknown  364s 309ms/step - loss: 1.6435 - mean_absolute_error: 1.0010

<div class="k-default-codeblock">
```

```
</div>
   1165/Unknown  364s 310ms/step - loss: 1.6432 - mean_absolute_error: 1.0009

<div class="k-default-codeblock">
```

```
</div>
   1166/Unknown  364s 310ms/step - loss: 1.6429 - mean_absolute_error: 1.0009

<div class="k-default-codeblock">
```

```
</div>
   1167/Unknown  365s 310ms/step - loss: 1.6426 - mean_absolute_error: 1.0008

<div class="k-default-codeblock">
```

```
</div>
   1168/Unknown  365s 310ms/step - loss: 1.6423 - mean_absolute_error: 1.0007

<div class="k-default-codeblock">
```

```
</div>
   1169/Unknown  365s 310ms/step - loss: 1.6419 - mean_absolute_error: 1.0006

<div class="k-default-codeblock">
```

```
</div>
   1170/Unknown  366s 310ms/step - loss: 1.6416 - mean_absolute_error: 1.0005

<div class="k-default-codeblock">
```

```
</div>
   1171/Unknown  366s 310ms/step - loss: 1.6413 - mean_absolute_error: 1.0004

<div class="k-default-codeblock">
```

```
</div>
   1172/Unknown  367s 310ms/step - loss: 1.6410 - mean_absolute_error: 1.0003

<div class="k-default-codeblock">
```

```
</div>
   1173/Unknown  367s 310ms/step - loss: 1.6407 - mean_absolute_error: 1.0002

<div class="k-default-codeblock">
```

```
</div>
   1174/Unknown  367s 310ms/step - loss: 1.6404 - mean_absolute_error: 1.0001

<div class="k-default-codeblock">
```

```
</div>
   1175/Unknown  368s 310ms/step - loss: 1.6400 - mean_absolute_error: 1.0000

<div class="k-default-codeblock">
```

```
</div>
   1176/Unknown  368s 310ms/step - loss: 1.6397 - mean_absolute_error: 1.0000

<div class="k-default-codeblock">
```

```
</div>
   1177/Unknown  368s 310ms/step - loss: 1.6394 - mean_absolute_error: 0.9999

<div class="k-default-codeblock">
```

```
</div>
   1178/Unknown  369s 310ms/step - loss: 1.6391 - mean_absolute_error: 0.9998

<div class="k-default-codeblock">
```

```
</div>
   1179/Unknown  369s 310ms/step - loss: 1.6388 - mean_absolute_error: 0.9997

<div class="k-default-codeblock">
```

```
</div>
   1180/Unknown  369s 310ms/step - loss: 1.6385 - mean_absolute_error: 0.9996

<div class="k-default-codeblock">
```

```
</div>
   1181/Unknown  370s 310ms/step - loss: 1.6382 - mean_absolute_error: 0.9995

<div class="k-default-codeblock">
```

```
</div>
   1182/Unknown  370s 310ms/step - loss: 1.6378 - mean_absolute_error: 0.9994

<div class="k-default-codeblock">
```

```
</div>
   1183/Unknown  371s 310ms/step - loss: 1.6375 - mean_absolute_error: 0.9993

<div class="k-default-codeblock">
```

```
</div>
   1184/Unknown  371s 310ms/step - loss: 1.6372 - mean_absolute_error: 0.9992

<div class="k-default-codeblock">
```

```
</div>
   1185/Unknown  371s 310ms/step - loss: 1.6369 - mean_absolute_error: 0.9992

<div class="k-default-codeblock">
```

```
</div>
   1186/Unknown  372s 310ms/step - loss: 1.6366 - mean_absolute_error: 0.9991

<div class="k-default-codeblock">
```

```
</div>
   1187/Unknown  372s 311ms/step - loss: 1.6363 - mean_absolute_error: 0.9990

<div class="k-default-codeblock">
```

```
</div>
   1188/Unknown  372s 311ms/step - loss: 1.6360 - mean_absolute_error: 0.9989

<div class="k-default-codeblock">
```

```
</div>
   1189/Unknown  373s 311ms/step - loss: 1.6357 - mean_absolute_error: 0.9988

<div class="k-default-codeblock">
```

```
</div>
   1190/Unknown  373s 311ms/step - loss: 1.6354 - mean_absolute_error: 0.9987

<div class="k-default-codeblock">
```

```
</div>
   1191/Unknown  373s 311ms/step - loss: 1.6351 - mean_absolute_error: 0.9986

<div class="k-default-codeblock">
```

```
</div>
   1192/Unknown  374s 311ms/step - loss: 1.6347 - mean_absolute_error: 0.9985

<div class="k-default-codeblock">
```

```
</div>
   1193/Unknown  374s 311ms/step - loss: 1.6344 - mean_absolute_error: 0.9985

<div class="k-default-codeblock">
```

```
</div>
   1194/Unknown  375s 311ms/step - loss: 1.6341 - mean_absolute_error: 0.9984

<div class="k-default-codeblock">
```

```
</div>
   1195/Unknown  375s 311ms/step - loss: 1.6338 - mean_absolute_error: 0.9983

<div class="k-default-codeblock">
```

```
</div>
   1196/Unknown  375s 311ms/step - loss: 1.6335 - mean_absolute_error: 0.9982

<div class="k-default-codeblock">
```

```
</div>
   1197/Unknown  376s 311ms/step - loss: 1.6332 - mean_absolute_error: 0.9981

<div class="k-default-codeblock">
```

```
</div>
   1198/Unknown  376s 311ms/step - loss: 1.6329 - mean_absolute_error: 0.9980

<div class="k-default-codeblock">
```

```
</div>
   1199/Unknown  376s 311ms/step - loss: 1.6326 - mean_absolute_error: 0.9979

<div class="k-default-codeblock">
```

```
</div>
   1200/Unknown  377s 311ms/step - loss: 1.6323 - mean_absolute_error: 0.9978

<div class="k-default-codeblock">
```

```
</div>
   1201/Unknown  377s 311ms/step - loss: 1.6320 - mean_absolute_error: 0.9978

<div class="k-default-codeblock">
```

```
</div>
   1202/Unknown  377s 311ms/step - loss: 1.6317 - mean_absolute_error: 0.9977

<div class="k-default-codeblock">
```

```
</div>
   1203/Unknown  378s 311ms/step - loss: 1.6314 - mean_absolute_error: 0.9976

<div class="k-default-codeblock">
```

```
</div>
   1204/Unknown  378s 311ms/step - loss: 1.6311 - mean_absolute_error: 0.9975

<div class="k-default-codeblock">
```

```
</div>
   1205/Unknown  378s 311ms/step - loss: 1.6308 - mean_absolute_error: 0.9974

<div class="k-default-codeblock">
```

```
</div>
   1206/Unknown  379s 311ms/step - loss: 1.6305 - mean_absolute_error: 0.9973

<div class="k-default-codeblock">
```

```
</div>
   1207/Unknown  379s 311ms/step - loss: 1.6302 - mean_absolute_error: 0.9972

<div class="k-default-codeblock">
```

```
</div>
   1208/Unknown  380s 311ms/step - loss: 1.6299 - mean_absolute_error: 0.9972

<div class="k-default-codeblock">
```

```
</div>
   1209/Unknown  380s 311ms/step - loss: 1.6296 - mean_absolute_error: 0.9971

<div class="k-default-codeblock">
```

```
</div>
   1210/Unknown  380s 312ms/step - loss: 1.6293 - mean_absolute_error: 0.9970

<div class="k-default-codeblock">
```

```
</div>
   1211/Unknown  381s 312ms/step - loss: 1.6290 - mean_absolute_error: 0.9969

<div class="k-default-codeblock">
```

```
</div>
   1212/Unknown  381s 312ms/step - loss: 1.6287 - mean_absolute_error: 0.9968

<div class="k-default-codeblock">
```

```
</div>
   1213/Unknown  381s 312ms/step - loss: 1.6284 - mean_absolute_error: 0.9967

<div class="k-default-codeblock">
```

```
</div>
   1214/Unknown  382s 312ms/step - loss: 1.6281 - mean_absolute_error: 0.9966

<div class="k-default-codeblock">
```

```
</div>
   1215/Unknown  382s 312ms/step - loss: 1.6278 - mean_absolute_error: 0.9966

<div class="k-default-codeblock">
```

```
</div>
   1216/Unknown  382s 312ms/step - loss: 1.6275 - mean_absolute_error: 0.9965

<div class="k-default-codeblock">
```

```
</div>
   1217/Unknown  383s 312ms/step - loss: 1.6272 - mean_absolute_error: 0.9964

<div class="k-default-codeblock">
```

```
</div>
   1218/Unknown  383s 312ms/step - loss: 1.6269 - mean_absolute_error: 0.9963

<div class="k-default-codeblock">
```

```
</div>
   1219/Unknown  384s 312ms/step - loss: 1.6266 - mean_absolute_error: 0.9962

<div class="k-default-codeblock">
```

```
</div>
   1220/Unknown  384s 312ms/step - loss: 1.6263 - mean_absolute_error: 0.9961

<div class="k-default-codeblock">
```

```
</div>
   1221/Unknown  384s 312ms/step - loss: 1.6260 - mean_absolute_error: 0.9960

<div class="k-default-codeblock">
```

```
</div>
   1222/Unknown  385s 312ms/step - loss: 1.6257 - mean_absolute_error: 0.9960

<div class="k-default-codeblock">
```

```
</div>
   1223/Unknown  385s 312ms/step - loss: 1.6254 - mean_absolute_error: 0.9959

<div class="k-default-codeblock">
```

```
</div>
   1224/Unknown  385s 312ms/step - loss: 1.6251 - mean_absolute_error: 0.9958

<div class="k-default-codeblock">
```

```
</div>
   1225/Unknown  386s 312ms/step - loss: 1.6248 - mean_absolute_error: 0.9957

<div class="k-default-codeblock">
```

```
</div>
   1226/Unknown  386s 312ms/step - loss: 1.6245 - mean_absolute_error: 0.9956

<div class="k-default-codeblock">
```

```
</div>
   1227/Unknown  386s 312ms/step - loss: 1.6242 - mean_absolute_error: 0.9955

<div class="k-default-codeblock">
```

```
</div>
   1228/Unknown  387s 312ms/step - loss: 1.6239 - mean_absolute_error: 0.9955

<div class="k-default-codeblock">
```

```
</div>
   1229/Unknown  387s 312ms/step - loss: 1.6236 - mean_absolute_error: 0.9954

<div class="k-default-codeblock">
```

```
</div>
   1230/Unknown  388s 312ms/step - loss: 1.6233 - mean_absolute_error: 0.9953

<div class="k-default-codeblock">
```

```
</div>
   1231/Unknown  388s 312ms/step - loss: 1.6230 - mean_absolute_error: 0.9952

<div class="k-default-codeblock">
```

```
</div>
   1232/Unknown  388s 312ms/step - loss: 1.6227 - mean_absolute_error: 0.9951

<div class="k-default-codeblock">
```

```
</div>
   1233/Unknown  389s 313ms/step - loss: 1.6224 - mean_absolute_error: 0.9950

<div class="k-default-codeblock">
```

```
</div>
   1234/Unknown  389s 313ms/step - loss: 1.6222 - mean_absolute_error: 0.9950

<div class="k-default-codeblock">
```

```
</div>
   1235/Unknown  389s 313ms/step - loss: 1.6219 - mean_absolute_error: 0.9949

<div class="k-default-codeblock">
```

```
</div>
   1236/Unknown  390s 313ms/step - loss: 1.6216 - mean_absolute_error: 0.9948

<div class="k-default-codeblock">
```

```
</div>
   1237/Unknown  390s 313ms/step - loss: 1.6213 - mean_absolute_error: 0.9947

<div class="k-default-codeblock">
```

```
</div>
   1238/Unknown  390s 313ms/step - loss: 1.6210 - mean_absolute_error: 0.9946

<div class="k-default-codeblock">
```

```
</div>
   1239/Unknown  391s 313ms/step - loss: 1.6207 - mean_absolute_error: 0.9945

<div class="k-default-codeblock">
```

```
</div>
   1240/Unknown  391s 313ms/step - loss: 1.6204 - mean_absolute_error: 0.9945

<div class="k-default-codeblock">
```

```
</div>
   1241/Unknown  391s 313ms/step - loss: 1.6201 - mean_absolute_error: 0.9944

<div class="k-default-codeblock">
```

```
</div>
   1242/Unknown  392s 313ms/step - loss: 1.6198 - mean_absolute_error: 0.9943

<div class="k-default-codeblock">
```

```
</div>
   1243/Unknown  392s 313ms/step - loss: 1.6195 - mean_absolute_error: 0.9942

<div class="k-default-codeblock">
```

```
</div>
   1244/Unknown  392s 313ms/step - loss: 1.6193 - mean_absolute_error: 0.9941

<div class="k-default-codeblock">
```

```
</div>
   1245/Unknown  393s 313ms/step - loss: 1.6190 - mean_absolute_error: 0.9940

<div class="k-default-codeblock">
```

```
</div>
   1246/Unknown  393s 313ms/step - loss: 1.6187 - mean_absolute_error: 0.9940

<div class="k-default-codeblock">
```

```
</div>
   1247/Unknown  393s 313ms/step - loss: 1.6184 - mean_absolute_error: 0.9939

<div class="k-default-codeblock">
```

```
</div>
   1248/Unknown  394s 313ms/step - loss: 1.6181 - mean_absolute_error: 0.9938

<div class="k-default-codeblock">
```

```
</div>
   1249/Unknown  394s 313ms/step - loss: 1.6178 - mean_absolute_error: 0.9937

<div class="k-default-codeblock">
```

```
</div>
   1250/Unknown  394s 313ms/step - loss: 1.6175 - mean_absolute_error: 0.9936

<div class="k-default-codeblock">
```

```
</div>
   1251/Unknown  395s 313ms/step - loss: 1.6172 - mean_absolute_error: 0.9936

<div class="k-default-codeblock">
```

```
</div>
   1252/Unknown  395s 313ms/step - loss: 1.6170 - mean_absolute_error: 0.9935

<div class="k-default-codeblock">
```

```
</div>
   1253/Unknown  395s 313ms/step - loss: 1.6167 - mean_absolute_error: 0.9934

<div class="k-default-codeblock">
```

```
</div>
   1254/Unknown  396s 313ms/step - loss: 1.6164 - mean_absolute_error: 0.9933

<div class="k-default-codeblock">
```

```
</div>
   1255/Unknown  396s 313ms/step - loss: 1.6161 - mean_absolute_error: 0.9932

<div class="k-default-codeblock">
```

```
</div>
   1256/Unknown  397s 313ms/step - loss: 1.6158 - mean_absolute_error: 0.9932

<div class="k-default-codeblock">
```

```
</div>
   1257/Unknown  397s 313ms/step - loss: 1.6155 - mean_absolute_error: 0.9931

<div class="k-default-codeblock">
```

```
</div>
   1258/Unknown  397s 313ms/step - loss: 1.6153 - mean_absolute_error: 0.9930

<div class="k-default-codeblock">
```

```
</div>
   1259/Unknown  398s 313ms/step - loss: 1.6150 - mean_absolute_error: 0.9929

<div class="k-default-codeblock">
```

```
</div>
   1260/Unknown  398s 313ms/step - loss: 1.6147 - mean_absolute_error: 0.9928

<div class="k-default-codeblock">
```

```
</div>
   1261/Unknown  399s 313ms/step - loss: 1.6144 - mean_absolute_error: 0.9927

<div class="k-default-codeblock">
```

```
</div>
   1262/Unknown  399s 314ms/step - loss: 1.6141 - mean_absolute_error: 0.9927

<div class="k-default-codeblock">
```

```
</div>
   1263/Unknown  399s 314ms/step - loss: 1.6138 - mean_absolute_error: 0.9926

<div class="k-default-codeblock">
```

```
</div>
   1264/Unknown  400s 314ms/step - loss: 1.6136 - mean_absolute_error: 0.9925

<div class="k-default-codeblock">
```

```
</div>
   1265/Unknown  400s 314ms/step - loss: 1.6133 - mean_absolute_error: 0.9924

<div class="k-default-codeblock">
```

```
</div>
   1266/Unknown  400s 314ms/step - loss: 1.6130 - mean_absolute_error: 0.9923

<div class="k-default-codeblock">
```

```
</div>
   1267/Unknown  401s 314ms/step - loss: 1.6127 - mean_absolute_error: 0.9923

<div class="k-default-codeblock">
```

```
</div>
   1268/Unknown  401s 314ms/step - loss: 1.6124 - mean_absolute_error: 0.9922

<div class="k-default-codeblock">
```

```
</div>
   1269/Unknown  401s 314ms/step - loss: 1.6122 - mean_absolute_error: 0.9921

<div class="k-default-codeblock">
```

```
</div>
   1270/Unknown  402s 314ms/step - loss: 1.6119 - mean_absolute_error: 0.9920

<div class="k-default-codeblock">
```

```
</div>
   1271/Unknown  402s 314ms/step - loss: 1.6116 - mean_absolute_error: 0.9919

<div class="k-default-codeblock">
```

```
</div>
   1272/Unknown  403s 314ms/step - loss: 1.6113 - mean_absolute_error: 0.9919

<div class="k-default-codeblock">
```

```
</div>
   1273/Unknown  403s 314ms/step - loss: 1.6110 - mean_absolute_error: 0.9918

<div class="k-default-codeblock">
```

```
</div>
   1274/Unknown  403s 314ms/step - loss: 1.6108 - mean_absolute_error: 0.9917

<div class="k-default-codeblock">
```

```
</div>
   1275/Unknown  404s 314ms/step - loss: 1.6105 - mean_absolute_error: 0.9916

<div class="k-default-codeblock">
```

```
</div>
   1276/Unknown  404s 314ms/step - loss: 1.6102 - mean_absolute_error: 0.9916

<div class="k-default-codeblock">
```

```
</div>
   1277/Unknown  404s 314ms/step - loss: 1.6099 - mean_absolute_error: 0.9915

<div class="k-default-codeblock">
```

```
</div>
   1278/Unknown  405s 314ms/step - loss: 1.6097 - mean_absolute_error: 0.9914

<div class="k-default-codeblock">
```

```
</div>
   1279/Unknown  405s 314ms/step - loss: 1.6094 - mean_absolute_error: 0.9913

<div class="k-default-codeblock">
```

```
</div>
   1280/Unknown  406s 314ms/step - loss: 1.6091 - mean_absolute_error: 0.9912

<div class="k-default-codeblock">
```

```
</div>
   1281/Unknown  406s 314ms/step - loss: 1.6088 - mean_absolute_error: 0.9912

<div class="k-default-codeblock">
```

```
</div>
   1282/Unknown  406s 314ms/step - loss: 1.6086 - mean_absolute_error: 0.9911

<div class="k-default-codeblock">
```

```
</div>
   1283/Unknown  407s 314ms/step - loss: 1.6083 - mean_absolute_error: 0.9910

<div class="k-default-codeblock">
```

```
</div>
   1284/Unknown  407s 314ms/step - loss: 1.6080 - mean_absolute_error: 0.9909

<div class="k-default-codeblock">
```

```
</div>
   1285/Unknown  407s 314ms/step - loss: 1.6077 - mean_absolute_error: 0.9908

<div class="k-default-codeblock">
```

```
</div>
   1286/Unknown  408s 314ms/step - loss: 1.6075 - mean_absolute_error: 0.9908

<div class="k-default-codeblock">
```

```
</div>
   1287/Unknown  408s 315ms/step - loss: 1.6072 - mean_absolute_error: 0.9907

<div class="k-default-codeblock">
```

```
</div>
   1288/Unknown  409s 315ms/step - loss: 1.6069 - mean_absolute_error: 0.9906

<div class="k-default-codeblock">
```

```
</div>
   1289/Unknown  409s 315ms/step - loss: 1.6066 - mean_absolute_error: 0.9905

<div class="k-default-codeblock">
```

```
</div>
   1290/Unknown  409s 315ms/step - loss: 1.6064 - mean_absolute_error: 0.9905

<div class="k-default-codeblock">
```

```
</div>
   1291/Unknown  410s 315ms/step - loss: 1.6061 - mean_absolute_error: 0.9904

<div class="k-default-codeblock">
```

```
</div>
   1292/Unknown  410s 315ms/step - loss: 1.6058 - mean_absolute_error: 0.9903

<div class="k-default-codeblock">
```

```
</div>
   1293/Unknown  410s 315ms/step - loss: 1.6056 - mean_absolute_error: 0.9902

<div class="k-default-codeblock">
```

```
</div>
   1294/Unknown  411s 315ms/step - loss: 1.6053 - mean_absolute_error: 0.9901

<div class="k-default-codeblock">
```

```
</div>
   1295/Unknown  411s 315ms/step - loss: 1.6050 - mean_absolute_error: 0.9901

<div class="k-default-codeblock">
```

```
</div>
   1296/Unknown  412s 315ms/step - loss: 1.6047 - mean_absolute_error: 0.9900

<div class="k-default-codeblock">
```

```
</div>
   1297/Unknown  412s 315ms/step - loss: 1.6045 - mean_absolute_error: 0.9899

<div class="k-default-codeblock">
```

```
</div>
   1298/Unknown  412s 315ms/step - loss: 1.6042 - mean_absolute_error: 0.9898

<div class="k-default-codeblock">
```

```
</div>
   1299/Unknown  413s 315ms/step - loss: 1.6039 - mean_absolute_error: 0.9898

<div class="k-default-codeblock">
```

```
</div>
   1300/Unknown  413s 315ms/step - loss: 1.6037 - mean_absolute_error: 0.9897

<div class="k-default-codeblock">
```

```
</div>
   1301/Unknown  413s 315ms/step - loss: 1.6034 - mean_absolute_error: 0.9896

<div class="k-default-codeblock">
```

```
</div>
   1302/Unknown  414s 315ms/step - loss: 1.6031 - mean_absolute_error: 0.9895

<div class="k-default-codeblock">
```

```
</div>
   1303/Unknown  414s 315ms/step - loss: 1.6029 - mean_absolute_error: 0.9895

<div class="k-default-codeblock">
```

```
</div>
   1304/Unknown  414s 315ms/step - loss: 1.6026 - mean_absolute_error: 0.9894

<div class="k-default-codeblock">
```

```
</div>
   1305/Unknown  415s 315ms/step - loss: 1.6023 - mean_absolute_error: 0.9893

<div class="k-default-codeblock">
```

```
</div>
   1306/Unknown  415s 315ms/step - loss: 1.6021 - mean_absolute_error: 0.9892

<div class="k-default-codeblock">
```

```
</div>
   1307/Unknown  416s 315ms/step - loss: 1.6018 - mean_absolute_error: 0.9891

<div class="k-default-codeblock">
```

```
</div>
   1308/Unknown  416s 315ms/step - loss: 1.6015 - mean_absolute_error: 0.9891

<div class="k-default-codeblock">
```

```
</div>
   1309/Unknown  416s 315ms/step - loss: 1.6013 - mean_absolute_error: 0.9890

<div class="k-default-codeblock">
```

```
</div>
   1310/Unknown  417s 315ms/step - loss: 1.6010 - mean_absolute_error: 0.9889

<div class="k-default-codeblock">
```

```
</div>
   1311/Unknown  417s 316ms/step - loss: 1.6007 - mean_absolute_error: 0.9888

<div class="k-default-codeblock">
```

```
</div>
   1312/Unknown  417s 316ms/step - loss: 1.6005 - mean_absolute_error: 0.9888

<div class="k-default-codeblock">
```

```
</div>
   1313/Unknown  418s 316ms/step - loss: 1.6002 - mean_absolute_error: 0.9887

<div class="k-default-codeblock">
```

```
</div>
   1314/Unknown  418s 316ms/step - loss: 1.5999 - mean_absolute_error: 0.9886

<div class="k-default-codeblock">
```

```
</div>
   1315/Unknown  418s 316ms/step - loss: 1.5997 - mean_absolute_error: 0.9885

<div class="k-default-codeblock">
```

```
</div>
   1316/Unknown  419s 316ms/step - loss: 1.5994 - mean_absolute_error: 0.9885

<div class="k-default-codeblock">
```

```
</div>
   1317/Unknown  419s 316ms/step - loss: 1.5992 - mean_absolute_error: 0.9884

<div class="k-default-codeblock">
```

```
</div>
   1318/Unknown  420s 316ms/step - loss: 1.5989 - mean_absolute_error: 0.9883

<div class="k-default-codeblock">
```

```
</div>
   1319/Unknown  420s 316ms/step - loss: 1.5986 - mean_absolute_error: 0.9882

<div class="k-default-codeblock">
```

```
</div>
   1320/Unknown  421s 316ms/step - loss: 1.5984 - mean_absolute_error: 0.9882

<div class="k-default-codeblock">
```

```
</div>
   1321/Unknown  421s 316ms/step - loss: 1.5981 - mean_absolute_error: 0.9881

<div class="k-default-codeblock">
```

```
</div>
   1322/Unknown  421s 316ms/step - loss: 1.5978 - mean_absolute_error: 0.9880

<div class="k-default-codeblock">
```

```
</div>
   1323/Unknown  422s 316ms/step - loss: 1.5976 - mean_absolute_error: 0.9879

<div class="k-default-codeblock">
```

```
</div>
   1324/Unknown  422s 316ms/step - loss: 1.5973 - mean_absolute_error: 0.9879

<div class="k-default-codeblock">
```

```
</div>
   1325/Unknown  422s 316ms/step - loss: 1.5971 - mean_absolute_error: 0.9878

<div class="k-default-codeblock">
```

```
</div>
   1326/Unknown  423s 316ms/step - loss: 1.5968 - mean_absolute_error: 0.9877

<div class="k-default-codeblock">
```

```
</div>
   1327/Unknown  423s 316ms/step - loss: 1.5965 - mean_absolute_error: 0.9876

<div class="k-default-codeblock">
```

```
</div>
   1328/Unknown  423s 316ms/step - loss: 1.5963 - mean_absolute_error: 0.9876

<div class="k-default-codeblock">
```

```
</div>
   1329/Unknown  424s 316ms/step - loss: 1.5960 - mean_absolute_error: 0.9875

<div class="k-default-codeblock">
```

```
</div>
   1330/Unknown  424s 316ms/step - loss: 1.5958 - mean_absolute_error: 0.9874

<div class="k-default-codeblock">
```

```
</div>
   1331/Unknown  424s 316ms/step - loss: 1.5955 - mean_absolute_error: 0.9873

<div class="k-default-codeblock">
```

```
</div>
   1332/Unknown  425s 316ms/step - loss: 1.5952 - mean_absolute_error: 0.9873

<div class="k-default-codeblock">
```

```
</div>
   1333/Unknown  425s 316ms/step - loss: 1.5950 - mean_absolute_error: 0.9872

<div class="k-default-codeblock">
```

```
</div>
   1334/Unknown  426s 317ms/step - loss: 1.5947 - mean_absolute_error: 0.9871

<div class="k-default-codeblock">
```

```
</div>
   1335/Unknown  426s 317ms/step - loss: 1.5945 - mean_absolute_error: 0.9871

<div class="k-default-codeblock">
```

```
</div>
   1336/Unknown  426s 317ms/step - loss: 1.5942 - mean_absolute_error: 0.9870

<div class="k-default-codeblock">
```

```
</div>
   1337/Unknown  427s 317ms/step - loss: 1.5940 - mean_absolute_error: 0.9869

<div class="k-default-codeblock">
```

```
</div>
   1338/Unknown  427s 317ms/step - loss: 1.5937 - mean_absolute_error: 0.9868

<div class="k-default-codeblock">
```

```
</div>
   1339/Unknown  428s 317ms/step - loss: 1.5934 - mean_absolute_error: 0.9868

<div class="k-default-codeblock">
```

```
</div>
   1340/Unknown  428s 317ms/step - loss: 1.5932 - mean_absolute_error: 0.9867

<div class="k-default-codeblock">
```

```
</div>
   1341/Unknown  428s 317ms/step - loss: 1.5929 - mean_absolute_error: 0.9866

<div class="k-default-codeblock">
```

```
</div>
   1342/Unknown  429s 317ms/step - loss: 1.5927 - mean_absolute_error: 0.9865

<div class="k-default-codeblock">
```

```
</div>
   1343/Unknown  429s 317ms/step - loss: 1.5924 - mean_absolute_error: 0.9865

<div class="k-default-codeblock">
```

```
</div>
   1344/Unknown  430s 317ms/step - loss: 1.5922 - mean_absolute_error: 0.9864

<div class="k-default-codeblock">
```

```
</div>
   1345/Unknown  430s 317ms/step - loss: 1.5919 - mean_absolute_error: 0.9863

<div class="k-default-codeblock">
```

```
</div>
   1346/Unknown  430s 317ms/step - loss: 1.5917 - mean_absolute_error: 0.9863

<div class="k-default-codeblock">
```

```
</div>
   1347/Unknown  431s 317ms/step - loss: 1.5914 - mean_absolute_error: 0.9862

<div class="k-default-codeblock">
```

```
</div>
   1348/Unknown  431s 317ms/step - loss: 1.5912 - mean_absolute_error: 0.9861

<div class="k-default-codeblock">
```

```
</div>
   1349/Unknown  431s 317ms/step - loss: 1.5909 - mean_absolute_error: 0.9860

<div class="k-default-codeblock">
```

```
</div>
   1350/Unknown  432s 317ms/step - loss: 1.5906 - mean_absolute_error: 0.9860

<div class="k-default-codeblock">
```

```
</div>
   1351/Unknown  432s 317ms/step - loss: 1.5904 - mean_absolute_error: 0.9859

<div class="k-default-codeblock">
```

```
</div>
   1352/Unknown  433s 318ms/step - loss: 1.5901 - mean_absolute_error: 0.9858

<div class="k-default-codeblock">
```

```
</div>
   1353/Unknown  433s 318ms/step - loss: 1.5899 - mean_absolute_error: 0.9857

<div class="k-default-codeblock">
```

```
</div>
   1354/Unknown  433s 318ms/step - loss: 1.5896 - mean_absolute_error: 0.9857

<div class="k-default-codeblock">
```

```
</div>
   1355/Unknown  434s 318ms/step - loss: 1.5894 - mean_absolute_error: 0.9856

<div class="k-default-codeblock">
```

```
</div>
   1356/Unknown  434s 318ms/step - loss: 1.5891 - mean_absolute_error: 0.9855

<div class="k-default-codeblock">
```

```
</div>
   1357/Unknown  434s 318ms/step - loss: 1.5889 - mean_absolute_error: 0.9855

<div class="k-default-codeblock">
```

```
</div>
   1358/Unknown  435s 318ms/step - loss: 1.5886 - mean_absolute_error: 0.9854

<div class="k-default-codeblock">
```

```
</div>
   1359/Unknown  435s 318ms/step - loss: 1.5884 - mean_absolute_error: 0.9853

<div class="k-default-codeblock">
```

```
</div>
   1360/Unknown  435s 318ms/step - loss: 1.5881 - mean_absolute_error: 0.9852

<div class="k-default-codeblock">
```

```
</div>
   1361/Unknown  436s 318ms/step - loss: 1.5879 - mean_absolute_error: 0.9852

<div class="k-default-codeblock">
```

```
</div>
   1362/Unknown  436s 318ms/step - loss: 1.5876 - mean_absolute_error: 0.9851

<div class="k-default-codeblock">
```

```
</div>
   1363/Unknown  437s 318ms/step - loss: 1.5874 - mean_absolute_error: 0.9850

<div class="k-default-codeblock">
```

```
</div>
   1364/Unknown  437s 318ms/step - loss: 1.5871 - mean_absolute_error: 0.9850

<div class="k-default-codeblock">
```

```
</div>
   1365/Unknown  437s 318ms/step - loss: 1.5869 - mean_absolute_error: 0.9849

<div class="k-default-codeblock">
```

```
</div>
   1366/Unknown  438s 318ms/step - loss: 1.5866 - mean_absolute_error: 0.9848

<div class="k-default-codeblock">
```

```
</div>
   1367/Unknown  438s 318ms/step - loss: 1.5864 - mean_absolute_error: 0.9847

<div class="k-default-codeblock">
```

```
</div>
   1368/Unknown  438s 318ms/step - loss: 1.5861 - mean_absolute_error: 0.9847

<div class="k-default-codeblock">
```

```
</div>
   1369/Unknown  439s 318ms/step - loss: 1.5859 - mean_absolute_error: 0.9846

<div class="k-default-codeblock">
```

```
</div>
   1370/Unknown  439s 318ms/step - loss: 1.5856 - mean_absolute_error: 0.9845

<div class="k-default-codeblock">
```

```
</div>
   1371/Unknown  440s 318ms/step - loss: 1.5854 - mean_absolute_error: 0.9845

<div class="k-default-codeblock">
```

```
</div>
   1372/Unknown  440s 318ms/step - loss: 1.5852 - mean_absolute_error: 0.9844

<div class="k-default-codeblock">
```

```
</div>
   1373/Unknown  440s 318ms/step - loss: 1.5849 - mean_absolute_error: 0.9843

<div class="k-default-codeblock">
```

```
</div>
   1374/Unknown  441s 318ms/step - loss: 1.5847 - mean_absolute_error: 0.9842

<div class="k-default-codeblock">
```

```
</div>
   1375/Unknown  441s 318ms/step - loss: 1.5844 - mean_absolute_error: 0.9842

<div class="k-default-codeblock">
```

```
</div>
   1376/Unknown  441s 318ms/step - loss: 1.5842 - mean_absolute_error: 0.9841

<div class="k-default-codeblock">
```

```
</div>
   1377/Unknown  442s 318ms/step - loss: 1.5839 - mean_absolute_error: 0.9840

<div class="k-default-codeblock">
```

```
</div>
   1378/Unknown  442s 318ms/step - loss: 1.5837 - mean_absolute_error: 0.9840

<div class="k-default-codeblock">
```

```
</div>
   1379/Unknown  442s 318ms/step - loss: 1.5834 - mean_absolute_error: 0.9839

<div class="k-default-codeblock">
```

```
</div>
   1380/Unknown  443s 318ms/step - loss: 1.5832 - mean_absolute_error: 0.9838

<div class="k-default-codeblock">
```

```
</div>
   1381/Unknown  443s 318ms/step - loss: 1.5829 - mean_absolute_error: 0.9838

<div class="k-default-codeblock">
```

```
</div>
   1382/Unknown  443s 318ms/step - loss: 1.5827 - mean_absolute_error: 0.9837

<div class="k-default-codeblock">
```

```
</div>
   1383/Unknown  444s 319ms/step - loss: 1.5825 - mean_absolute_error: 0.9836

<div class="k-default-codeblock">
```

```
</div>
   1384/Unknown  444s 319ms/step - loss: 1.5822 - mean_absolute_error: 0.9835

<div class="k-default-codeblock">
```

```
</div>
   1385/Unknown  445s 319ms/step - loss: 1.5820 - mean_absolute_error: 0.9835

<div class="k-default-codeblock">
```

```
</div>
   1386/Unknown  445s 319ms/step - loss: 1.5817 - mean_absolute_error: 0.9834

<div class="k-default-codeblock">
```

```
</div>
   1387/Unknown  445s 319ms/step - loss: 1.5815 - mean_absolute_error: 0.9833

<div class="k-default-codeblock">
```

```
</div>
   1388/Unknown  446s 319ms/step - loss: 1.5812 - mean_absolute_error: 0.9833

<div class="k-default-codeblock">
```

```
</div>
   1389/Unknown  446s 319ms/step - loss: 1.5810 - mean_absolute_error: 0.9832

<div class="k-default-codeblock">
```

```
</div>
   1390/Unknown  446s 319ms/step - loss: 1.5808 - mean_absolute_error: 0.9831

<div class="k-default-codeblock">
```

```
</div>
   1391/Unknown  447s 319ms/step - loss: 1.5805 - mean_absolute_error: 0.9831

<div class="k-default-codeblock">
```

```
</div>
   1392/Unknown  447s 319ms/step - loss: 1.5803 - mean_absolute_error: 0.9830

<div class="k-default-codeblock">
```

```
</div>
   1393/Unknown  447s 319ms/step - loss: 1.5800 - mean_absolute_error: 0.9829

<div class="k-default-codeblock">
```

```
</div>
   1394/Unknown  448s 319ms/step - loss: 1.5798 - mean_absolute_error: 0.9829

<div class="k-default-codeblock">
```

```
</div>
   1395/Unknown  448s 319ms/step - loss: 1.5796 - mean_absolute_error: 0.9828

<div class="k-default-codeblock">
```

```
</div>
   1396/Unknown  449s 319ms/step - loss: 1.5793 - mean_absolute_error: 0.9827

<div class="k-default-codeblock">
```

```
</div>
   1397/Unknown  449s 319ms/step - loss: 1.5791 - mean_absolute_error: 0.9826

<div class="k-default-codeblock">
```

```
</div>
   1398/Unknown  449s 319ms/step - loss: 1.5788 - mean_absolute_error: 0.9826

<div class="k-default-codeblock">
```

```
</div>
   1399/Unknown  450s 319ms/step - loss: 1.5786 - mean_absolute_error: 0.9825

<div class="k-default-codeblock">
```

```
</div>
   1400/Unknown  450s 319ms/step - loss: 1.5784 - mean_absolute_error: 0.9824

<div class="k-default-codeblock">
```

```
</div>
   1401/Unknown  451s 319ms/step - loss: 1.5781 - mean_absolute_error: 0.9824

<div class="k-default-codeblock">
```

```
</div>
   1402/Unknown  451s 319ms/step - loss: 1.5779 - mean_absolute_error: 0.9823

<div class="k-default-codeblock">
```

```
</div>
   1403/Unknown  451s 319ms/step - loss: 1.5776 - mean_absolute_error: 0.9822

<div class="k-default-codeblock">
```

```
</div>
   1404/Unknown  452s 319ms/step - loss: 1.5774 - mean_absolute_error: 0.9822

<div class="k-default-codeblock">
```

```
</div>
   1405/Unknown  452s 320ms/step - loss: 1.5772 - mean_absolute_error: 0.9821

<div class="k-default-codeblock">
```

```
</div>
   1406/Unknown  453s 320ms/step - loss: 1.5769 - mean_absolute_error: 0.9820

<div class="k-default-codeblock">
```

```
</div>
   1407/Unknown  453s 320ms/step - loss: 1.5767 - mean_absolute_error: 0.9820

<div class="k-default-codeblock">
```

```
</div>
   1408/Unknown  453s 320ms/step - loss: 1.5765 - mean_absolute_error: 0.9819

<div class="k-default-codeblock">
```

```
</div>
   1409/Unknown  454s 320ms/step - loss: 1.5762 - mean_absolute_error: 0.9818

<div class="k-default-codeblock">
```

```
</div>
   1410/Unknown  454s 320ms/step - loss: 1.5760 - mean_absolute_error: 0.9818

<div class="k-default-codeblock">
```

```
</div>
   1411/Unknown  454s 320ms/step - loss: 1.5758 - mean_absolute_error: 0.9817

<div class="k-default-codeblock">
```

```
</div>
   1412/Unknown  455s 320ms/step - loss: 1.5755 - mean_absolute_error: 0.9816

<div class="k-default-codeblock">
```

```
</div>
   1413/Unknown  455s 320ms/step - loss: 1.5753 - mean_absolute_error: 0.9816

<div class="k-default-codeblock">
```

```
</div>
   1414/Unknown  455s 320ms/step - loss: 1.5751 - mean_absolute_error: 0.9815

<div class="k-default-codeblock">
```

```
</div>
   1415/Unknown  456s 320ms/step - loss: 1.5748 - mean_absolute_error: 0.9814

<div class="k-default-codeblock">
```

```
</div>
   1416/Unknown  456s 320ms/step - loss: 1.5746 - mean_absolute_error: 0.9814

<div class="k-default-codeblock">
```

```
</div>
   1417/Unknown  456s 320ms/step - loss: 1.5743 - mean_absolute_error: 0.9813

<div class="k-default-codeblock">
```

```
</div>
   1418/Unknown  457s 320ms/step - loss: 1.5741 - mean_absolute_error: 0.9812

<div class="k-default-codeblock">
```

```
</div>
   1419/Unknown  457s 320ms/step - loss: 1.5739 - mean_absolute_error: 0.9812

<div class="k-default-codeblock">
```

```
</div>
   1420/Unknown  457s 320ms/step - loss: 1.5736 - mean_absolute_error: 0.9811

<div class="k-default-codeblock">
```

```
</div>
   1421/Unknown  458s 320ms/step - loss: 1.5734 - mean_absolute_error: 0.9810

<div class="k-default-codeblock">
```

```
</div>
   1422/Unknown  458s 320ms/step - loss: 1.5732 - mean_absolute_error: 0.9810

<div class="k-default-codeblock">
```

```
</div>
   1423/Unknown  458s 320ms/step - loss: 1.5729 - mean_absolute_error: 0.9809

<div class="k-default-codeblock">
```

```
</div>
   1424/Unknown  459s 320ms/step - loss: 1.5727 - mean_absolute_error: 0.9808

<div class="k-default-codeblock">
```

```
</div>
   1425/Unknown  459s 320ms/step - loss: 1.5725 - mean_absolute_error: 0.9808

<div class="k-default-codeblock">
```

```
</div>
   1426/Unknown  460s 320ms/step - loss: 1.5723 - mean_absolute_error: 0.9807

<div class="k-default-codeblock">
```

```
</div>
   1427/Unknown  460s 320ms/step - loss: 1.5720 - mean_absolute_error: 0.9806

<div class="k-default-codeblock">
```

```
</div>
   1428/Unknown  460s 320ms/step - loss: 1.5718 - mean_absolute_error: 0.9806

<div class="k-default-codeblock">
```

```
</div>
   1429/Unknown  461s 320ms/step - loss: 1.5716 - mean_absolute_error: 0.9805

<div class="k-default-codeblock">
```

```
</div>
   1430/Unknown  461s 320ms/step - loss: 1.5713 - mean_absolute_error: 0.9804

<div class="k-default-codeblock">
```

```
</div>
   1431/Unknown  461s 320ms/step - loss: 1.5711 - mean_absolute_error: 0.9804

<div class="k-default-codeblock">
```

```
</div>
   1432/Unknown  462s 320ms/step - loss: 1.5709 - mean_absolute_error: 0.9803

<div class="k-default-codeblock">
```

```
</div>
   1433/Unknown  462s 320ms/step - loss: 1.5706 - mean_absolute_error: 0.9802

<div class="k-default-codeblock">
```

```
</div>
   1434/Unknown  463s 320ms/step - loss: 1.5704 - mean_absolute_error: 0.9802

<div class="k-default-codeblock">
```

```
</div>
   1435/Unknown  463s 320ms/step - loss: 1.5702 - mean_absolute_error: 0.9801

<div class="k-default-codeblock">
```

```
</div>
   1436/Unknown  463s 320ms/step - loss: 1.5699 - mean_absolute_error: 0.9800

<div class="k-default-codeblock">
```

```
</div>
   1437/Unknown  464s 320ms/step - loss: 1.5697 - mean_absolute_error: 0.9800

<div class="k-default-codeblock">
```

```
</div>
   1438/Unknown  464s 320ms/step - loss: 1.5695 - mean_absolute_error: 0.9799

<div class="k-default-codeblock">
```

```
</div>
   1439/Unknown  464s 320ms/step - loss: 1.5693 - mean_absolute_error: 0.9798

<div class="k-default-codeblock">
```

```
</div>
   1440/Unknown  465s 320ms/step - loss: 1.5690 - mean_absolute_error: 0.9798

<div class="k-default-codeblock">
```

```
</div>
   1441/Unknown  465s 321ms/step - loss: 1.5688 - mean_absolute_error: 0.9797

<div class="k-default-codeblock">
```

```
</div>
   1442/Unknown  466s 321ms/step - loss: 1.5686 - mean_absolute_error: 0.9796

<div class="k-default-codeblock">
```

```
</div>
   1443/Unknown  466s 321ms/step - loss: 1.5684 - mean_absolute_error: 0.9796

<div class="k-default-codeblock">
```

```
</div>
   1444/Unknown  466s 321ms/step - loss: 1.5681 - mean_absolute_error: 0.9795

<div class="k-default-codeblock">
```

```
</div>
   1445/Unknown  467s 321ms/step - loss: 1.5679 - mean_absolute_error: 0.9794

<div class="k-default-codeblock">
```

```
</div>
   1446/Unknown  467s 321ms/step - loss: 1.5677 - mean_absolute_error: 0.9794

<div class="k-default-codeblock">
```

```
</div>
   1447/Unknown  467s 321ms/step - loss: 1.5674 - mean_absolute_error: 0.9793

<div class="k-default-codeblock">
```

```
</div>
   1448/Unknown  468s 321ms/step - loss: 1.5672 - mean_absolute_error: 0.9792

<div class="k-default-codeblock">
```

```
</div>
   1449/Unknown  468s 321ms/step - loss: 1.5670 - mean_absolute_error: 0.9792

<div class="k-default-codeblock">
```

```
</div>
   1450/Unknown  469s 321ms/step - loss: 1.5668 - mean_absolute_error: 0.9791

<div class="k-default-codeblock">
```

```
</div>
   1451/Unknown  469s 321ms/step - loss: 1.5665 - mean_absolute_error: 0.9790

<div class="k-default-codeblock">
```

```
</div>
   1452/Unknown  469s 321ms/step - loss: 1.5663 - mean_absolute_error: 0.9790

<div class="k-default-codeblock">
```

```
</div>
   1453/Unknown  470s 321ms/step - loss: 1.5661 - mean_absolute_error: 0.9789

<div class="k-default-codeblock">
```

```
</div>
   1454/Unknown  470s 321ms/step - loss: 1.5659 - mean_absolute_error: 0.9788

<div class="k-default-codeblock">
```

```
</div>
   1455/Unknown  470s 321ms/step - loss: 1.5656 - mean_absolute_error: 0.9788

<div class="k-default-codeblock">
```

```
</div>
   1456/Unknown  471s 321ms/step - loss: 1.5654 - mean_absolute_error: 0.9787

<div class="k-default-codeblock">
```

```
</div>
   1457/Unknown  471s 321ms/step - loss: 1.5652 - mean_absolute_error: 0.9787

<div class="k-default-codeblock">
```

```
</div>
   1458/Unknown  471s 321ms/step - loss: 1.5650 - mean_absolute_error: 0.9786

<div class="k-default-codeblock">
```

```
</div>
   1459/Unknown  472s 321ms/step - loss: 1.5647 - mean_absolute_error: 0.9785

<div class="k-default-codeblock">
```

```
</div>
   1460/Unknown  472s 321ms/step - loss: 1.5645 - mean_absolute_error: 0.9785

<div class="k-default-codeblock">
```

```
</div>
   1461/Unknown  472s 321ms/step - loss: 1.5643 - mean_absolute_error: 0.9784

<div class="k-default-codeblock">
```

```
</div>
   1462/Unknown  473s 321ms/step - loss: 1.5641 - mean_absolute_error: 0.9783

<div class="k-default-codeblock">
```

```
</div>
   1463/Unknown  473s 321ms/step - loss: 1.5639 - mean_absolute_error: 0.9783

<div class="k-default-codeblock">
```

```
</div>
   1464/Unknown  473s 321ms/step - loss: 1.5636 - mean_absolute_error: 0.9782

<div class="k-default-codeblock">
```

```
</div>
   1465/Unknown  474s 321ms/step - loss: 1.5634 - mean_absolute_error: 0.9781

<div class="k-default-codeblock">
```

```
</div>
   1466/Unknown  474s 321ms/step - loss: 1.5632 - mean_absolute_error: 0.9781

<div class="k-default-codeblock">
```

```
</div>
   1467/Unknown  475s 321ms/step - loss: 1.5630 - mean_absolute_error: 0.9780

<div class="k-default-codeblock">
```

```
</div>
   1468/Unknown  475s 322ms/step - loss: 1.5627 - mean_absolute_error: 0.9779

<div class="k-default-codeblock">
```

```
</div>
   1469/Unknown  476s 322ms/step - loss: 1.5625 - mean_absolute_error: 0.9779

<div class="k-default-codeblock">
```

```
</div>
   1470/Unknown  476s 322ms/step - loss: 1.5623 - mean_absolute_error: 0.9778

<div class="k-default-codeblock">
```

```
</div>
   1471/Unknown  476s 322ms/step - loss: 1.5621 - mean_absolute_error: 0.9778

<div class="k-default-codeblock">
```

```
</div>
   1472/Unknown  477s 322ms/step - loss: 1.5619 - mean_absolute_error: 0.9777

<div class="k-default-codeblock">
```

```
</div>
   1473/Unknown  477s 322ms/step - loss: 1.5616 - mean_absolute_error: 0.9776

<div class="k-default-codeblock">
```

```
</div>
   1474/Unknown  478s 322ms/step - loss: 1.5614 - mean_absolute_error: 0.9776

<div class="k-default-codeblock">
```

```
</div>
   1475/Unknown  478s 322ms/step - loss: 1.5612 - mean_absolute_error: 0.9775

<div class="k-default-codeblock">
```

```
</div>
   1476/Unknown  478s 322ms/step - loss: 1.5610 - mean_absolute_error: 0.9774

<div class="k-default-codeblock">
```

```
</div>
   1477/Unknown  479s 322ms/step - loss: 1.5608 - mean_absolute_error: 0.9774

<div class="k-default-codeblock">
```

```
</div>
   1478/Unknown  479s 322ms/step - loss: 1.5605 - mean_absolute_error: 0.9773

<div class="k-default-codeblock">
```

```
</div>
   1479/Unknown  479s 322ms/step - loss: 1.5603 - mean_absolute_error: 0.9773

<div class="k-default-codeblock">
```

```
</div>
   1480/Unknown  480s 322ms/step - loss: 1.5601 - mean_absolute_error: 0.9772

<div class="k-default-codeblock">
```

```
</div>
   1481/Unknown  480s 322ms/step - loss: 1.5599 - mean_absolute_error: 0.9771

<div class="k-default-codeblock">
```

```
</div>
   1482/Unknown  480s 322ms/step - loss: 1.5597 - mean_absolute_error: 0.9771

<div class="k-default-codeblock">
```

```
</div>
   1483/Unknown  481s 322ms/step - loss: 1.5595 - mean_absolute_error: 0.9770

<div class="k-default-codeblock">
```

```
</div>
   1484/Unknown  481s 322ms/step - loss: 1.5592 - mean_absolute_error: 0.9769

<div class="k-default-codeblock">
```

```
</div>
   1485/Unknown  482s 322ms/step - loss: 1.5590 - mean_absolute_error: 0.9769

<div class="k-default-codeblock">
```

```
</div>
   1486/Unknown  482s 322ms/step - loss: 1.5588 - mean_absolute_error: 0.9768

<div class="k-default-codeblock">
```

```
</div>
   1487/Unknown  482s 322ms/step - loss: 1.5586 - mean_absolute_error: 0.9767

<div class="k-default-codeblock">
```

```
</div>
   1488/Unknown  483s 322ms/step - loss: 1.5584 - mean_absolute_error: 0.9767

<div class="k-default-codeblock">
```

```
</div>
   1489/Unknown  483s 322ms/step - loss: 1.5582 - mean_absolute_error: 0.9766

<div class="k-default-codeblock">
```

```
</div>
   1490/Unknown  484s 322ms/step - loss: 1.5579 - mean_absolute_error: 0.9766

<div class="k-default-codeblock">
```

```
</div>
   1491/Unknown  484s 322ms/step - loss: 1.5577 - mean_absolute_error: 0.9765

<div class="k-default-codeblock">
```

```
</div>
   1492/Unknown  484s 322ms/step - loss: 1.5575 - mean_absolute_error: 0.9764

<div class="k-default-codeblock">
```

```
</div>
   1493/Unknown  485s 322ms/step - loss: 1.5573 - mean_absolute_error: 0.9764

<div class="k-default-codeblock">
```

```
</div>
   1494/Unknown  485s 322ms/step - loss: 1.5571 - mean_absolute_error: 0.9763

<div class="k-default-codeblock">
```

```
</div>
   1495/Unknown  485s 322ms/step - loss: 1.5569 - mean_absolute_error: 0.9763

<div class="k-default-codeblock">
```

```
</div>
   1496/Unknown  486s 322ms/step - loss: 1.5566 - mean_absolute_error: 0.9762

<div class="k-default-codeblock">
```

```
</div>
   1497/Unknown  486s 322ms/step - loss: 1.5564 - mean_absolute_error: 0.9761

<div class="k-default-codeblock">
```

```
</div>
   1498/Unknown  486s 322ms/step - loss: 1.5562 - mean_absolute_error: 0.9761

<div class="k-default-codeblock">
```

```
</div>
   1499/Unknown  487s 322ms/step - loss: 1.5560 - mean_absolute_error: 0.9760

<div class="k-default-codeblock">
```

```
</div>
   1500/Unknown  487s 323ms/step - loss: 1.5558 - mean_absolute_error: 0.9759

<div class="k-default-codeblock">
```

```
</div>
   1501/Unknown  487s 323ms/step - loss: 1.5556 - mean_absolute_error: 0.9759

<div class="k-default-codeblock">
```

```
</div>
   1502/Unknown  488s 323ms/step - loss: 1.5554 - mean_absolute_error: 0.9758

<div class="k-default-codeblock">
```

```
</div>
   1503/Unknown  488s 323ms/step - loss: 1.5551 - mean_absolute_error: 0.9758

<div class="k-default-codeblock">
```

```
</div>
   1504/Unknown  489s 323ms/step - loss: 1.5549 - mean_absolute_error: 0.9757

<div class="k-default-codeblock">
```

```
</div>
   1505/Unknown  489s 323ms/step - loss: 1.5547 - mean_absolute_error: 0.9756

<div class="k-default-codeblock">
```

```
</div>
   1506/Unknown  489s 323ms/step - loss: 1.5545 - mean_absolute_error: 0.9756

<div class="k-default-codeblock">
```

```
</div>
   1507/Unknown  490s 323ms/step - loss: 1.5543 - mean_absolute_error: 0.9755

<div class="k-default-codeblock">
```

```
</div>
   1508/Unknown  490s 323ms/step - loss: 1.5541 - mean_absolute_error: 0.9754

<div class="k-default-codeblock">
```

```
</div>
   1509/Unknown  491s 323ms/step - loss: 1.5539 - mean_absolute_error: 0.9754

<div class="k-default-codeblock">
```

```
</div>
   1510/Unknown  491s 323ms/step - loss: 1.5537 - mean_absolute_error: 0.9753

<div class="k-default-codeblock">
```

```
</div>
   1511/Unknown  491s 323ms/step - loss: 1.5534 - mean_absolute_error: 0.9753

<div class="k-default-codeblock">
```

```
</div>
   1512/Unknown  492s 323ms/step - loss: 1.5532 - mean_absolute_error: 0.9752

<div class="k-default-codeblock">
```

```
</div>
   1513/Unknown  492s 323ms/step - loss: 1.5530 - mean_absolute_error: 0.9751

<div class="k-default-codeblock">
```

```
</div>
   1514/Unknown  492s 323ms/step - loss: 1.5528 - mean_absolute_error: 0.9751

<div class="k-default-codeblock">
```

```
</div>
   1515/Unknown  493s 323ms/step - loss: 1.5526 - mean_absolute_error: 0.9750

<div class="k-default-codeblock">
```

```
</div>
   1516/Unknown  493s 323ms/step - loss: 1.5524 - mean_absolute_error: 0.9750

<div class="k-default-codeblock">
```

```
</div>
   1517/Unknown  494s 323ms/step - loss: 1.5522 - mean_absolute_error: 0.9749

<div class="k-default-codeblock">
```

```
</div>
   1518/Unknown  494s 323ms/step - loss: 1.5520 - mean_absolute_error: 0.9748

<div class="k-default-codeblock">
```

```
</div>
   1519/Unknown  494s 323ms/step - loss: 1.5518 - mean_absolute_error: 0.9748

<div class="k-default-codeblock">
```

```
</div>
   1520/Unknown  495s 323ms/step - loss: 1.5515 - mean_absolute_error: 0.9747

<div class="k-default-codeblock">
```

```
</div>
   1521/Unknown  495s 323ms/step - loss: 1.5513 - mean_absolute_error: 0.9747

<div class="k-default-codeblock">
```

```
</div>
   1522/Unknown  496s 323ms/step - loss: 1.5511 - mean_absolute_error: 0.9746

<div class="k-default-codeblock">
```

```
</div>
   1523/Unknown  496s 323ms/step - loss: 1.5509 - mean_absolute_error: 0.9745

<div class="k-default-codeblock">
```

```
</div>
   1524/Unknown  496s 324ms/step - loss: 1.5507 - mean_absolute_error: 0.9745

<div class="k-default-codeblock">
```

```
</div>
   1525/Unknown  497s 324ms/step - loss: 1.5505 - mean_absolute_error: 0.9744

<div class="k-default-codeblock">
```

```
</div>
   1526/Unknown  497s 324ms/step - loss: 1.5503 - mean_absolute_error: 0.9744

<div class="k-default-codeblock">
```

```
</div>
   1527/Unknown  497s 324ms/step - loss: 1.5501 - mean_absolute_error: 0.9743

<div class="k-default-codeblock">
```

```
</div>
   1528/Unknown  498s 324ms/step - loss: 1.5499 - mean_absolute_error: 0.9742

<div class="k-default-codeblock">
```

```
</div>
   1529/Unknown  498s 324ms/step - loss: 1.5497 - mean_absolute_error: 0.9742

<div class="k-default-codeblock">
```

```
</div>
   1530/Unknown  499s 324ms/step - loss: 1.5495 - mean_absolute_error: 0.9741

<div class="k-default-codeblock">
```

```
</div>
   1531/Unknown  499s 324ms/step - loss: 1.5493 - mean_absolute_error: 0.9741

<div class="k-default-codeblock">
```

```
</div>
   1532/Unknown  499s 324ms/step - loss: 1.5490 - mean_absolute_error: 0.9740

<div class="k-default-codeblock">
```

```
</div>
   1533/Unknown  500s 324ms/step - loss: 1.5488 - mean_absolute_error: 0.9739

<div class="k-default-codeblock">
```

```
</div>
   1534/Unknown  500s 324ms/step - loss: 1.5486 - mean_absolute_error: 0.9739

<div class="k-default-codeblock">
```

```
</div>
   1535/Unknown  500s 324ms/step - loss: 1.5484 - mean_absolute_error: 0.9738

<div class="k-default-codeblock">
```

```
</div>
   1536/Unknown  501s 324ms/step - loss: 1.5482 - mean_absolute_error: 0.9738

<div class="k-default-codeblock">
```

```
</div>
   1537/Unknown  501s 324ms/step - loss: 1.5480 - mean_absolute_error: 0.9737

<div class="k-default-codeblock">
```

```
</div>
   1538/Unknown  501s 324ms/step - loss: 1.5478 - mean_absolute_error: 0.9736

<div class="k-default-codeblock">
```

```
</div>
   1539/Unknown  502s 324ms/step - loss: 1.5476 - mean_absolute_error: 0.9736

<div class="k-default-codeblock">
```

```
</div>
   1540/Unknown  502s 324ms/step - loss: 1.5474 - mean_absolute_error: 0.9735

<div class="k-default-codeblock">
```

```
</div>
   1541/Unknown  503s 324ms/step - loss: 1.5472 - mean_absolute_error: 0.9735

<div class="k-default-codeblock">
```

```
</div>
   1542/Unknown  503s 324ms/step - loss: 1.5470 - mean_absolute_error: 0.9734

<div class="k-default-codeblock">
```

```
</div>
   1543/Unknown  503s 324ms/step - loss: 1.5468 - mean_absolute_error: 0.9733

<div class="k-default-codeblock">
```

```
</div>
   1544/Unknown  504s 324ms/step - loss: 1.5466 - mean_absolute_error: 0.9733

<div class="k-default-codeblock">
```

```
</div>
   1545/Unknown  504s 324ms/step - loss: 1.5464 - mean_absolute_error: 0.9732

<div class="k-default-codeblock">
```

```
</div>
   1546/Unknown  505s 324ms/step - loss: 1.5462 - mean_absolute_error: 0.9732

<div class="k-default-codeblock">
```

```
</div>
   1547/Unknown  505s 324ms/step - loss: 1.5460 - mean_absolute_error: 0.9731

<div class="k-default-codeblock">
```

```
</div>
   1548/Unknown  505s 324ms/step - loss: 1.5458 - mean_absolute_error: 0.9730

<div class="k-default-codeblock">
```

```
</div>
   1549/Unknown  506s 324ms/step - loss: 1.5456 - mean_absolute_error: 0.9730

<div class="k-default-codeblock">
```

```
</div>
   1550/Unknown  506s 324ms/step - loss: 1.5454 - mean_absolute_error: 0.9729

<div class="k-default-codeblock">
```

```
</div>
   1551/Unknown  506s 324ms/step - loss: 1.5452 - mean_absolute_error: 0.9729

<div class="k-default-codeblock">
```

```
</div>
   1552/Unknown  507s 324ms/step - loss: 1.5449 - mean_absolute_error: 0.9728

<div class="k-default-codeblock">
```

```
</div>
   1553/Unknown  507s 324ms/step - loss: 1.5447 - mean_absolute_error: 0.9727

<div class="k-default-codeblock">
```

```
</div>
   1554/Unknown  507s 324ms/step - loss: 1.5445 - mean_absolute_error: 0.9727

<div class="k-default-codeblock">
```

```
</div>
   1555/Unknown  508s 324ms/step - loss: 1.5443 - mean_absolute_error: 0.9726

<div class="k-default-codeblock">
```

```
</div>
   1556/Unknown  508s 324ms/step - loss: 1.5441 - mean_absolute_error: 0.9726

<div class="k-default-codeblock">
```

```
</div>
   1557/Unknown  508s 324ms/step - loss: 1.5439 - mean_absolute_error: 0.9725

<div class="k-default-codeblock">
```

```
</div>
   1558/Unknown  509s 324ms/step - loss: 1.5437 - mean_absolute_error: 0.9725

<div class="k-default-codeblock">
```

```
</div>
   1559/Unknown  509s 324ms/step - loss: 1.5435 - mean_absolute_error: 0.9724

<div class="k-default-codeblock">
```

```
</div>
   1560/Unknown  510s 324ms/step - loss: 1.5433 - mean_absolute_error: 0.9723

<div class="k-default-codeblock">
```

```
</div>
   1561/Unknown  510s 325ms/step - loss: 1.5431 - mean_absolute_error: 0.9723

<div class="k-default-codeblock">
```

```
</div>
   1562/Unknown  510s 325ms/step - loss: 1.5429 - mean_absolute_error: 0.9722

<div class="k-default-codeblock">
```

```
</div>
   1563/Unknown  511s 325ms/step - loss: 1.5427 - mean_absolute_error: 0.9722

<div class="k-default-codeblock">
```

```
</div>
   1564/Unknown  511s 325ms/step - loss: 1.5425 - mean_absolute_error: 0.9721

<div class="k-default-codeblock">
```

```
</div>
   1565/Unknown  511s 325ms/step - loss: 1.5423 - mean_absolute_error: 0.9720

<div class="k-default-codeblock">
```

```
</div>
   1566/Unknown  512s 325ms/step - loss: 1.5421 - mean_absolute_error: 0.9720

<div class="k-default-codeblock">
```

```
</div>
   1567/Unknown  512s 325ms/step - loss: 1.5419 - mean_absolute_error: 0.9719

<div class="k-default-codeblock">
```

```
</div>
   1568/Unknown  513s 325ms/step - loss: 1.5417 - mean_absolute_error: 0.9719

<div class="k-default-codeblock">
```

```
</div>
   1569/Unknown  513s 325ms/step - loss: 1.5415 - mean_absolute_error: 0.9718

<div class="k-default-codeblock">
```

```
</div>
   1570/Unknown  513s 325ms/step - loss: 1.5413 - mean_absolute_error: 0.9718

<div class="k-default-codeblock">
```

```
</div>
   1571/Unknown  514s 325ms/step - loss: 1.5411 - mean_absolute_error: 0.9717

<div class="k-default-codeblock">
```

```
</div>
   1572/Unknown  514s 325ms/step - loss: 1.5409 - mean_absolute_error: 0.9716

<div class="k-default-codeblock">
```

```
</div>
   1573/Unknown  514s 325ms/step - loss: 1.5407 - mean_absolute_error: 0.9716

<div class="k-default-codeblock">
```

```
</div>
   1574/Unknown  515s 325ms/step - loss: 1.5405 - mean_absolute_error: 0.9715

<div class="k-default-codeblock">
```

```
</div>
   1575/Unknown  515s 325ms/step - loss: 1.5403 - mean_absolute_error: 0.9715

<div class="k-default-codeblock">
```

```
</div>
   1576/Unknown  516s 325ms/step - loss: 1.5401 - mean_absolute_error: 0.9714

<div class="k-default-codeblock">
```

```
</div>
   1577/Unknown  516s 325ms/step - loss: 1.5399 - mean_absolute_error: 0.9713

<div class="k-default-codeblock">
```

```
</div>
   1578/Unknown  516s 325ms/step - loss: 1.5397 - mean_absolute_error: 0.9713

<div class="k-default-codeblock">
```

```
</div>
   1579/Unknown  517s 325ms/step - loss: 1.5395 - mean_absolute_error: 0.9712

<div class="k-default-codeblock">
```

```
</div>
   1580/Unknown  517s 325ms/step - loss: 1.5393 - mean_absolute_error: 0.9712

<div class="k-default-codeblock">
```

```
</div>
   1581/Unknown  517s 325ms/step - loss: 1.5391 - mean_absolute_error: 0.9711

<div class="k-default-codeblock">
```

```
</div>
   1582/Unknown  518s 325ms/step - loss: 1.5389 - mean_absolute_error: 0.9711

<div class="k-default-codeblock">
```

```
</div>
   1583/Unknown  518s 325ms/step - loss: 1.5387 - mean_absolute_error: 0.9710

<div class="k-default-codeblock">
```

```
</div>
   1584/Unknown  519s 325ms/step - loss: 1.5385 - mean_absolute_error: 0.9709

<div class="k-default-codeblock">
```

```
</div>
   1585/Unknown  519s 325ms/step - loss: 1.5383 - mean_absolute_error: 0.9709

<div class="k-default-codeblock">
```

```
</div>
   1586/Unknown  519s 325ms/step - loss: 1.5381 - mean_absolute_error: 0.9708

<div class="k-default-codeblock">
```

```
</div>
   1587/Unknown  520s 325ms/step - loss: 1.5380 - mean_absolute_error: 0.9708

<div class="k-default-codeblock">
```

```
</div>
   1588/Unknown  520s 325ms/step - loss: 1.5378 - mean_absolute_error: 0.9707

<div class="k-default-codeblock">
```

```
</div>
   1589/Unknown  520s 325ms/step - loss: 1.5376 - mean_absolute_error: 0.9707

<div class="k-default-codeblock">
```

```
</div>
   1590/Unknown  521s 325ms/step - loss: 1.5374 - mean_absolute_error: 0.9706

<div class="k-default-codeblock">
```

```
</div>
   1591/Unknown  521s 325ms/step - loss: 1.5372 - mean_absolute_error: 0.9705

<div class="k-default-codeblock">
```

```
</div>
   1592/Unknown  521s 325ms/step - loss: 1.5370 - mean_absolute_error: 0.9705

<div class="k-default-codeblock">
```

```
</div>
   1593/Unknown  522s 325ms/step - loss: 1.5368 - mean_absolute_error: 0.9704

<div class="k-default-codeblock">
```

```
</div>
   1594/Unknown  522s 325ms/step - loss: 1.5366 - mean_absolute_error: 0.9704

<div class="k-default-codeblock">
```

```
</div>
   1595/Unknown  522s 325ms/step - loss: 1.5364 - mean_absolute_error: 0.9703

<div class="k-default-codeblock">
```

```
</div>
   1596/Unknown  523s 325ms/step - loss: 1.5362 - mean_absolute_error: 0.9703

<div class="k-default-codeblock">
```

```
</div>
   1597/Unknown  523s 326ms/step - loss: 1.5360 - mean_absolute_error: 0.9702

<div class="k-default-codeblock">
```

```
</div>
   1598/Unknown  524s 326ms/step - loss: 1.5358 - mean_absolute_error: 0.9701

<div class="k-default-codeblock">
```

```
</div>
   1599/Unknown  524s 326ms/step - loss: 1.5356 - mean_absolute_error: 0.9701

<div class="k-default-codeblock">
```

```
</div>
 1599/1599 ━━━━━━━━━━━━━━━━━━━━ 524s 326ms/step - loss: 1.5354 - mean_absolute_error: 0.9700


<div class="k-default-codeblock">
```
Epoch 2/2

/home/humbulani/tensorflow-env/env/lib/python3.11/site-packages/keras/src/trainers/epoch_iterator.py:151: UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset.
  self._interrupted_warning()

```
</div>
    
    1/1599 [37m━━━━━━━━━━━━━━━━━━━━  15:10 570ms/step - loss: 1.0496 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
    2/1599 [37m━━━━━━━━━━━━━━━━━━━━  11:15 423ms/step - loss: 1.0938 - mean_absolute_error: 0.8357

<div class="k-default-codeblock">
```

```
</div>
    3/1599 [37m━━━━━━━━━━━━━━━━━━━━  11:28 431ms/step - loss: 1.1301 - mean_absolute_error: 0.8512

<div class="k-default-codeblock">
```

```
</div>
    4/1599 [37m━━━━━━━━━━━━━━━━━━━━  11:48 444ms/step - loss: 1.1428 - mean_absolute_error: 0.8564

<div class="k-default-codeblock">
```

```
</div>
    5/1599 [37m━━━━━━━━━━━━━━━━━━━━  11:48 444ms/step - loss: 1.1508 - mean_absolute_error: 0.8585

<div class="k-default-codeblock">
```

```
</div>
    6/1599 [37m━━━━━━━━━━━━━━━━━━━━  11:26 431ms/step - loss: 1.1566 - mean_absolute_error: 0.8594

<div class="k-default-codeblock">
```

```
</div>
    7/1599 [37m━━━━━━━━━━━━━━━━━━━━  11:19 427ms/step - loss: 1.1591 - mean_absolute_error: 0.8599

<div class="k-default-codeblock">
```

```
</div>
    8/1599 [37m━━━━━━━━━━━━━━━━━━━━  11:12 423ms/step - loss: 1.1578 - mean_absolute_error: 0.8590

<div class="k-default-codeblock">
```

```
</div>
    9/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:54 412ms/step - loss: 1.1560 - mean_absolute_error: 0.8582

<div class="k-default-codeblock">
```

```
</div>
   10/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:50 409ms/step - loss: 1.1552 - mean_absolute_error: 0.8579

<div class="k-default-codeblock">
```

```
</div>
   11/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:33 399ms/step - loss: 1.1532 - mean_absolute_error: 0.8573

<div class="k-default-codeblock">
```

```
</div>
   12/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:24 394ms/step - loss: 1.1511 - mean_absolute_error: 0.8563

<div class="k-default-codeblock">
```

```
</div>
   13/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:21 392ms/step - loss: 1.1489 - mean_absolute_error: 0.8555

<div class="k-default-codeblock">
```

```
</div>
   14/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:15 388ms/step - loss: 1.1464 - mean_absolute_error: 0.8546

<div class="k-default-codeblock">
```

```
</div>
   15/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:13 387ms/step - loss: 1.1446 - mean_absolute_error: 0.8539

<div class="k-default-codeblock">
```

```
</div>
   16/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:11 386ms/step - loss: 1.1429 - mean_absolute_error: 0.8532

<div class="k-default-codeblock">
```

```
</div>
   17/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:08 385ms/step - loss: 1.1409 - mean_absolute_error: 0.8525

<div class="k-default-codeblock">
```

```
</div>
   18/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:06 384ms/step - loss: 1.1387 - mean_absolute_error: 0.8516

<div class="k-default-codeblock">
```

```
</div>
   19/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:05 383ms/step - loss: 1.1366 - mean_absolute_error: 0.8508

<div class="k-default-codeblock">
```

```
</div>
   20/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:01 381ms/step - loss: 1.1346 - mean_absolute_error: 0.8500

<div class="k-default-codeblock">
```

```
</div>
   21/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:59 380ms/step - loss: 1.1327 - mean_absolute_error: 0.8493 

<div class="k-default-codeblock">
```

```
</div>
   22/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:01 381ms/step - loss: 1.1311 - mean_absolute_error: 0.8487

<div class="k-default-codeblock">
```

```
</div>
   23/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:03 383ms/step - loss: 1.1299 - mean_absolute_error: 0.8482

<div class="k-default-codeblock">
```

```
</div>
   24/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:05 385ms/step - loss: 1.1288 - mean_absolute_error: 0.8477

<div class="k-default-codeblock">
```

```
</div>
   25/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:05 385ms/step - loss: 1.1276 - mean_absolute_error: 0.8472

<div class="k-default-codeblock">
```

```
</div>
   26/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:03 383ms/step - loss: 1.1267 - mean_absolute_error: 0.8468

<div class="k-default-codeblock">
```

```
</div>
   27/1599 [37m━━━━━━━━━━━━━━━━━━━━  10:01 383ms/step - loss: 1.1257 - mean_absolute_error: 0.8464

<div class="k-default-codeblock">
```

```
</div>
   28/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:57 380ms/step - loss: 1.1246 - mean_absolute_error: 0.8460 

<div class="k-default-codeblock">
```

```
</div>
   29/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:55 380ms/step - loss: 1.1234 - mean_absolute_error: 0.8456

<div class="k-default-codeblock">
```

```
</div>
   30/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:55 379ms/step - loss: 1.1224 - mean_absolute_error: 0.8451

<div class="k-default-codeblock">
```

```
</div>
   31/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:53 379ms/step - loss: 1.1213 - mean_absolute_error: 0.8448

<div class="k-default-codeblock">
```

```
</div>
   32/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:55 380ms/step - loss: 1.1203 - mean_absolute_error: 0.8444

<div class="k-default-codeblock">
```

```
</div>
   33/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:52 379ms/step - loss: 1.1194 - mean_absolute_error: 0.8440

<div class="k-default-codeblock">
```

```
</div>
   34/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:52 378ms/step - loss: 1.1186 - mean_absolute_error: 0.8437

<div class="k-default-codeblock">
```

```
</div>
   35/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:51 378ms/step - loss: 1.1178 - mean_absolute_error: 0.8434

<div class="k-default-codeblock">
```

```
</div>
   36/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:52 379ms/step - loss: 1.1169 - mean_absolute_error: 0.8431

<div class="k-default-codeblock">
```

```
</div>
   37/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:51 378ms/step - loss: 1.1161 - mean_absolute_error: 0.8428

<div class="k-default-codeblock">
```

```
</div>
   38/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:51 379ms/step - loss: 1.1152 - mean_absolute_error: 0.8425

<div class="k-default-codeblock">
```

```
</div>
   39/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:51 379ms/step - loss: 1.1144 - mean_absolute_error: 0.8423

<div class="k-default-codeblock">
```

```
</div>
   40/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:52 380ms/step - loss: 1.1136 - mean_absolute_error: 0.8420

<div class="k-default-codeblock">
```

```
</div>
   41/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:51 380ms/step - loss: 1.1130 - mean_absolute_error: 0.8418

<div class="k-default-codeblock">
```

```
</div>
   42/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:51 380ms/step - loss: 1.1123 - mean_absolute_error: 0.8416

<div class="k-default-codeblock">
```

```
</div>
   43/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:50 380ms/step - loss: 1.1116 - mean_absolute_error: 0.8414

<div class="k-default-codeblock">
```

```
</div>
   44/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:49 379ms/step - loss: 1.1108 - mean_absolute_error: 0.8411

<div class="k-default-codeblock">
```

```
</div>
   45/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:48 379ms/step - loss: 1.1100 - mean_absolute_error: 0.8408

<div class="k-default-codeblock">
```

```
</div>
   46/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 379ms/step - loss: 1.1092 - mean_absolute_error: 0.8406

<div class="k-default-codeblock">
```

```
</div>
   47/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 379ms/step - loss: 1.1084 - mean_absolute_error: 0.8403

<div class="k-default-codeblock">
```

```
</div>
   48/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 379ms/step - loss: 1.1076 - mean_absolute_error: 0.8400

<div class="k-default-codeblock">
```

```
</div>
   49/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:46 378ms/step - loss: 1.1069 - mean_absolute_error: 0.8398

<div class="k-default-codeblock">
```

```
</div>
   50/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 379ms/step - loss: 1.1062 - mean_absolute_error: 0.8396

<div class="k-default-codeblock">
```

```
</div>
   51/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 379ms/step - loss: 1.1055 - mean_absolute_error: 0.8393

<div class="k-default-codeblock">
```

```
</div>
   52/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:46 379ms/step - loss: 1.1048 - mean_absolute_error: 0.8391

<div class="k-default-codeblock">
```

```
</div>
   53/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:45 379ms/step - loss: 1.1041 - mean_absolute_error: 0.8389

<div class="k-default-codeblock">
```

```
</div>
   54/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:45 379ms/step - loss: 1.1034 - mean_absolute_error: 0.8387

<div class="k-default-codeblock">
```

```
</div>
   55/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:46 380ms/step - loss: 1.1027 - mean_absolute_error: 0.8384

<div class="k-default-codeblock">
```

```
</div>
   56/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:46 380ms/step - loss: 1.1021 - mean_absolute_error: 0.8382

<div class="k-default-codeblock">
```

```
</div>
   57/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:46 381ms/step - loss: 1.1014 - mean_absolute_error: 0.8380

<div class="k-default-codeblock">
```

```
</div>
   58/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 381ms/step - loss: 1.1008 - mean_absolute_error: 0.8378

<div class="k-default-codeblock">
```

```
</div>
   59/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 381ms/step - loss: 1.1002 - mean_absolute_error: 0.8376

<div class="k-default-codeblock">
```

```
</div>
   60/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:47 382ms/step - loss: 1.0997 - mean_absolute_error: 0.8375

<div class="k-default-codeblock">
```

```
</div>
   61/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:46 381ms/step - loss: 1.0991 - mean_absolute_error: 0.8373

<div class="k-default-codeblock">
```

```
</div>
   62/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:46 381ms/step - loss: 1.0985 - mean_absolute_error: 0.8371

<div class="k-default-codeblock">
```

```
</div>
   63/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:45 381ms/step - loss: 1.0980 - mean_absolute_error: 0.8369

<div class="k-default-codeblock">
```

```
</div>
   64/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:44 381ms/step - loss: 1.0975 - mean_absolute_error: 0.8368

<div class="k-default-codeblock">
```

```
</div>
   65/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:44 381ms/step - loss: 1.0969 - mean_absolute_error: 0.8366

<div class="k-default-codeblock">
```

```
</div>
   66/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:44 381ms/step - loss: 1.0964 - mean_absolute_error: 0.8364

<div class="k-default-codeblock">
```

```
</div>
   67/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:42 381ms/step - loss: 1.0959 - mean_absolute_error: 0.8363

<div class="k-default-codeblock">
```

```
</div>
   68/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:42 381ms/step - loss: 1.0955 - mean_absolute_error: 0.8361

<div class="k-default-codeblock">
```

```
</div>
   69/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:41 380ms/step - loss: 1.0950 - mean_absolute_error: 0.8359

<div class="k-default-codeblock">
```

```
</div>
   70/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:40 380ms/step - loss: 1.0945 - mean_absolute_error: 0.8358

<div class="k-default-codeblock">
```

```
</div>
   71/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:40 380ms/step - loss: 1.0940 - mean_absolute_error: 0.8356

<div class="k-default-codeblock">
```

```
</div>
   72/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:40 380ms/step - loss: 1.0936 - mean_absolute_error: 0.8354

<div class="k-default-codeblock">
```

```
</div>
   73/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:40 380ms/step - loss: 1.0932 - mean_absolute_error: 0.8353

<div class="k-default-codeblock">
```

```
</div>
   74/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:40 380ms/step - loss: 1.0927 - mean_absolute_error: 0.8351

<div class="k-default-codeblock">
```

```
</div>
   75/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:39 380ms/step - loss: 1.0922 - mean_absolute_error: 0.8350

<div class="k-default-codeblock">
```

```
</div>
   76/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:38 380ms/step - loss: 1.0917 - mean_absolute_error: 0.8348

<div class="k-default-codeblock">
```

```
</div>
   77/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:38 380ms/step - loss: 1.0913 - mean_absolute_error: 0.8346

<div class="k-default-codeblock">
```

```
</div>
   78/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:38 380ms/step - loss: 1.0908 - mean_absolute_error: 0.8345

<div class="k-default-codeblock">
```

```
</div>
   79/1599 [37m━━━━━━━━━━━━━━━━━━━━  9:37 380ms/step - loss: 1.0904 - mean_absolute_error: 0.8343

<div class="k-default-codeblock">
```

```
</div>
   80/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:37 380ms/step - loss: 1.0899 - mean_absolute_error: 0.8342

<div class="k-default-codeblock">
```

```
</div>
   81/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:36 380ms/step - loss: 1.0895 - mean_absolute_error: 0.8340

<div class="k-default-codeblock">
```

```
</div>
   82/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:36 380ms/step - loss: 1.0891 - mean_absolute_error: 0.8339

<div class="k-default-codeblock">
```

```
</div>
   83/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:35 380ms/step - loss: 1.0887 - mean_absolute_error: 0.8337

<div class="k-default-codeblock">
```

```
</div>
   84/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:35 380ms/step - loss: 1.0883 - mean_absolute_error: 0.8336

<div class="k-default-codeblock">
```

```
</div>
   85/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:34 380ms/step - loss: 1.0879 - mean_absolute_error: 0.8335

<div class="k-default-codeblock">
```

```
</div>
   86/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:34 380ms/step - loss: 1.0876 - mean_absolute_error: 0.8333

<div class="k-default-codeblock">
```

```
</div>
   87/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:33 380ms/step - loss: 1.0872 - mean_absolute_error: 0.8332

<div class="k-default-codeblock">
```

```
</div>
   88/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:33 380ms/step - loss: 1.0869 - mean_absolute_error: 0.8331

<div class="k-default-codeblock">
```

```
</div>
   89/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:32 379ms/step - loss: 1.0865 - mean_absolute_error: 0.8330

<div class="k-default-codeblock">
```

```
</div>
   90/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:32 380ms/step - loss: 1.0862 - mean_absolute_error: 0.8329

<div class="k-default-codeblock">
```

```
</div>
   91/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:32 380ms/step - loss: 1.0859 - mean_absolute_error: 0.8327

<div class="k-default-codeblock">
```

```
</div>
   92/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:32 380ms/step - loss: 1.0856 - mean_absolute_error: 0.8326

<div class="k-default-codeblock">
```

```
</div>
   93/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:31 380ms/step - loss: 1.0853 - mean_absolute_error: 0.8325

<div class="k-default-codeblock">
```

```
</div>
   94/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:31 380ms/step - loss: 1.0850 - mean_absolute_error: 0.8324

<div class="k-default-codeblock">
```

```
</div>
   95/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:31 380ms/step - loss: 1.0847 - mean_absolute_error: 0.8323

<div class="k-default-codeblock">
```

```
</div>
   96/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:31 380ms/step - loss: 1.0844 - mean_absolute_error: 0.8322

<div class="k-default-codeblock">
```

```
</div>
   97/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:31 380ms/step - loss: 1.0842 - mean_absolute_error: 0.8321

<div class="k-default-codeblock">
```

```
</div>
   98/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:31 381ms/step - loss: 1.0839 - mean_absolute_error: 0.8321

<div class="k-default-codeblock">
```

```
</div>
   99/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:31 381ms/step - loss: 1.0837 - mean_absolute_error: 0.8320

<div class="k-default-codeblock">
```

```
</div>
  100/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:30 380ms/step - loss: 1.0835 - mean_absolute_error: 0.8319

<div class="k-default-codeblock">
```

```
</div>
  101/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:29 380ms/step - loss: 1.0833 - mean_absolute_error: 0.8318

<div class="k-default-codeblock">
```

```
</div>
  102/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:28 380ms/step - loss: 1.0830 - mean_absolute_error: 0.8318

<div class="k-default-codeblock">
```

```
</div>
  103/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:28 380ms/step - loss: 1.0828 - mean_absolute_error: 0.8317

<div class="k-default-codeblock">
```

```
</div>
  104/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:27 380ms/step - loss: 1.0826 - mean_absolute_error: 0.8316

<div class="k-default-codeblock">
```

```
</div>
  105/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:27 380ms/step - loss: 1.0824 - mean_absolute_error: 0.8315

<div class="k-default-codeblock">
```

```
</div>
  106/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:26 380ms/step - loss: 1.0822 - mean_absolute_error: 0.8315

<div class="k-default-codeblock">
```

```
</div>
  107/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:26 380ms/step - loss: 1.0820 - mean_absolute_error: 0.8314

<div class="k-default-codeblock">
```

```
</div>
  108/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:25 379ms/step - loss: 1.0818 - mean_absolute_error: 0.8313

<div class="k-default-codeblock">
```

```
</div>
  109/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:25 380ms/step - loss: 1.0816 - mean_absolute_error: 0.8312

<div class="k-default-codeblock">
```

```
</div>
  110/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:25 380ms/step - loss: 1.0814 - mean_absolute_error: 0.8311

<div class="k-default-codeblock">
```

```
</div>
  111/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:25 380ms/step - loss: 1.0812 - mean_absolute_error: 0.8311

<div class="k-default-codeblock">
```

```
</div>
  112/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:25 381ms/step - loss: 1.0810 - mean_absolute_error: 0.8310

<div class="k-default-codeblock">
```

```
</div>
  113/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:26 381ms/step - loss: 1.0808 - mean_absolute_error: 0.8309

<div class="k-default-codeblock">
```

```
</div>
  114/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:27 382ms/step - loss: 1.0806 - mean_absolute_error: 0.8309

<div class="k-default-codeblock">
```

```
</div>
  115/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:28 383ms/step - loss: 1.0805 - mean_absolute_error: 0.8308

<div class="k-default-codeblock">
```

```
</div>
  116/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:28 383ms/step - loss: 1.0803 - mean_absolute_error: 0.8308

<div class="k-default-codeblock">
```

```
</div>
  117/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:28 384ms/step - loss: 1.0801 - mean_absolute_error: 0.8307

<div class="k-default-codeblock">
```

```
</div>
  118/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:29 384ms/step - loss: 1.0799 - mean_absolute_error: 0.8306

<div class="k-default-codeblock">
```

```
</div>
  119/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:28 384ms/step - loss: 1.0798 - mean_absolute_error: 0.8306

<div class="k-default-codeblock">
```

```
</div>
  120/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:28 384ms/step - loss: 1.0796 - mean_absolute_error: 0.8305

<div class="k-default-codeblock">
```

```
</div>
  121/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:27 384ms/step - loss: 1.0794 - mean_absolute_error: 0.8304

<div class="k-default-codeblock">
```

```
</div>
  122/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:26 383ms/step - loss: 1.0793 - mean_absolute_error: 0.8304

<div class="k-default-codeblock">
```

```
</div>
  123/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:24 383ms/step - loss: 1.0791 - mean_absolute_error: 0.8303

<div class="k-default-codeblock">
```

```
</div>
  124/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:23 382ms/step - loss: 1.0789 - mean_absolute_error: 0.8303

<div class="k-default-codeblock">
```

```
</div>
  125/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:22 382ms/step - loss: 1.0788 - mean_absolute_error: 0.8302

<div class="k-default-codeblock">
```

```
</div>
  126/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:22 382ms/step - loss: 1.0786 - mean_absolute_error: 0.8301

<div class="k-default-codeblock">
```

```
</div>
  127/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:21 381ms/step - loss: 1.0784 - mean_absolute_error: 0.8301

<div class="k-default-codeblock">
```

```
</div>
  128/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:21 381ms/step - loss: 1.0783 - mean_absolute_error: 0.8300

<div class="k-default-codeblock">
```

```
</div>
  129/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:20 381ms/step - loss: 1.0781 - mean_absolute_error: 0.8299

<div class="k-default-codeblock">
```

```
</div>
  130/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:19 381ms/step - loss: 1.0780 - mean_absolute_error: 0.8299

<div class="k-default-codeblock">
```

```
</div>
  131/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:18 381ms/step - loss: 1.0778 - mean_absolute_error: 0.8298

<div class="k-default-codeblock">
```

```
</div>
  132/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:17 380ms/step - loss: 1.0777 - mean_absolute_error: 0.8297

<div class="k-default-codeblock">
```

```
</div>
  133/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:17 380ms/step - loss: 1.0775 - mean_absolute_error: 0.8297

<div class="k-default-codeblock">
```

```
</div>
  134/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:16 380ms/step - loss: 1.0773 - mean_absolute_error: 0.8296

<div class="k-default-codeblock">
```

```
</div>
  135/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:17 381ms/step - loss: 1.0772 - mean_absolute_error: 0.8296

<div class="k-default-codeblock">
```

```
</div>
  136/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:17 381ms/step - loss: 1.0770 - mean_absolute_error: 0.8295

<div class="k-default-codeblock">
```

```
</div>
  137/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:18 382ms/step - loss: 1.0769 - mean_absolute_error: 0.8294

<div class="k-default-codeblock">
```

```
</div>
  138/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:18 382ms/step - loss: 1.0767 - mean_absolute_error: 0.8294

<div class="k-default-codeblock">
```

```
</div>
  139/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:18 382ms/step - loss: 1.0766 - mean_absolute_error: 0.8293

<div class="k-default-codeblock">
```

```
</div>
  140/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:18 383ms/step - loss: 1.0765 - mean_absolute_error: 0.8293

<div class="k-default-codeblock">
```

```
</div>
  141/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:17 383ms/step - loss: 1.0763 - mean_absolute_error: 0.8292

<div class="k-default-codeblock">
```

```
</div>
  142/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:17 383ms/step - loss: 1.0762 - mean_absolute_error: 0.8292

<div class="k-default-codeblock">
```

```
</div>
  143/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:16 382ms/step - loss: 1.0760 - mean_absolute_error: 0.8291

<div class="k-default-codeblock">
```

```
</div>
  144/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:15 382ms/step - loss: 1.0759 - mean_absolute_error: 0.8291

<div class="k-default-codeblock">
```

```
</div>
  145/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:15 382ms/step - loss: 1.0758 - mean_absolute_error: 0.8290

<div class="k-default-codeblock">
```

```
</div>
  146/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:14 382ms/step - loss: 1.0756 - mean_absolute_error: 0.8290

<div class="k-default-codeblock">
```

```
</div>
  147/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:14 382ms/step - loss: 1.0755 - mean_absolute_error: 0.8289

<div class="k-default-codeblock">
```

```
</div>
  148/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:14 382ms/step - loss: 1.0754 - mean_absolute_error: 0.8289

<div class="k-default-codeblock">
```

```
</div>
  149/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:14 382ms/step - loss: 1.0752 - mean_absolute_error: 0.8288

<div class="k-default-codeblock">
```

```
</div>
  150/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:14 383ms/step - loss: 1.0751 - mean_absolute_error: 0.8287

<div class="k-default-codeblock">
```

```
</div>
  151/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:13 382ms/step - loss: 1.0749 - mean_absolute_error: 0.8287

<div class="k-default-codeblock">
```

```
</div>
  152/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:13 383ms/step - loss: 1.0748 - mean_absolute_error: 0.8286

<div class="k-default-codeblock">
```

```
</div>
  153/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:13 383ms/step - loss: 1.0746 - mean_absolute_error: 0.8286

<div class="k-default-codeblock">
```

```
</div>
  154/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:13 383ms/step - loss: 1.0745 - mean_absolute_error: 0.8285

<div class="k-default-codeblock">
```

```
</div>
  155/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:12 383ms/step - loss: 1.0744 - mean_absolute_error: 0.8285

<div class="k-default-codeblock">
```

```
</div>
  156/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:12 383ms/step - loss: 1.0742 - mean_absolute_error: 0.8284

<div class="k-default-codeblock">
```

```
</div>
  157/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:12 383ms/step - loss: 1.0741 - mean_absolute_error: 0.8284

<div class="k-default-codeblock">
```

```
</div>
  158/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:11 383ms/step - loss: 1.0740 - mean_absolute_error: 0.8283

<div class="k-default-codeblock">
```

```
</div>
  159/1599 ━[37m━━━━━━━━━━━━━━━━━━━  9:11 383ms/step - loss: 1.0738 - mean_absolute_error: 0.8283

<div class="k-default-codeblock">
```

```
</div>
  160/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 383ms/step - loss: 1.0737 - mean_absolute_error: 0.8282

<div class="k-default-codeblock">
```

```
</div>
  161/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 383ms/step - loss: 1.0736 - mean_absolute_error: 0.8282

<div class="k-default-codeblock">
```

```
</div>
  162/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 383ms/step - loss: 1.0734 - mean_absolute_error: 0.8281

<div class="k-default-codeblock">
```

```
</div>
  163/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 383ms/step - loss: 1.0733 - mean_absolute_error: 0.8281

<div class="k-default-codeblock">
```

```
</div>
  164/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 384ms/step - loss: 1.0732 - mean_absolute_error: 0.8280

<div class="k-default-codeblock">
```

```
</div>
  165/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:11 384ms/step - loss: 1.0730 - mean_absolute_error: 0.8280

<div class="k-default-codeblock">
```

```
</div>
  166/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:11 385ms/step - loss: 1.0729 - mean_absolute_error: 0.8279

<div class="k-default-codeblock">
```

```
</div>
  167/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:11 385ms/step - loss: 1.0727 - mean_absolute_error: 0.8279

<div class="k-default-codeblock">
```

```
</div>
  168/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 385ms/step - loss: 1.0726 - mean_absolute_error: 0.8278

<div class="k-default-codeblock">
```

```
</div>
  169/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 385ms/step - loss: 1.0725 - mean_absolute_error: 0.8278

<div class="k-default-codeblock">
```

```
</div>
  170/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:10 385ms/step - loss: 1.0723 - mean_absolute_error: 0.8277

<div class="k-default-codeblock">
```

```
</div>
  171/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:09 385ms/step - loss: 1.0722 - mean_absolute_error: 0.8277

<div class="k-default-codeblock">
```

```
</div>
  172/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:09 385ms/step - loss: 1.0721 - mean_absolute_error: 0.8276

<div class="k-default-codeblock">
```

```
</div>
  173/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:08 384ms/step - loss: 1.0719 - mean_absolute_error: 0.8276

<div class="k-default-codeblock">
```

```
</div>
  174/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:07 384ms/step - loss: 1.0718 - mean_absolute_error: 0.8275

<div class="k-default-codeblock">
```

```
</div>
  175/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:06 384ms/step - loss: 1.0717 - mean_absolute_error: 0.8275

<div class="k-default-codeblock">
```

```
</div>
  176/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:05 383ms/step - loss: 1.0716 - mean_absolute_error: 0.8274

<div class="k-default-codeblock">
```

```
</div>
  177/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:04 383ms/step - loss: 1.0714 - mean_absolute_error: 0.8274

<div class="k-default-codeblock">
```

```
</div>
  178/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:04 383ms/step - loss: 1.0713 - mean_absolute_error: 0.8273

<div class="k-default-codeblock">
```

```
</div>
  179/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:03 383ms/step - loss: 1.0712 - mean_absolute_error: 0.8273

<div class="k-default-codeblock">
```

```
</div>
  180/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:02 382ms/step - loss: 1.0711 - mean_absolute_error: 0.8273

<div class="k-default-codeblock">
```

```
</div>
  181/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:02 382ms/step - loss: 1.0710 - mean_absolute_error: 0.8272

<div class="k-default-codeblock">
```

```
</div>
  182/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:01 382ms/step - loss: 1.0709 - mean_absolute_error: 0.8272

<div class="k-default-codeblock">
```

```
</div>
  183/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:00 382ms/step - loss: 1.0708 - mean_absolute_error: 0.8271

<div class="k-default-codeblock">
```

```
</div>
  184/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:00 382ms/step - loss: 1.0707 - mean_absolute_error: 0.8271

<div class="k-default-codeblock">
```

```
</div>
  185/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:59 382ms/step - loss: 1.0706 - mean_absolute_error: 0.8271

<div class="k-default-codeblock">
```

```
</div>
  186/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:00 382ms/step - loss: 1.0705 - mean_absolute_error: 0.8270

<div class="k-default-codeblock">
```

```
</div>
  187/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:59 382ms/step - loss: 1.0704 - mean_absolute_error: 0.8270

<div class="k-default-codeblock">
```

```
</div>
  188/1599 ━━[37m━━━━━━━━━━━━━━━━━━  9:00 383ms/step - loss: 1.0703 - mean_absolute_error: 0.8270

<div class="k-default-codeblock">
```

```
</div>
  189/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:59 383ms/step - loss: 1.0702 - mean_absolute_error: 0.8269

<div class="k-default-codeblock">
```

```
</div>
  190/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:59 383ms/step - loss: 1.0701 - mean_absolute_error: 0.8269

<div class="k-default-codeblock">
```

```
</div>
  191/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:59 383ms/step - loss: 1.0700 - mean_absolute_error: 0.8269

<div class="k-default-codeblock">
```

```
</div>
  192/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:58 383ms/step - loss: 1.0699 - mean_absolute_error: 0.8269

<div class="k-default-codeblock">
```

```
</div>
  193/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:57 383ms/step - loss: 1.0698 - mean_absolute_error: 0.8268

<div class="k-default-codeblock">
```

```
</div>
  194/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:57 382ms/step - loss: 1.0698 - mean_absolute_error: 0.8268

<div class="k-default-codeblock">
```

```
</div>
  195/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:57 382ms/step - loss: 1.0697 - mean_absolute_error: 0.8268

<div class="k-default-codeblock">
```

```
</div>
  196/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:56 382ms/step - loss: 1.0696 - mean_absolute_error: 0.8267

<div class="k-default-codeblock">
```

```
</div>
  197/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:56 382ms/step - loss: 1.0695 - mean_absolute_error: 0.8267

<div class="k-default-codeblock">
```

```
</div>
  198/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:55 382ms/step - loss: 1.0694 - mean_absolute_error: 0.8267

<div class="k-default-codeblock">
```

```
</div>
  199/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:55 383ms/step - loss: 1.0693 - mean_absolute_error: 0.8267

<div class="k-default-codeblock">
```

```
</div>
  200/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:55 383ms/step - loss: 1.0692 - mean_absolute_error: 0.8266

<div class="k-default-codeblock">
```

```
</div>
  201/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:54 383ms/step - loss: 1.0692 - mean_absolute_error: 0.8266

<div class="k-default-codeblock">
```

```
</div>
  202/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:54 383ms/step - loss: 1.0691 - mean_absolute_error: 0.8266

<div class="k-default-codeblock">
```

```
</div>
  203/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:54 383ms/step - loss: 1.0690 - mean_absolute_error: 0.8266

<div class="k-default-codeblock">
```

```
</div>
  204/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:53 383ms/step - loss: 1.0689 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
  205/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:53 383ms/step - loss: 1.0689 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
  206/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:53 383ms/step - loss: 1.0688 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
  207/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:53 383ms/step - loss: 1.0687 - mean_absolute_error: 0.8265

<div class="k-default-codeblock">
```

```
</div>
  208/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:52 383ms/step - loss: 1.0686 - mean_absolute_error: 0.8264

<div class="k-default-codeblock">
```

```
</div>
  209/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:52 383ms/step - loss: 1.0686 - mean_absolute_error: 0.8264

<div class="k-default-codeblock">
```

```
</div>
  210/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:52 383ms/step - loss: 1.0685 - mean_absolute_error: 0.8264

<div class="k-default-codeblock">
```

```
</div>
  211/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:52 383ms/step - loss: 1.0684 - mean_absolute_error: 0.8264

<div class="k-default-codeblock">
```

```
</div>
  212/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:51 383ms/step - loss: 1.0683 - mean_absolute_error: 0.8263

<div class="k-default-codeblock">
```

```
</div>
  213/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:51 383ms/step - loss: 1.0683 - mean_absolute_error: 0.8263

<div class="k-default-codeblock">
```

```
</div>
  214/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:51 383ms/step - loss: 1.0682 - mean_absolute_error: 0.8263

<div class="k-default-codeblock">
```

```
</div>
  215/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:50 383ms/step - loss: 1.0681 - mean_absolute_error: 0.8263

<div class="k-default-codeblock">
```

```
</div>
  216/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:50 384ms/step - loss: 1.0680 - mean_absolute_error: 0.8262

<div class="k-default-codeblock">
```

```
</div>
  217/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:50 384ms/step - loss: 1.0680 - mean_absolute_error: 0.8262

<div class="k-default-codeblock">
```

```
</div>
  218/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:50 384ms/step - loss: 1.0679 - mean_absolute_error: 0.8262

<div class="k-default-codeblock">
```

```
</div>
  219/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:49 384ms/step - loss: 1.0678 - mean_absolute_error: 0.8262

<div class="k-default-codeblock">
```

```
</div>
  220/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:49 384ms/step - loss: 1.0677 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
  221/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:48 384ms/step - loss: 1.0677 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
  222/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:48 384ms/step - loss: 1.0676 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
  223/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:48 384ms/step - loss: 1.0675 - mean_absolute_error: 0.8261

<div class="k-default-codeblock">
```

```
</div>
  224/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:47 384ms/step - loss: 1.0674 - mean_absolute_error: 0.8260

<div class="k-default-codeblock">
```

```
</div>
  225/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:47 384ms/step - loss: 1.0674 - mean_absolute_error: 0.8260

<div class="k-default-codeblock">
```

```
</div>
  226/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:47 384ms/step - loss: 1.0673 - mean_absolute_error: 0.8260

<div class="k-default-codeblock">
```

```
</div>
  227/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:46 384ms/step - loss: 1.0672 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
  228/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:46 384ms/step - loss: 1.0672 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
  229/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:45 384ms/step - loss: 1.0671 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
  230/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:45 384ms/step - loss: 1.0670 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
  231/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:45 384ms/step - loss: 1.0670 - mean_absolute_error: 0.8259

<div class="k-default-codeblock">
```

```
</div>
  232/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:44 384ms/step - loss: 1.0669 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
  233/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:44 384ms/step - loss: 1.0669 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
  234/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:44 384ms/step - loss: 1.0668 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
  235/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:43 384ms/step - loss: 1.0667 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
  236/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:43 384ms/step - loss: 1.0667 - mean_absolute_error: 0.8258

<div class="k-default-codeblock">
```

```
</div>
  237/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:42 384ms/step - loss: 1.0666 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
  238/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:42 384ms/step - loss: 1.0665 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
  239/1599 ━━[37m━━━━━━━━━━━━━━━━━━  8:42 384ms/step - loss: 1.0665 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
  240/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:42 384ms/step - loss: 1.0664 - mean_absolute_error: 0.8257

<div class="k-default-codeblock">
```

```
</div>
  241/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:42 385ms/step - loss: 1.0663 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  242/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:42 385ms/step - loss: 1.0663 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  243/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:43 386ms/step - loss: 1.0662 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  244/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:43 386ms/step - loss: 1.0662 - mean_absolute_error: 0.8256

<div class="k-default-codeblock">
```

```
</div>
  245/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:43 386ms/step - loss: 1.0661 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  246/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:42 386ms/step - loss: 1.0660 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  247/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:42 386ms/step - loss: 1.0659 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  248/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:42 386ms/step - loss: 1.0659 - mean_absolute_error: 0.8255

<div class="k-default-codeblock">
```

```
</div>
  249/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:41 386ms/step - loss: 1.0658 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  250/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:40 386ms/step - loss: 1.0657 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  251/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:40 386ms/step - loss: 1.0657 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  252/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:39 386ms/step - loss: 1.0656 - mean_absolute_error: 0.8254

<div class="k-default-codeblock">
```

```
</div>
  253/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:38 385ms/step - loss: 1.0655 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  254/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:38 385ms/step - loss: 1.0654 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  255/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:37 385ms/step - loss: 1.0654 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  256/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:36 385ms/step - loss: 1.0653 - mean_absolute_error: 0.8253

<div class="k-default-codeblock">
```

```
</div>
  257/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:36 385ms/step - loss: 1.0652 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  258/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:35 385ms/step - loss: 1.0652 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  259/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:35 385ms/step - loss: 1.0651 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  260/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:34 384ms/step - loss: 1.0650 - mean_absolute_error: 0.8252

<div class="k-default-codeblock">
```

```
</div>
  261/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:34 384ms/step - loss: 1.0649 - mean_absolute_error: 0.8251

<div class="k-default-codeblock">
```

```
</div>
  262/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:33 384ms/step - loss: 1.0648 - mean_absolute_error: 0.8251

<div class="k-default-codeblock">
```

```
</div>
  263/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:33 384ms/step - loss: 1.0648 - mean_absolute_error: 0.8251

<div class="k-default-codeblock">
```

```
</div>
  264/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:33 385ms/step - loss: 1.0647 - mean_absolute_error: 0.8250

<div class="k-default-codeblock">
```

```
</div>
  265/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:33 385ms/step - loss: 1.0646 - mean_absolute_error: 0.8250

<div class="k-default-codeblock">
```

```
</div>
  266/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:33 385ms/step - loss: 1.0645 - mean_absolute_error: 0.8250

<div class="k-default-codeblock">
```

```
</div>
  267/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:33 385ms/step - loss: 1.0644 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  268/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:32 385ms/step - loss: 1.0643 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  269/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:32 385ms/step - loss: 1.0643 - mean_absolute_error: 0.8249

<div class="k-default-codeblock">
```

```
</div>
  270/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:31 385ms/step - loss: 1.0642 - mean_absolute_error: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  271/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:31 385ms/step - loss: 1.0641 - mean_absolute_error: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  272/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:30 385ms/step - loss: 1.0640 - mean_absolute_error: 0.8248

<div class="k-default-codeblock">
```

```
</div>
  273/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:30 385ms/step - loss: 1.0639 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  274/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:29 385ms/step - loss: 1.0638 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  275/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:29 385ms/step - loss: 1.0638 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  276/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:29 385ms/step - loss: 1.0637 - mean_absolute_error: 0.8247

<div class="k-default-codeblock">
```

```
</div>
  277/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:28 385ms/step - loss: 1.0636 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  278/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:28 385ms/step - loss: 1.0635 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  279/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:28 385ms/step - loss: 1.0634 - mean_absolute_error: 0.8246

<div class="k-default-codeblock">
```

```
</div>
  280/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:27 385ms/step - loss: 1.0633 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  281/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:27 385ms/step - loss: 1.0633 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  282/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:27 385ms/step - loss: 1.0632 - mean_absolute_error: 0.8245

<div class="k-default-codeblock">
```

```
</div>
  283/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:27 386ms/step - loss: 1.0631 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  284/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:27 386ms/step - loss: 1.0630 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  285/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:27 386ms/step - loss: 1.0629 - mean_absolute_error: 0.8244

<div class="k-default-codeblock">
```

```
</div>
  286/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:26 386ms/step - loss: 1.0628 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  287/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:26 386ms/step - loss: 1.0627 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  288/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:25 386ms/step - loss: 1.0626 - mean_absolute_error: 0.8243

<div class="k-default-codeblock">
```

```
</div>
  289/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:25 386ms/step - loss: 1.0626 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  290/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:24 386ms/step - loss: 1.0625 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  291/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:24 386ms/step - loss: 1.0624 - mean_absolute_error: 0.8242

<div class="k-default-codeblock">
```

```
</div>
  292/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:23 386ms/step - loss: 1.0623 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  293/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:23 386ms/step - loss: 1.0622 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  294/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:23 386ms/step - loss: 1.0621 - mean_absolute_error: 0.8241

<div class="k-default-codeblock">
```

```
</div>
  295/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:22 386ms/step - loss: 1.0620 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  296/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:22 386ms/step - loss: 1.0619 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  297/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:22 386ms/step - loss: 1.0619 - mean_absolute_error: 0.8240

<div class="k-default-codeblock">
```

```
</div>
  298/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:21 386ms/step - loss: 1.0618 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  299/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:21 386ms/step - loss: 1.0617 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  300/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:21 386ms/step - loss: 1.0616 - mean_absolute_error: 0.8239

<div class="k-default-codeblock">
```

```
</div>
  301/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:21 387ms/step - loss: 1.0615 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  302/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:22 387ms/step - loss: 1.0615 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  303/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:22 387ms/step - loss: 1.0614 - mean_absolute_error: 0.8238

<div class="k-default-codeblock">
```

```
</div>
  304/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:21 388ms/step - loss: 1.0613 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  305/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:21 388ms/step - loss: 1.0612 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  306/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:21 388ms/step - loss: 1.0611 - mean_absolute_error: 0.8237

<div class="k-default-codeblock">
```

```
</div>
  307/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:20 388ms/step - loss: 1.0610 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  308/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:20 388ms/step - loss: 1.0610 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  309/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:19 387ms/step - loss: 1.0609 - mean_absolute_error: 0.8236

<div class="k-default-codeblock">
```

```
</div>
  310/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:19 387ms/step - loss: 1.0608 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  311/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:18 387ms/step - loss: 1.0607 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  312/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:18 387ms/step - loss: 1.0606 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  313/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:18 387ms/step - loss: 1.0606 - mean_absolute_error: 0.8235

<div class="k-default-codeblock">
```

```
</div>
  314/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:17 387ms/step - loss: 1.0605 - mean_absolute_error: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  315/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:17 387ms/step - loss: 1.0604 - mean_absolute_error: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  316/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:16 387ms/step - loss: 1.0603 - mean_absolute_error: 0.8234

<div class="k-default-codeblock">
```

```
</div>
  317/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:16 387ms/step - loss: 1.0602 - mean_absolute_error: 0.8233

<div class="k-default-codeblock">
```

```
</div>
  318/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:16 387ms/step - loss: 1.0602 - mean_absolute_error: 0.8233

<div class="k-default-codeblock">
```

```
</div>
  319/1599 ━━━[37m━━━━━━━━━━━━━━━━━  8:16 388ms/step - loss: 1.0601 - mean_absolute_error: 0.8233

<div class="k-default-codeblock">
```

```
</div>
  320/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:16 388ms/step - loss: 1.0600 - mean_absolute_error: 0.8232

<div class="k-default-codeblock">
```

```
</div>
  321/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:16 388ms/step - loss: 1.0599 - mean_absolute_error: 0.8232

<div class="k-default-codeblock">
```

```
</div>
  322/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:16 389ms/step - loss: 1.0599 - mean_absolute_error: 0.8232

<div class="k-default-codeblock">
```

```
</div>
  323/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:15 389ms/step - loss: 1.0598 - mean_absolute_error: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  324/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:15 389ms/step - loss: 1.0597 - mean_absolute_error: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  325/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:15 389ms/step - loss: 1.0596 - mean_absolute_error: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  326/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:14 389ms/step - loss: 1.0595 - mean_absolute_error: 0.8231

<div class="k-default-codeblock">
```

```
</div>
  327/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:14 389ms/step - loss: 1.0595 - mean_absolute_error: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  328/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:14 389ms/step - loss: 1.0594 - mean_absolute_error: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  329/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:13 389ms/step - loss: 1.0593 - mean_absolute_error: 0.8230

<div class="k-default-codeblock">
```

```
</div>
  330/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:13 389ms/step - loss: 1.0593 - mean_absolute_error: 0.8229

<div class="k-default-codeblock">
```

```
</div>
  331/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:12 389ms/step - loss: 1.0592 - mean_absolute_error: 0.8229

<div class="k-default-codeblock">
```

```
</div>
  332/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:12 389ms/step - loss: 1.0591 - mean_absolute_error: 0.8229

<div class="k-default-codeblock">
```

```
</div>
  333/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:11 388ms/step - loss: 1.0590 - mean_absolute_error: 0.8229

<div class="k-default-codeblock">
```

```
</div>
  334/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:11 388ms/step - loss: 1.0590 - mean_absolute_error: 0.8228

<div class="k-default-codeblock">
```

```
</div>
  335/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:10 388ms/step - loss: 1.0589 - mean_absolute_error: 0.8228

<div class="k-default-codeblock">
```

```
</div>
  336/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:10 388ms/step - loss: 1.0588 - mean_absolute_error: 0.8228

<div class="k-default-codeblock">
```

```
</div>
  337/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:10 388ms/step - loss: 1.0588 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  338/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:09 388ms/step - loss: 1.0587 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  339/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:09 388ms/step - loss: 1.0586 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  340/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:08 388ms/step - loss: 1.0585 - mean_absolute_error: 0.8227

<div class="k-default-codeblock">
```

```
</div>
  341/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:08 388ms/step - loss: 1.0585 - mean_absolute_error: 0.8226

<div class="k-default-codeblock">
```

```
</div>
  342/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:08 389ms/step - loss: 1.0584 - mean_absolute_error: 0.8226

<div class="k-default-codeblock">
```

```
</div>
  343/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:08 389ms/step - loss: 1.0583 - mean_absolute_error: 0.8226

<div class="k-default-codeblock">
```

```
</div>
  344/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:08 389ms/step - loss: 1.0583 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  345/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:08 389ms/step - loss: 1.0582 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  346/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:07 389ms/step - loss: 1.0581 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  347/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:07 389ms/step - loss: 1.0580 - mean_absolute_error: 0.8225

<div class="k-default-codeblock">
```

```
</div>
  348/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:07 389ms/step - loss: 1.0580 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  349/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:06 389ms/step - loss: 1.0579 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  350/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:06 389ms/step - loss: 1.0578 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  351/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:05 389ms/step - loss: 1.0578 - mean_absolute_error: 0.8224

<div class="k-default-codeblock">
```

```
</div>
  352/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:05 389ms/step - loss: 1.0577 - mean_absolute_error: 0.8223

<div class="k-default-codeblock">
```

```
</div>
  353/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:04 389ms/step - loss: 1.0577 - mean_absolute_error: 0.8223

<div class="k-default-codeblock">
```

```
</div>
  354/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:04 389ms/step - loss: 1.0576 - mean_absolute_error: 0.8223

<div class="k-default-codeblock">
```

```
</div>
  355/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:04 389ms/step - loss: 1.0575 - mean_absolute_error: 0.8223

<div class="k-default-codeblock">
```

```
</div>
  356/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:03 389ms/step - loss: 1.0575 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  357/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:03 389ms/step - loss: 1.0574 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  358/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:02 389ms/step - loss: 1.0573 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  359/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:02 389ms/step - loss: 1.0573 - mean_absolute_error: 0.8222

<div class="k-default-codeblock">
```

```
</div>
  360/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:01 389ms/step - loss: 1.0572 - mean_absolute_error: 0.8221

<div class="k-default-codeblock">
```

```
</div>
  361/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:01 389ms/step - loss: 1.0571 - mean_absolute_error: 0.8221

<div class="k-default-codeblock">
```

```
</div>
  362/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:00 389ms/step - loss: 1.0571 - mean_absolute_error: 0.8221

<div class="k-default-codeblock">
```

```
</div>
  363/1599 ━━━━[37m━━━━━━━━━━━━━━━━  8:00 389ms/step - loss: 1.0570 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  364/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:59 388ms/step - loss: 1.0569 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  365/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:59 388ms/step - loss: 1.0569 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  366/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:58 388ms/step - loss: 1.0568 - mean_absolute_error: 0.8220

<div class="k-default-codeblock">
```

```
</div>
  367/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:58 388ms/step - loss: 1.0567 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  368/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:57 388ms/step - loss: 1.0567 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  369/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:57 388ms/step - loss: 1.0566 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  370/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:56 388ms/step - loss: 1.0565 - mean_absolute_error: 0.8219

<div class="k-default-codeblock">
```

```
</div>
  371/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:56 388ms/step - loss: 1.0565 - mean_absolute_error: 0.8218

<div class="k-default-codeblock">
```

```
</div>
  372/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:55 388ms/step - loss: 1.0564 - mean_absolute_error: 0.8218

<div class="k-default-codeblock">
```

```
</div>
  373/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:55 388ms/step - loss: 1.0563 - mean_absolute_error: 0.8218

<div class="k-default-codeblock">
```

```
</div>
  374/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:55 388ms/step - loss: 1.0563 - mean_absolute_error: 0.8218

<div class="k-default-codeblock">
```

```
</div>
  375/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:54 388ms/step - loss: 1.0562 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  376/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:54 388ms/step - loss: 1.0562 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  377/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:54 388ms/step - loss: 1.0561 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  378/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:54 389ms/step - loss: 1.0560 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  379/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:54 389ms/step - loss: 1.0560 - mean_absolute_error: 0.8217

<div class="k-default-codeblock">
```

```
</div>
  380/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:53 389ms/step - loss: 1.0559 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  381/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:53 389ms/step - loss: 1.0559 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  382/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:53 389ms/step - loss: 1.0558 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  383/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:53 389ms/step - loss: 1.0557 - mean_absolute_error: 0.8216

<div class="k-default-codeblock">
```

```
</div>
  384/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:52 389ms/step - loss: 1.0557 - mean_absolute_error: 0.8215

<div class="k-default-codeblock">
```

```
</div>
  385/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:52 389ms/step - loss: 1.0556 - mean_absolute_error: 0.8215

<div class="k-default-codeblock">
```

```
</div>
  386/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:51 389ms/step - loss: 1.0555 - mean_absolute_error: 0.8215

<div class="k-default-codeblock">
```

```
</div>
  387/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:51 389ms/step - loss: 1.0555 - mean_absolute_error: 0.8215

<div class="k-default-codeblock">
```

```
</div>
  388/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:50 389ms/step - loss: 1.0554 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  389/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:50 389ms/step - loss: 1.0554 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  390/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:50 389ms/step - loss: 1.0553 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  391/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:49 389ms/step - loss: 1.0552 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  392/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:49 389ms/step - loss: 1.0552 - mean_absolute_error: 0.8214

<div class="k-default-codeblock">
```

```
</div>
  393/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:49 389ms/step - loss: 1.0551 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  394/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:48 389ms/step - loss: 1.0551 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  395/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:48 389ms/step - loss: 1.0550 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  396/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:48 389ms/step - loss: 1.0549 - mean_absolute_error: 0.8213

<div class="k-default-codeblock">
```

```
</div>
  397/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:48 390ms/step - loss: 1.0549 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  398/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:48 390ms/step - loss: 1.0548 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  399/1599 ━━━━[37m━━━━━━━━━━━━━━━━  7:47 390ms/step - loss: 1.0548 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  400/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:47 390ms/step - loss: 1.0547 - mean_absolute_error: 0.8212

<div class="k-default-codeblock">
```

```
</div>
  401/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:47 390ms/step - loss: 1.0546 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  402/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:46 390ms/step - loss: 1.0546 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  403/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:46 390ms/step - loss: 1.0545 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  404/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:46 390ms/step - loss: 1.0545 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  405/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:45 390ms/step - loss: 1.0544 - mean_absolute_error: 0.8211

<div class="k-default-codeblock">
```

```
</div>
  406/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:45 390ms/step - loss: 1.0543 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  407/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:45 390ms/step - loss: 1.0543 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  408/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:44 390ms/step - loss: 1.0542 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  409/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:44 390ms/step - loss: 1.0542 - mean_absolute_error: 0.8210

<div class="k-default-codeblock">
```

```
</div>
  410/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:44 390ms/step - loss: 1.0541 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  411/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:43 390ms/step - loss: 1.0540 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  412/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:43 390ms/step - loss: 1.0540 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  413/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:42 390ms/step - loss: 1.0539 - mean_absolute_error: 0.8209

<div class="k-default-codeblock">
```

```
</div>
  414/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:42 390ms/step - loss: 1.0539 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  415/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:41 390ms/step - loss: 1.0538 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  416/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:41 390ms/step - loss: 1.0537 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  417/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:40 390ms/step - loss: 1.0537 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  418/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:40 390ms/step - loss: 1.0536 - mean_absolute_error: 0.8208

<div class="k-default-codeblock">
```

```
</div>
  419/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:39 390ms/step - loss: 1.0536 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  420/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:39 390ms/step - loss: 1.0535 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  421/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:38 390ms/step - loss: 1.0534 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  422/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:38 390ms/step - loss: 1.0534 - mean_absolute_error: 0.8207

<div class="k-default-codeblock">
```

```
</div>
  423/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:38 389ms/step - loss: 1.0533 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  424/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:37 390ms/step - loss: 1.0533 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  425/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:37 390ms/step - loss: 1.0532 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  426/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:37 390ms/step - loss: 1.0531 - mean_absolute_error: 0.8206

<div class="k-default-codeblock">
```

```
</div>
  427/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:36 390ms/step - loss: 1.0531 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  428/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:36 390ms/step - loss: 1.0530 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  429/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:36 390ms/step - loss: 1.0530 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  430/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:36 390ms/step - loss: 1.0529 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  431/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:35 390ms/step - loss: 1.0528 - mean_absolute_error: 0.8205

<div class="k-default-codeblock">
```

```
</div>
  432/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:35 391ms/step - loss: 1.0528 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  433/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:35 391ms/step - loss: 1.0527 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  434/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:35 391ms/step - loss: 1.0527 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  435/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:34 391ms/step - loss: 1.0526 - mean_absolute_error: 0.8204

<div class="k-default-codeblock">
```

```
</div>
  436/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:34 391ms/step - loss: 1.0525 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  437/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:34 391ms/step - loss: 1.0525 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  438/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:33 391ms/step - loss: 1.0524 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  439/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:33 391ms/step - loss: 1.0524 - mean_absolute_error: 0.8203

<div class="k-default-codeblock">
```

```
</div>
  440/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:33 391ms/step - loss: 1.0523 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  441/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:33 391ms/step - loss: 1.0522 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  442/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:32 391ms/step - loss: 1.0522 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  443/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:32 392ms/step - loss: 1.0521 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  444/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:32 392ms/step - loss: 1.0521 - mean_absolute_error: 0.8202

<div class="k-default-codeblock">
```

```
</div>
  445/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:32 392ms/step - loss: 1.0520 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  446/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:31 392ms/step - loss: 1.0520 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  447/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:31 392ms/step - loss: 1.0519 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  448/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:31 392ms/step - loss: 1.0518 - mean_absolute_error: 0.8201

<div class="k-default-codeblock">
```

```
</div>
  449/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:31 392ms/step - loss: 1.0518 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  450/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:30 392ms/step - loss: 1.0517 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  451/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:30 393ms/step - loss: 1.0517 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  452/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:30 393ms/step - loss: 1.0516 - mean_absolute_error: 0.8200

<div class="k-default-codeblock">
```

```
</div>
  453/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:30 393ms/step - loss: 1.0515 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  454/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:29 393ms/step - loss: 1.0515 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  455/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:29 393ms/step - loss: 1.0514 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  456/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:29 393ms/step - loss: 1.0514 - mean_absolute_error: 0.8199

<div class="k-default-codeblock">
```

```
</div>
  457/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:29 393ms/step - loss: 1.0513 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  458/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:28 393ms/step - loss: 1.0512 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  459/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:28 394ms/step - loss: 1.0512 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  460/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:28 394ms/step - loss: 1.0511 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  461/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:28 394ms/step - loss: 1.0511 - mean_absolute_error: 0.8198

<div class="k-default-codeblock">
```

```
</div>
  462/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:27 394ms/step - loss: 1.0510 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  463/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:27 394ms/step - loss: 1.0509 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  464/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:27 394ms/step - loss: 1.0509 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  465/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:26 394ms/step - loss: 1.0508 - mean_absolute_error: 0.8197

<div class="k-default-codeblock">
```

```
</div>
  466/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:26 394ms/step - loss: 1.0508 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  467/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:26 394ms/step - loss: 1.0507 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  468/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:26 395ms/step - loss: 1.0506 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  469/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:25 395ms/step - loss: 1.0506 - mean_absolute_error: 0.8196

<div class="k-default-codeblock">
```

```
</div>
  470/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:25 395ms/step - loss: 1.0505 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  471/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:25 395ms/step - loss: 1.0505 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  472/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:25 395ms/step - loss: 1.0504 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  473/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:24 395ms/step - loss: 1.0504 - mean_absolute_error: 0.8195

<div class="k-default-codeblock">
```

```
</div>
  474/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:24 395ms/step - loss: 1.0503 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  475/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:24 395ms/step - loss: 1.0502 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  476/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:23 395ms/step - loss: 1.0502 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  477/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:23 395ms/step - loss: 1.0501 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  478/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:23 395ms/step - loss: 1.0501 - mean_absolute_error: 0.8194

<div class="k-default-codeblock">
```

```
</div>
  479/1599 ━━━━━[37m━━━━━━━━━━━━━━━  7:22 395ms/step - loss: 1.0500 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  480/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:22 396ms/step - loss: 1.0500 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  481/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:22 396ms/step - loss: 1.0499 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  482/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:21 396ms/step - loss: 1.0498 - mean_absolute_error: 0.8193

<div class="k-default-codeblock">
```

```
</div>
  483/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:21 396ms/step - loss: 1.0498 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  484/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:21 396ms/step - loss: 1.0497 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  485/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:20 396ms/step - loss: 1.0497 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  486/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:20 396ms/step - loss: 1.0496 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  487/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:20 396ms/step - loss: 1.0496 - mean_absolute_error: 0.8192

<div class="k-default-codeblock">
```

```
</div>
  488/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:19 396ms/step - loss: 1.0495 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  489/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:19 396ms/step - loss: 1.0494 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  490/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:19 396ms/step - loss: 1.0494 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  491/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:19 396ms/step - loss: 1.0493 - mean_absolute_error: 0.8191

<div class="k-default-codeblock">
```

```
</div>
  492/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:18 396ms/step - loss: 1.0493 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  493/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:18 396ms/step - loss: 1.0492 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  494/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:18 397ms/step - loss: 1.0492 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  495/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:17 397ms/step - loss: 1.0491 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  496/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:17 397ms/step - loss: 1.0491 - mean_absolute_error: 0.8190

<div class="k-default-codeblock">
```

```
</div>
  497/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:17 397ms/step - loss: 1.0490 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  498/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:16 397ms/step - loss: 1.0489 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  499/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:16 397ms/step - loss: 1.0489 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  500/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:16 397ms/step - loss: 1.0488 - mean_absolute_error: 0.8189

<div class="k-default-codeblock">
```

```
</div>
  501/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:15 397ms/step - loss: 1.0488 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  502/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:15 397ms/step - loss: 1.0487 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  503/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:15 397ms/step - loss: 1.0487 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  504/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:15 397ms/step - loss: 1.0486 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  505/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:14 397ms/step - loss: 1.0486 - mean_absolute_error: 0.8188

<div class="k-default-codeblock">
```

```
</div>
  506/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:14 397ms/step - loss: 1.0485 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  507/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:13 397ms/step - loss: 1.0485 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  508/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:13 397ms/step - loss: 1.0484 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  509/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:12 397ms/step - loss: 1.0483 - mean_absolute_error: 0.8187

<div class="k-default-codeblock">
```

```
</div>
  510/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:12 397ms/step - loss: 1.0483 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  511/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:12 397ms/step - loss: 1.0482 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  512/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:11 397ms/step - loss: 1.0482 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  513/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:11 397ms/step - loss: 1.0481 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  514/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:10 397ms/step - loss: 1.0481 - mean_absolute_error: 0.8186

<div class="k-default-codeblock">
```

```
</div>
  515/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:10 397ms/step - loss: 1.0480 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  516/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:10 397ms/step - loss: 1.0480 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  517/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:09 397ms/step - loss: 1.0479 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  518/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:09 398ms/step - loss: 1.0479 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  519/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:09 398ms/step - loss: 1.0478 - mean_absolute_error: 0.8185

<div class="k-default-codeblock">
```

```
</div>
  520/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:09 398ms/step - loss: 1.0478 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  521/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:09 398ms/step - loss: 1.0477 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  522/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:08 398ms/step - loss: 1.0477 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  523/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:08 398ms/step - loss: 1.0476 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  524/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:08 398ms/step - loss: 1.0476 - mean_absolute_error: 0.8184

<div class="k-default-codeblock">
```

```
</div>
  525/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:07 398ms/step - loss: 1.0475 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  526/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:07 398ms/step - loss: 1.0475 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  527/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:07 398ms/step - loss: 1.0474 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  528/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:06 399ms/step - loss: 1.0474 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  529/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:06 399ms/step - loss: 1.0473 - mean_absolute_error: 0.8183

<div class="k-default-codeblock">
```

```
</div>
  530/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:06 399ms/step - loss: 1.0473 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  531/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:05 399ms/step - loss: 1.0472 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  532/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:05 399ms/step - loss: 1.0472 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  533/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:05 399ms/step - loss: 1.0471 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  534/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:04 399ms/step - loss: 1.0471 - mean_absolute_error: 0.8182

<div class="k-default-codeblock">
```

```
</div>
  535/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:04 399ms/step - loss: 1.0470 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  536/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:04 399ms/step - loss: 1.0470 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  537/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:03 399ms/step - loss: 1.0470 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  538/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:03 399ms/step - loss: 1.0469 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  539/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:03 399ms/step - loss: 1.0469 - mean_absolute_error: 0.8181

<div class="k-default-codeblock">
```

```
</div>
  540/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:02 399ms/step - loss: 1.0468 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  541/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:02 400ms/step - loss: 1.0468 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  542/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:02 400ms/step - loss: 1.0467 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  543/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:02 400ms/step - loss: 1.0467 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  544/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:01 400ms/step - loss: 1.0466 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  545/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:01 400ms/step - loss: 1.0466 - mean_absolute_error: 0.8180

<div class="k-default-codeblock">
```

```
</div>
  546/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:01 400ms/step - loss: 1.0465 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  547/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:00 400ms/step - loss: 1.0465 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  548/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:00 400ms/step - loss: 1.0465 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  549/1599 ━━━━━━[37m━━━━━━━━━━━━━━  7:00 400ms/step - loss: 1.0464 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  550/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:59 400ms/step - loss: 1.0464 - mean_absolute_error: 0.8179

<div class="k-default-codeblock">
```

```
</div>
  551/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:59 400ms/step - loss: 1.0463 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  552/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:59 400ms/step - loss: 1.0463 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  553/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:58 400ms/step - loss: 1.0462 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  554/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:58 400ms/step - loss: 1.0462 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  555/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:57 400ms/step - loss: 1.0461 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  556/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:57 400ms/step - loss: 1.0461 - mean_absolute_error: 0.8178

<div class="k-default-codeblock">
```

```
</div>
  557/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:57 400ms/step - loss: 1.0461 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  558/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:56 400ms/step - loss: 1.0460 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  559/1599 ━━━━━━[37m━━━━━━━━━━━━━━  6:56 400ms/step - loss: 1.0460 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  560/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:55 400ms/step - loss: 1.0459 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  561/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:55 400ms/step - loss: 1.0459 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  562/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:55 400ms/step - loss: 1.0458 - mean_absolute_error: 0.8177

<div class="k-default-codeblock">
```

```
</div>
  563/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:54 400ms/step - loss: 1.0458 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  564/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:54 400ms/step - loss: 1.0458 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  565/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:54 401ms/step - loss: 1.0457 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  566/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:54 401ms/step - loss: 1.0457 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  567/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:53 401ms/step - loss: 1.0456 - mean_absolute_error: 0.8176

<div class="k-default-codeblock">
```

```
</div>
  568/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:53 401ms/step - loss: 1.0456 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  569/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:53 401ms/step - loss: 1.0455 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  570/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:52 401ms/step - loss: 1.0455 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  571/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:52 401ms/step - loss: 1.0455 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  572/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:51 401ms/step - loss: 1.0454 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  573/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:51 401ms/step - loss: 1.0454 - mean_absolute_error: 0.8175

<div class="k-default-codeblock">
```

```
</div>
  574/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:51 401ms/step - loss: 1.0453 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  575/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:50 401ms/step - loss: 1.0453 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  576/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:50 401ms/step - loss: 1.0452 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  577/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:50 401ms/step - loss: 1.0452 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  578/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:49 401ms/step - loss: 1.0451 - mean_absolute_error: 0.8174

<div class="k-default-codeblock">
```

```
</div>
  579/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:49 402ms/step - loss: 1.0451 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  580/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:49 402ms/step - loss: 1.0451 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  581/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:48 402ms/step - loss: 1.0450 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  582/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:48 401ms/step - loss: 1.0450 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  583/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:47 401ms/step - loss: 1.0449 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  584/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:47 401ms/step - loss: 1.0449 - mean_absolute_error: 0.8173

<div class="k-default-codeblock">
```

```
</div>
  585/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:46 401ms/step - loss: 1.0448 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  586/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:46 401ms/step - loss: 1.0448 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  587/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:46 401ms/step - loss: 1.0447 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  588/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:45 401ms/step - loss: 1.0447 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  589/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:45 401ms/step - loss: 1.0447 - mean_absolute_error: 0.8172

<div class="k-default-codeblock">
```

```
</div>
  590/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:44 401ms/step - loss: 1.0446 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  591/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:44 401ms/step - loss: 1.0446 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  592/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:43 401ms/step - loss: 1.0445 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  593/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:43 401ms/step - loss: 1.0445 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  594/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:43 402ms/step - loss: 1.0444 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  595/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:43 402ms/step - loss: 1.0444 - mean_absolute_error: 0.8171

<div class="k-default-codeblock">
```

```
</div>
  596/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:42 402ms/step - loss: 1.0444 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  597/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:42 402ms/step - loss: 1.0443 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  598/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:42 402ms/step - loss: 1.0443 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  599/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:41 402ms/step - loss: 1.0442 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  600/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:41 402ms/step - loss: 1.0442 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  601/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:41 402ms/step - loss: 1.0442 - mean_absolute_error: 0.8170

<div class="k-default-codeblock">
```

```
</div>
  602/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:40 402ms/step - loss: 1.0441 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  603/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:40 402ms/step - loss: 1.0441 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  604/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:40 402ms/step - loss: 1.0440 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  605/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:39 402ms/step - loss: 1.0440 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  606/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:39 402ms/step - loss: 1.0439 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  607/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:39 403ms/step - loss: 1.0439 - mean_absolute_error: 0.8169

<div class="k-default-codeblock">
```

```
</div>
  608/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:38 403ms/step - loss: 1.0439 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  609/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:38 403ms/step - loss: 1.0438 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  610/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:38 403ms/step - loss: 1.0438 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  611/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:38 403ms/step - loss: 1.0437 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  612/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:37 403ms/step - loss: 1.0437 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  613/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:37 403ms/step - loss: 1.0437 - mean_absolute_error: 0.8168

<div class="k-default-codeblock">
```

```
</div>
  614/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:37 403ms/step - loss: 1.0436 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  615/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:36 403ms/step - loss: 1.0436 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  616/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:36 403ms/step - loss: 1.0435 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  617/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:36 403ms/step - loss: 1.0435 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  618/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:35 403ms/step - loss: 1.0435 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  619/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:35 403ms/step - loss: 1.0434 - mean_absolute_error: 0.8167

<div class="k-default-codeblock">
```

```
</div>
  620/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:35 404ms/step - loss: 1.0434 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  621/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:34 404ms/step - loss: 1.0433 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  622/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:34 404ms/step - loss: 1.0433 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  623/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:33 404ms/step - loss: 1.0433 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  624/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:33 404ms/step - loss: 1.0432 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  625/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:33 404ms/step - loss: 1.0432 - mean_absolute_error: 0.8166

<div class="k-default-codeblock">
```

```
</div>
  626/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:32 404ms/step - loss: 1.0431 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  627/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:32 404ms/step - loss: 1.0431 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  628/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:32 404ms/step - loss: 1.0431 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  629/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:31 404ms/step - loss: 1.0430 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  630/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:31 404ms/step - loss: 1.0430 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  631/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:31 404ms/step - loss: 1.0430 - mean_absolute_error: 0.8165

<div class="k-default-codeblock">
```

```
</div>
  632/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:30 404ms/step - loss: 1.0429 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  633/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:30 404ms/step - loss: 1.0429 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  634/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:29 404ms/step - loss: 1.0428 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  635/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:29 404ms/step - loss: 1.0428 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  636/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:28 404ms/step - loss: 1.0428 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  637/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:28 404ms/step - loss: 1.0427 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  638/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:27 404ms/step - loss: 1.0427 - mean_absolute_error: 0.8164

<div class="k-default-codeblock">
```

```
</div>
  639/1599 ━━━━━━━[37m━━━━━━━━━━━━━  6:27 404ms/step - loss: 1.0426 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  640/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:27 404ms/step - loss: 1.0426 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  641/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:26 404ms/step - loss: 1.0426 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  642/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:26 404ms/step - loss: 1.0425 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  643/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:25 404ms/step - loss: 1.0425 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  644/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:25 404ms/step - loss: 1.0425 - mean_absolute_error: 0.8163

<div class="k-default-codeblock">
```

```
</div>
  645/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:25 404ms/step - loss: 1.0424 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  646/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:25 404ms/step - loss: 1.0424 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  647/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:25 404ms/step - loss: 1.0424 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  648/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:24 405ms/step - loss: 1.0423 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  649/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:24 405ms/step - loss: 1.0423 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  650/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:24 405ms/step - loss: 1.0422 - mean_absolute_error: 0.8162

<div class="k-default-codeblock">
```

```
</div>
  651/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:23 405ms/step - loss: 1.0422 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  652/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:23 405ms/step - loss: 1.0422 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  653/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:22 405ms/step - loss: 1.0421 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  654/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:22 405ms/step - loss: 1.0421 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  655/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:22 405ms/step - loss: 1.0421 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  656/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:21 405ms/step - loss: 1.0420 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  657/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:21 405ms/step - loss: 1.0420 - mean_absolute_error: 0.8161

<div class="k-default-codeblock">
```

```
</div>
  658/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:21 405ms/step - loss: 1.0420 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  659/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:20 405ms/step - loss: 1.0419 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  660/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:20 405ms/step - loss: 1.0419 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  661/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:20 405ms/step - loss: 1.0419 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  662/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:19 405ms/step - loss: 1.0418 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  663/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:19 406ms/step - loss: 1.0418 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  664/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:19 406ms/step - loss: 1.0417 - mean_absolute_error: 0.8160

<div class="k-default-codeblock">
```

```
</div>
  665/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:18 406ms/step - loss: 1.0417 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  666/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:18 406ms/step - loss: 1.0417 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  667/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:18 406ms/step - loss: 1.0416 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  668/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:18 406ms/step - loss: 1.0416 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  669/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:17 406ms/step - loss: 1.0416 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  670/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:17 407ms/step - loss: 1.0415 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  671/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:17 407ms/step - loss: 1.0415 - mean_absolute_error: 0.8159

<div class="k-default-codeblock">
```

```
</div>
  672/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:17 407ms/step - loss: 1.0415 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  673/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:16 407ms/step - loss: 1.0414 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  674/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:16 407ms/step - loss: 1.0414 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  675/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:15 407ms/step - loss: 1.0414 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  676/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:15 407ms/step - loss: 1.0413 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  677/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:15 407ms/step - loss: 1.0413 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  678/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:14 407ms/step - loss: 1.0413 - mean_absolute_error: 0.8158

<div class="k-default-codeblock">
```

```
</div>
  679/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:14 407ms/step - loss: 1.0412 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  680/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:13 407ms/step - loss: 1.0412 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  681/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:13 407ms/step - loss: 1.0412 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  682/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:12 407ms/step - loss: 1.0411 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  683/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:12 407ms/step - loss: 1.0411 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  684/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:12 407ms/step - loss: 1.0411 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  685/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:11 407ms/step - loss: 1.0410 - mean_absolute_error: 0.8157

<div class="k-default-codeblock">
```

```
</div>
  686/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:11 407ms/step - loss: 1.0410 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  687/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:10 407ms/step - loss: 1.0410 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  688/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:10 407ms/step - loss: 1.0409 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  689/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:10 407ms/step - loss: 1.0409 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  690/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:09 407ms/step - loss: 1.0409 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  691/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:09 407ms/step - loss: 1.0408 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  692/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:09 407ms/step - loss: 1.0408 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  693/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:09 407ms/step - loss: 1.0408 - mean_absolute_error: 0.8156

<div class="k-default-codeblock">
```

```
</div>
  694/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:08 408ms/step - loss: 1.0407 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  695/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:08 408ms/step - loss: 1.0407 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  696/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:08 408ms/step - loss: 1.0407 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  697/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:07 408ms/step - loss: 1.0406 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  698/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:07 408ms/step - loss: 1.0406 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  699/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:06 408ms/step - loss: 1.0406 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  700/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:06 408ms/step - loss: 1.0405 - mean_absolute_error: 0.8155

<div class="k-default-codeblock">
```

```
</div>
  701/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:06 408ms/step - loss: 1.0405 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  702/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:05 408ms/step - loss: 1.0405 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  703/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:05 408ms/step - loss: 1.0405 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  704/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:05 408ms/step - loss: 1.0404 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  705/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:04 408ms/step - loss: 1.0404 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  706/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:04 408ms/step - loss: 1.0404 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  707/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:04 408ms/step - loss: 1.0403 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  708/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:03 409ms/step - loss: 1.0403 - mean_absolute_error: 0.8154

<div class="k-default-codeblock">
```

```
</div>
  709/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:03 409ms/step - loss: 1.0403 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  710/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:03 409ms/step - loss: 1.0402 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  711/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:03 409ms/step - loss: 1.0402 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  712/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:02 409ms/step - loss: 1.0402 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  713/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:02 409ms/step - loss: 1.0401 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  714/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:02 409ms/step - loss: 1.0401 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  715/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:01 409ms/step - loss: 1.0401 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  716/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:01 409ms/step - loss: 1.0401 - mean_absolute_error: 0.8153

<div class="k-default-codeblock">
```

```
</div>
  717/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:00 409ms/step - loss: 1.0400 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  718/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:00 409ms/step - loss: 1.0400 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  719/1599 ━━━━━━━━[37m━━━━━━━━━━━━  6:00 409ms/step - loss: 1.0400 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  720/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:59 409ms/step - loss: 1.0399 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  721/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:59 409ms/step - loss: 1.0399 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  722/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:58 409ms/step - loss: 1.0399 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  723/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:58 409ms/step - loss: 1.0399 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  724/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:58 409ms/step - loss: 1.0398 - mean_absolute_error: 0.8152

<div class="k-default-codeblock">
```

```
</div>
  725/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:57 410ms/step - loss: 1.0398 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  726/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:57 410ms/step - loss: 1.0398 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  727/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:57 410ms/step - loss: 1.0397 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  728/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:56 410ms/step - loss: 1.0397 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  729/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:56 410ms/step - loss: 1.0397 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  730/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:56 410ms/step - loss: 1.0397 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  731/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:55 410ms/step - loss: 1.0396 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  732/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:55 410ms/step - loss: 1.0396 - mean_absolute_error: 0.8151

<div class="k-default-codeblock">
```

```
</div>
  733/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:54 410ms/step - loss: 1.0396 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  734/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:54 410ms/step - loss: 1.0396 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  735/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:54 410ms/step - loss: 1.0395 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  736/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:53 410ms/step - loss: 1.0395 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  737/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:53 410ms/step - loss: 1.0395 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  738/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:53 410ms/step - loss: 1.0394 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  739/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:52 410ms/step - loss: 1.0394 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  740/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:52 410ms/step - loss: 1.0394 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  741/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:51 410ms/step - loss: 1.0394 - mean_absolute_error: 0.8150

<div class="k-default-codeblock">
```

```
</div>
  742/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:51 410ms/step - loss: 1.0393 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  743/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:51 410ms/step - loss: 1.0393 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  744/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:50 410ms/step - loss: 1.0393 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  745/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:50 410ms/step - loss: 1.0393 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  746/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:49 410ms/step - loss: 1.0392 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  747/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:49 410ms/step - loss: 1.0392 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  748/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:49 410ms/step - loss: 1.0392 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  749/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:48 410ms/step - loss: 1.0392 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  750/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:48 410ms/step - loss: 1.0391 - mean_absolute_error: 0.8149

<div class="k-default-codeblock">
```

```
</div>
  751/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:48 411ms/step - loss: 1.0391 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  752/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:47 411ms/step - loss: 1.0391 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  753/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:47 411ms/step - loss: 1.0391 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  754/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:46 411ms/step - loss: 1.0390 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  755/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:46 411ms/step - loss: 1.0390 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  756/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:46 411ms/step - loss: 1.0390 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  757/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:45 411ms/step - loss: 1.0390 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  758/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:45 411ms/step - loss: 1.0389 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  759/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:45 411ms/step - loss: 1.0389 - mean_absolute_error: 0.8148

<div class="k-default-codeblock">
```

```
</div>
  760/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:44 411ms/step - loss: 1.0389 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  761/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:44 411ms/step - loss: 1.0389 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  762/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:44 411ms/step - loss: 1.0388 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  763/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:43 411ms/step - loss: 1.0388 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  764/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:43 411ms/step - loss: 1.0388 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  765/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:43 411ms/step - loss: 1.0388 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  766/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:42 411ms/step - loss: 1.0387 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  767/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:42 411ms/step - loss: 1.0387 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  768/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:41 411ms/step - loss: 1.0387 - mean_absolute_error: 0.8147

<div class="k-default-codeblock">
```

```
</div>
  769/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:41 411ms/step - loss: 1.0387 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  770/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:41 411ms/step - loss: 1.0386 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  771/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:40 411ms/step - loss: 1.0386 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  772/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:40 411ms/step - loss: 1.0386 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  773/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:39 411ms/step - loss: 1.0386 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  774/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:39 411ms/step - loss: 1.0385 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  775/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:38 411ms/step - loss: 1.0385 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  776/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:38 411ms/step - loss: 1.0385 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  777/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:37 411ms/step - loss: 1.0385 - mean_absolute_error: 0.8146

<div class="k-default-codeblock">
```

```
</div>
  778/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:37 411ms/step - loss: 1.0384 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  779/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:37 411ms/step - loss: 1.0384 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  780/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:36 411ms/step - loss: 1.0384 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  781/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:36 411ms/step - loss: 1.0384 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  782/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:35 411ms/step - loss: 1.0383 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  783/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:35 411ms/step - loss: 1.0383 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  784/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:34 411ms/step - loss: 1.0383 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  785/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:34 411ms/step - loss: 1.0383 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  786/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:34 411ms/step - loss: 1.0382 - mean_absolute_error: 0.8145

<div class="k-default-codeblock">
```

```
</div>
  787/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:33 411ms/step - loss: 1.0382 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  788/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:33 411ms/step - loss: 1.0382 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  789/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:33 411ms/step - loss: 1.0382 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  790/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:32 411ms/step - loss: 1.0381 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  791/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:32 411ms/step - loss: 1.0381 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  792/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:32 411ms/step - loss: 1.0381 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  793/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:31 411ms/step - loss: 1.0381 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  794/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:31 411ms/step - loss: 1.0380 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  795/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:30 412ms/step - loss: 1.0380 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  796/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:30 412ms/step - loss: 1.0380 - mean_absolute_error: 0.8144

<div class="k-default-codeblock">
```

```
</div>
  797/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:30 412ms/step - loss: 1.0380 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  798/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:29 412ms/step - loss: 1.0380 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  799/1599 ━━━━━━━━━[37m━━━━━━━━━━━  5:29 412ms/step - loss: 1.0379 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  800/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:28 412ms/step - loss: 1.0379 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  801/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:28 412ms/step - loss: 1.0379 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  802/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:28 412ms/step - loss: 1.0379 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  803/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:27 412ms/step - loss: 1.0378 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  804/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:27 412ms/step - loss: 1.0378 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  805/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:26 412ms/step - loss: 1.0378 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  806/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:26 412ms/step - loss: 1.0378 - mean_absolute_error: 0.8143

<div class="k-default-codeblock">
```

```
</div>
  807/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:26 412ms/step - loss: 1.0377 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  808/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:25 412ms/step - loss: 1.0377 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  809/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:25 412ms/step - loss: 1.0377 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  810/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:25 412ms/step - loss: 1.0377 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  811/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:25 413ms/step - loss: 1.0376 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  812/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:24 413ms/step - loss: 1.0376 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  813/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:24 413ms/step - loss: 1.0376 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  814/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:24 413ms/step - loss: 1.0376 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  815/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:23 413ms/step - loss: 1.0376 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  816/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:23 413ms/step - loss: 1.0375 - mean_absolute_error: 0.8142

<div class="k-default-codeblock">
```

```
</div>
  817/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:23 413ms/step - loss: 1.0375 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  818/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:22 413ms/step - loss: 1.0375 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  819/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:22 413ms/step - loss: 1.0375 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  820/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:21 413ms/step - loss: 1.0374 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  821/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:21 413ms/step - loss: 1.0374 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  822/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:21 413ms/step - loss: 1.0374 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  823/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:20 413ms/step - loss: 1.0374 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  824/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:20 413ms/step - loss: 1.0374 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  825/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:19 413ms/step - loss: 1.0373 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  826/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:19 413ms/step - loss: 1.0373 - mean_absolute_error: 0.8141

<div class="k-default-codeblock">
```

```
</div>
  827/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:19 413ms/step - loss: 1.0373 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  828/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:18 413ms/step - loss: 1.0373 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  829/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:18 413ms/step - loss: 1.0372 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  830/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:17 413ms/step - loss: 1.0372 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  831/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:17 414ms/step - loss: 1.0372 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  832/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:17 414ms/step - loss: 1.0372 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  833/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:16 414ms/step - loss: 1.0372 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  834/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:16 414ms/step - loss: 1.0371 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  835/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:16 414ms/step - loss: 1.0371 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  836/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:15 414ms/step - loss: 1.0371 - mean_absolute_error: 0.8140

<div class="k-default-codeblock">
```

```
</div>
  837/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:15 414ms/step - loss: 1.0371 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  838/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:14 414ms/step - loss: 1.0371 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  839/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:14 414ms/step - loss: 1.0370 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  840/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:14 414ms/step - loss: 1.0370 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  841/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:13 414ms/step - loss: 1.0370 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  842/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:13 414ms/step - loss: 1.0370 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  843/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:12 414ms/step - loss: 1.0369 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  844/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:12 414ms/step - loss: 1.0369 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  845/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:12 414ms/step - loss: 1.0369 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  846/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:11 414ms/step - loss: 1.0369 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  847/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:11 414ms/step - loss: 1.0369 - mean_absolute_error: 0.8139

<div class="k-default-codeblock">
```

```
</div>
  848/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:11 414ms/step - loss: 1.0368 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  849/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:10 414ms/step - loss: 1.0368 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  850/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:10 414ms/step - loss: 1.0368 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  851/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:10 415ms/step - loss: 1.0368 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  852/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:09 415ms/step - loss: 1.0367 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  853/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:09 415ms/step - loss: 1.0367 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  854/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:09 415ms/step - loss: 1.0367 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  855/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:08 415ms/step - loss: 1.0367 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  856/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:08 415ms/step - loss: 1.0367 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  857/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:08 415ms/step - loss: 1.0366 - mean_absolute_error: 0.8138

<div class="k-default-codeblock">
```

```
</div>
  858/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:07 415ms/step - loss: 1.0366 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  859/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:07 415ms/step - loss: 1.0366 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  860/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:06 415ms/step - loss: 1.0366 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  861/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:06 415ms/step - loss: 1.0365 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  862/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:06 415ms/step - loss: 1.0365 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  863/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:05 415ms/step - loss: 1.0365 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  864/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:05 415ms/step - loss: 1.0365 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  865/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:04 415ms/step - loss: 1.0365 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  866/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:04 415ms/step - loss: 1.0364 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  867/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:04 416ms/step - loss: 1.0364 - mean_absolute_error: 0.8137

<div class="k-default-codeblock">
```

```
</div>
  868/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:03 416ms/step - loss: 1.0364 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  869/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:03 416ms/step - loss: 1.0364 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  870/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:03 416ms/step - loss: 1.0364 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  871/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:02 416ms/step - loss: 1.0363 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  872/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:02 416ms/step - loss: 1.0363 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  873/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:02 416ms/step - loss: 1.0363 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  874/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:02 417ms/step - loss: 1.0363 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  875/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:01 417ms/step - loss: 1.0362 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  876/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:01 417ms/step - loss: 1.0362 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  877/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:01 417ms/step - loss: 1.0362 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  878/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:00 417ms/step - loss: 1.0362 - mean_absolute_error: 0.8136

<div class="k-default-codeblock">
```

```
</div>
  879/1599 ━━━━━━━━━━[37m━━━━━━━━━━  5:00 417ms/step - loss: 1.0362 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  880/1599 ━━━━━━━━━━━[37m━━━━━━━━━  5:00 418ms/step - loss: 1.0361 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  881/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:59 418ms/step - loss: 1.0361 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  882/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:59 418ms/step - loss: 1.0361 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  883/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:59 418ms/step - loss: 1.0361 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  884/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:58 418ms/step - loss: 1.0361 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  885/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:58 418ms/step - loss: 1.0360 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  886/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:58 418ms/step - loss: 1.0360 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  887/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:57 418ms/step - loss: 1.0360 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  888/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:57 419ms/step - loss: 1.0360 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  889/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:57 419ms/step - loss: 1.0360 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  890/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:56 419ms/step - loss: 1.0359 - mean_absolute_error: 0.8135

<div class="k-default-codeblock">
```

```
</div>
  891/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:56 419ms/step - loss: 1.0359 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  892/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:56 419ms/step - loss: 1.0359 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  893/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:56 419ms/step - loss: 1.0359 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  894/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:55 419ms/step - loss: 1.0359 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  895/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:55 420ms/step - loss: 1.0358 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  896/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:55 420ms/step - loss: 1.0358 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  897/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:54 420ms/step - loss: 1.0358 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  898/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:54 420ms/step - loss: 1.0358 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  899/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:54 420ms/step - loss: 1.0358 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  900/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:53 420ms/step - loss: 1.0358 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  901/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:53 420ms/step - loss: 1.0357 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  902/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:53 420ms/step - loss: 1.0357 - mean_absolute_error: 0.8134

<div class="k-default-codeblock">
```

```
</div>
  903/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:52 421ms/step - loss: 1.0357 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  904/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:52 421ms/step - loss: 1.0357 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  905/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:52 421ms/step - loss: 1.0357 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  906/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:51 421ms/step - loss: 1.0356 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  907/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:51 421ms/step - loss: 1.0356 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  908/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:51 421ms/step - loss: 1.0356 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  909/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:50 422ms/step - loss: 1.0356 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  910/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:50 422ms/step - loss: 1.0356 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  911/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:50 422ms/step - loss: 1.0355 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  912/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:49 422ms/step - loss: 1.0355 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  913/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:49 422ms/step - loss: 1.0355 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  914/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:49 422ms/step - loss: 1.0355 - mean_absolute_error: 0.8133

<div class="k-default-codeblock">
```

```
</div>
  915/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:48 422ms/step - loss: 1.0355 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  916/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:48 422ms/step - loss: 1.0354 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  917/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:48 423ms/step - loss: 1.0354 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  918/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:47 423ms/step - loss: 1.0354 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  919/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:47 423ms/step - loss: 1.0354 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  920/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:47 423ms/step - loss: 1.0354 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  921/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:46 423ms/step - loss: 1.0353 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  922/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:46 423ms/step - loss: 1.0353 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  923/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:46 423ms/step - loss: 1.0353 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  924/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:45 423ms/step - loss: 1.0353 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  925/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:45 424ms/step - loss: 1.0353 - mean_absolute_error: 0.8132

<div class="k-default-codeblock">
```

```
</div>
  926/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:45 424ms/step - loss: 1.0352 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  927/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:44 424ms/step - loss: 1.0352 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  928/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:44 424ms/step - loss: 1.0352 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  929/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:44 424ms/step - loss: 1.0352 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  930/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:43 424ms/step - loss: 1.0352 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  931/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:43 424ms/step - loss: 1.0351 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  932/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:43 424ms/step - loss: 1.0351 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  933/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:42 425ms/step - loss: 1.0351 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  934/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:42 425ms/step - loss: 1.0351 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  935/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:42 425ms/step - loss: 1.0351 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  936/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:41 425ms/step - loss: 1.0350 - mean_absolute_error: 0.8131

<div class="k-default-codeblock">
```

```
</div>
  937/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:41 425ms/step - loss: 1.0350 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  938/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:41 425ms/step - loss: 1.0350 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  939/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:40 425ms/step - loss: 1.0350 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  940/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:40 426ms/step - loss: 1.0350 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  941/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:40 426ms/step - loss: 1.0349 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  942/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:39 426ms/step - loss: 1.0349 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  943/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:39 426ms/step - loss: 1.0349 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  944/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:38 426ms/step - loss: 1.0349 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  945/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:38 426ms/step - loss: 1.0348 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  946/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:38 426ms/step - loss: 1.0348 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  947/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:37 426ms/step - loss: 1.0348 - mean_absolute_error: 0.8130

<div class="k-default-codeblock">
```

```
</div>
  948/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:37 426ms/step - loss: 1.0348 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  949/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:36 426ms/step - loss: 1.0348 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  950/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:36 426ms/step - loss: 1.0347 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  951/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:35 426ms/step - loss: 1.0347 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  952/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:35 426ms/step - loss: 1.0347 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  953/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:35 426ms/step - loss: 1.0347 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  954/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:34 426ms/step - loss: 1.0346 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  955/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:34 426ms/step - loss: 1.0346 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  956/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:34 426ms/step - loss: 1.0346 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  957/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:33 426ms/step - loss: 1.0346 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  958/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:33 426ms/step - loss: 1.0346 - mean_absolute_error: 0.8129

<div class="k-default-codeblock">
```

```
</div>
  959/1599 ━━━━━━━━━━━[37m━━━━━━━━━  4:32 427ms/step - loss: 1.0345 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  960/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:32 427ms/step - loss: 1.0345 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  961/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:32 427ms/step - loss: 1.0345 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  962/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:31 427ms/step - loss: 1.0345 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  963/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:31 427ms/step - loss: 1.0345 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  964/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:30 427ms/step - loss: 1.0344 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  965/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:30 427ms/step - loss: 1.0344 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  966/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:30 427ms/step - loss: 1.0344 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  967/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:29 427ms/step - loss: 1.0344 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  968/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:29 427ms/step - loss: 1.0343 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  969/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:28 427ms/step - loss: 1.0343 - mean_absolute_error: 0.8128

<div class="k-default-codeblock">
```

```
</div>
  970/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:28 427ms/step - loss: 1.0343 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  971/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:27 427ms/step - loss: 1.0343 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  972/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:27 427ms/step - loss: 1.0343 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  973/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:27 427ms/step - loss: 1.0342 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  974/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:26 427ms/step - loss: 1.0342 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  975/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:26 427ms/step - loss: 1.0342 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  976/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:26 427ms/step - loss: 1.0342 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  977/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:25 427ms/step - loss: 1.0341 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  978/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:25 427ms/step - loss: 1.0341 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  979/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:24 427ms/step - loss: 1.0341 - mean_absolute_error: 0.8127

<div class="k-default-codeblock">
```

```
</div>
  980/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:24 427ms/step - loss: 1.0341 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  981/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:24 428ms/step - loss: 1.0341 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  982/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:23 428ms/step - loss: 1.0340 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  983/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:23 428ms/step - loss: 1.0340 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  984/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:22 428ms/step - loss: 1.0340 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  985/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:22 428ms/step - loss: 1.0340 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  986/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:22 427ms/step - loss: 1.0339 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  987/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:21 428ms/step - loss: 1.0339 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  988/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:21 428ms/step - loss: 1.0339 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  989/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:20 427ms/step - loss: 1.0339 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  990/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:20 428ms/step - loss: 1.0339 - mean_absolute_error: 0.8126

<div class="k-default-codeblock">
```

```
</div>
  991/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:19 428ms/step - loss: 1.0338 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  992/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:19 428ms/step - loss: 1.0338 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  993/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:19 428ms/step - loss: 1.0338 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  994/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:18 428ms/step - loss: 1.0338 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  995/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:18 428ms/step - loss: 1.0338 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  996/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:18 428ms/step - loss: 1.0337 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  997/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:17 428ms/step - loss: 1.0337 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  998/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:17 428ms/step - loss: 1.0337 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
  999/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:16 428ms/step - loss: 1.0337 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1000/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:16 428ms/step - loss: 1.0336 - mean_absolute_error: 0.8125

<div class="k-default-codeblock">
```

```
</div>
 1001/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:16 428ms/step - loss: 1.0336 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1002/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:15 428ms/step - loss: 1.0336 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1003/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:15 429ms/step - loss: 1.0336 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1004/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:14 428ms/step - loss: 1.0336 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1005/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:14 429ms/step - loss: 1.0335 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1006/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:14 428ms/step - loss: 1.0335 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1007/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:13 428ms/step - loss: 1.0335 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1008/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:13 428ms/step - loss: 1.0335 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1009/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:12 428ms/step - loss: 1.0335 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1010/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:12 428ms/step - loss: 1.0334 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1011/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:11 428ms/step - loss: 1.0334 - mean_absolute_error: 0.8124

<div class="k-default-codeblock">
```

```
</div>
 1012/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:11 428ms/step - loss: 1.0334 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1013/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:10 428ms/step - loss: 1.0334 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1014/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:10 428ms/step - loss: 1.0333 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1015/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:09 428ms/step - loss: 1.0333 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1016/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:09 428ms/step - loss: 1.0333 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1017/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:09 428ms/step - loss: 1.0333 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1018/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:08 428ms/step - loss: 1.0333 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1019/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:08 428ms/step - loss: 1.0332 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1020/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:07 428ms/step - loss: 1.0332 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1021/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:07 428ms/step - loss: 1.0332 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1022/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:07 428ms/step - loss: 1.0332 - mean_absolute_error: 0.8123

<div class="k-default-codeblock">
```

```
</div>
 1023/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:06 428ms/step - loss: 1.0332 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1024/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:06 428ms/step - loss: 1.0331 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1025/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:05 429ms/step - loss: 1.0331 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1026/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:05 429ms/step - loss: 1.0331 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1027/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:05 429ms/step - loss: 1.0331 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1028/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:04 428ms/step - loss: 1.0330 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1029/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:04 428ms/step - loss: 1.0330 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1030/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:03 428ms/step - loss: 1.0330 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1031/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:03 428ms/step - loss: 1.0330 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1032/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:02 428ms/step - loss: 1.0330 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1033/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:02 428ms/step - loss: 1.0329 - mean_absolute_error: 0.8122

<div class="k-default-codeblock">
```

```
</div>
 1034/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:02 428ms/step - loss: 1.0329 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1035/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:01 428ms/step - loss: 1.0329 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1036/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:01 428ms/step - loss: 1.0329 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1037/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:00 429ms/step - loss: 1.0328 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1038/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:00 429ms/step - loss: 1.0328 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1039/1599 ━━━━━━━━━━━━[37m━━━━━━━━  4:00 429ms/step - loss: 1.0328 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1040/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:59 429ms/step - loss: 1.0328 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1041/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:59 429ms/step - loss: 1.0328 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1042/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:58 429ms/step - loss: 1.0327 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1043/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:58 429ms/step - loss: 1.0327 - mean_absolute_error: 0.8121

<div class="k-default-codeblock">
```

```
</div>
 1044/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:58 429ms/step - loss: 1.0327 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1045/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:57 429ms/step - loss: 1.0327 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1046/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:57 429ms/step - loss: 1.0326 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1047/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:57 430ms/step - loss: 1.0326 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1048/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:56 430ms/step - loss: 1.0326 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1049/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:56 430ms/step - loss: 1.0326 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1050/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:55 430ms/step - loss: 1.0326 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1051/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:55 430ms/step - loss: 1.0325 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1052/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:55 430ms/step - loss: 1.0325 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1053/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:54 430ms/step - loss: 1.0325 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1054/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:54 430ms/step - loss: 1.0325 - mean_absolute_error: 0.8120

<div class="k-default-codeblock">
```

```
</div>
 1055/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:53 430ms/step - loss: 1.0324 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1056/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:53 430ms/step - loss: 1.0324 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1057/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:52 430ms/step - loss: 1.0324 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1058/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:52 430ms/step - loss: 1.0324 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1059/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:52 430ms/step - loss: 1.0324 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1060/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:51 430ms/step - loss: 1.0323 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1061/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:51 430ms/step - loss: 1.0323 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1062/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:50 430ms/step - loss: 1.0323 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1063/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:50 430ms/step - loss: 1.0323 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1064/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:50 430ms/step - loss: 1.0323 - mean_absolute_error: 0.8119

<div class="k-default-codeblock">
```

```
</div>
 1065/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:49 430ms/step - loss: 1.0322 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1066/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:49 430ms/step - loss: 1.0322 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1067/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:48 430ms/step - loss: 1.0322 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1068/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:48 430ms/step - loss: 1.0322 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1069/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:48 430ms/step - loss: 1.0321 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1070/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:47 430ms/step - loss: 1.0321 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1071/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:47 431ms/step - loss: 1.0321 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1072/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:46 431ms/step - loss: 1.0321 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1073/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:46 431ms/step - loss: 1.0321 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1074/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:46 431ms/step - loss: 1.0320 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1075/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:45 431ms/step - loss: 1.0320 - mean_absolute_error: 0.8118

<div class="k-default-codeblock">
```

```
</div>
 1076/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:45 431ms/step - loss: 1.0320 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1077/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:45 431ms/step - loss: 1.0320 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1078/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:44 431ms/step - loss: 1.0320 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1079/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:44 431ms/step - loss: 1.0319 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1080/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:43 431ms/step - loss: 1.0319 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1081/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:43 431ms/step - loss: 1.0319 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1082/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:42 431ms/step - loss: 1.0319 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1083/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:42 431ms/step - loss: 1.0318 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1084/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:42 431ms/step - loss: 1.0318 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1085/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:41 431ms/step - loss: 1.0318 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1086/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:41 431ms/step - loss: 1.0318 - mean_absolute_error: 0.8117

<div class="k-default-codeblock">
```

```
</div>
 1087/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:40 431ms/step - loss: 1.0318 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1088/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:40 431ms/step - loss: 1.0317 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1089/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:39 431ms/step - loss: 1.0317 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1090/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:39 431ms/step - loss: 1.0317 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1091/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:39 431ms/step - loss: 1.0317 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1092/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:38 431ms/step - loss: 1.0317 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1093/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:38 431ms/step - loss: 1.0316 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1094/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:37 431ms/step - loss: 1.0316 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1095/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:37 431ms/step - loss: 1.0316 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1096/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:37 432ms/step - loss: 1.0316 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1097/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:36 432ms/step - loss: 1.0315 - mean_absolute_error: 0.8116

<div class="k-default-codeblock">
```

```
</div>
 1098/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:36 432ms/step - loss: 1.0315 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1099/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:35 432ms/step - loss: 1.0315 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1100/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:35 432ms/step - loss: 1.0315 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1101/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:34 431ms/step - loss: 1.0315 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1102/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:34 431ms/step - loss: 1.0314 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1103/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:33 431ms/step - loss: 1.0314 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1104/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:33 431ms/step - loss: 1.0314 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1105/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:33 431ms/step - loss: 1.0314 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1106/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:32 431ms/step - loss: 1.0314 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1107/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:32 431ms/step - loss: 1.0313 - mean_absolute_error: 0.8115

<div class="k-default-codeblock">
```

```
</div>
 1108/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:31 431ms/step - loss: 1.0313 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1109/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:31 431ms/step - loss: 1.0313 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1110/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:30 431ms/step - loss: 1.0313 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1111/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:30 431ms/step - loss: 1.0312 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1112/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:29 431ms/step - loss: 1.0312 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1113/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:29 431ms/step - loss: 1.0312 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1114/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:29 431ms/step - loss: 1.0312 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1115/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:28 432ms/step - loss: 1.0312 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1116/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:28 432ms/step - loss: 1.0311 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1117/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:28 432ms/step - loss: 1.0311 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1118/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:27 432ms/step - loss: 1.0311 - mean_absolute_error: 0.8114

<div class="k-default-codeblock">
```

```
</div>
 1119/1599 ━━━━━━━━━━━━━[37m━━━━━━━  3:27 432ms/step - loss: 1.0311 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1120/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:26 432ms/step - loss: 1.0311 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1121/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:26 432ms/step - loss: 1.0310 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1122/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:26 432ms/step - loss: 1.0310 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1123/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:25 432ms/step - loss: 1.0310 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1124/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:25 432ms/step - loss: 1.0310 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1125/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:24 432ms/step - loss: 1.0309 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1126/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:24 433ms/step - loss: 1.0309 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1127/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:24 433ms/step - loss: 1.0309 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1128/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:23 433ms/step - loss: 1.0309 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1129/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:23 433ms/step - loss: 1.0309 - mean_absolute_error: 0.8113

<div class="k-default-codeblock">
```

```
</div>
 1130/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:23 433ms/step - loss: 1.0308 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1131/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:22 433ms/step - loss: 1.0308 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1132/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:22 433ms/step - loss: 1.0308 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1133/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:21 433ms/step - loss: 1.0308 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1134/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:21 433ms/step - loss: 1.0308 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1135/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:21 433ms/step - loss: 1.0307 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1136/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:20 434ms/step - loss: 1.0307 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1137/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:20 434ms/step - loss: 1.0307 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1138/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:19 434ms/step - loss: 1.0307 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1139/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:19 434ms/step - loss: 1.0307 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1140/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:19 434ms/step - loss: 1.0306 - mean_absolute_error: 0.8112

<div class="k-default-codeblock">
```

```
</div>
 1141/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:18 434ms/step - loss: 1.0306 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1142/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:18 434ms/step - loss: 1.0306 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1143/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:18 434ms/step - loss: 1.0306 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1144/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:17 434ms/step - loss: 1.0305 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1145/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:17 434ms/step - loss: 1.0305 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1146/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:16 435ms/step - loss: 1.0305 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1147/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:16 435ms/step - loss: 1.0305 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1148/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:16 435ms/step - loss: 1.0305 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1149/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:15 435ms/step - loss: 1.0304 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1150/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:15 436ms/step - loss: 1.0304 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1151/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:15 436ms/step - loss: 1.0304 - mean_absolute_error: 0.8111

<div class="k-default-codeblock">
```

```
</div>
 1152/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:14 436ms/step - loss: 1.0304 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1153/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:14 436ms/step - loss: 1.0304 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1154/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:14 436ms/step - loss: 1.0303 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1155/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:13 436ms/step - loss: 1.0303 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1156/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:13 436ms/step - loss: 1.0303 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1157/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:12 437ms/step - loss: 1.0303 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1158/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:12 437ms/step - loss: 1.0303 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1159/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:12 437ms/step - loss: 1.0302 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1160/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:11 437ms/step - loss: 1.0302 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1161/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:11 437ms/step - loss: 1.0302 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1162/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:10 437ms/step - loss: 1.0302 - mean_absolute_error: 0.8110

<div class="k-default-codeblock">
```

```
</div>
 1163/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:10 437ms/step - loss: 1.0301 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1164/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:10 437ms/step - loss: 1.0301 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1165/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:09 437ms/step - loss: 1.0301 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1166/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:09 437ms/step - loss: 1.0301 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1167/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:09 438ms/step - loss: 1.0301 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1168/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:08 438ms/step - loss: 1.0300 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1169/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:08 438ms/step - loss: 1.0300 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1170/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:07 438ms/step - loss: 1.0300 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1171/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:07 438ms/step - loss: 1.0300 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1172/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:07 438ms/step - loss: 1.0300 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1173/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:06 438ms/step - loss: 1.0299 - mean_absolute_error: 0.8109

<div class="k-default-codeblock">
```

```
</div>
 1174/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:06 438ms/step - loss: 1.0299 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1175/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:05 438ms/step - loss: 1.0299 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1176/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:05 438ms/step - loss: 1.0299 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1177/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:04 438ms/step - loss: 1.0299 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1178/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:04 438ms/step - loss: 1.0298 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1179/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:04 438ms/step - loss: 1.0298 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1180/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:03 439ms/step - loss: 1.0298 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1181/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:03 439ms/step - loss: 1.0298 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1182/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:02 439ms/step - loss: 1.0298 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1183/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:02 439ms/step - loss: 1.0297 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1184/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:02 439ms/step - loss: 1.0297 - mean_absolute_error: 0.8108

<div class="k-default-codeblock">
```

```
</div>
 1185/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:01 439ms/step - loss: 1.0297 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1186/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:01 439ms/step - loss: 1.0297 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1187/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:00 439ms/step - loss: 1.0296 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1188/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:00 439ms/step - loss: 1.0296 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1189/1599 ━━━━━━━━━━━━━━[37m━━━━━━  3:00 439ms/step - loss: 1.0296 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1190/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:59 439ms/step - loss: 1.0296 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1191/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:59 439ms/step - loss: 1.0296 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1192/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:58 439ms/step - loss: 1.0295 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1193/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:58 439ms/step - loss: 1.0295 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1194/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:57 439ms/step - loss: 1.0295 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1195/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:57 439ms/step - loss: 1.0295 - mean_absolute_error: 0.8107

<div class="k-default-codeblock">
```

```
</div>
 1196/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:57 439ms/step - loss: 1.0295 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1197/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:56 439ms/step - loss: 1.0294 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1198/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:56 439ms/step - loss: 1.0294 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1199/1599 ━━━━━━━━━━━━━━[37m━━━━━━  2:55 439ms/step - loss: 1.0294 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1200/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:55 439ms/step - loss: 1.0294 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1201/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:54 439ms/step - loss: 1.0294 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1202/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:54 440ms/step - loss: 1.0293 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1203/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:54 440ms/step - loss: 1.0293 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1204/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:53 440ms/step - loss: 1.0293 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1205/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:53 440ms/step - loss: 1.0293 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1206/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:52 440ms/step - loss: 1.0293 - mean_absolute_error: 0.8106

<div class="k-default-codeblock">
```

```
</div>
 1207/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:52 440ms/step - loss: 1.0292 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1208/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:52 440ms/step - loss: 1.0292 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1209/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:51 440ms/step - loss: 1.0292 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1210/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:51 440ms/step - loss: 1.0292 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1211/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:50 440ms/step - loss: 1.0292 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1212/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:50 440ms/step - loss: 1.0291 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1213/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:49 440ms/step - loss: 1.0291 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1214/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:49 440ms/step - loss: 1.0291 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1215/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:49 440ms/step - loss: 1.0291 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1216/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:48 440ms/step - loss: 1.0291 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1217/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:48 440ms/step - loss: 1.0290 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1218/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:47 440ms/step - loss: 1.0290 - mean_absolute_error: 0.8105

<div class="k-default-codeblock">
```

```
</div>
 1219/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:47 440ms/step - loss: 1.0290 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1220/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:46 440ms/step - loss: 1.0290 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1221/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:46 440ms/step - loss: 1.0290 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1222/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:46 440ms/step - loss: 1.0289 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1223/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:45 440ms/step - loss: 1.0289 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1224/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:45 440ms/step - loss: 1.0289 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1225/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:44 440ms/step - loss: 1.0289 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1226/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:44 440ms/step - loss: 1.0289 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1227/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:43 440ms/step - loss: 1.0288 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1228/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:43 440ms/step - loss: 1.0288 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1229/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:43 441ms/step - loss: 1.0288 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1230/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:42 441ms/step - loss: 1.0288 - mean_absolute_error: 0.8104

<div class="k-default-codeblock">
```

```
</div>
 1231/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:42 441ms/step - loss: 1.0288 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1232/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:41 441ms/step - loss: 1.0287 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1233/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:41 441ms/step - loss: 1.0287 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1234/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:40 441ms/step - loss: 1.0287 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1235/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:40 441ms/step - loss: 1.0287 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1236/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:40 441ms/step - loss: 1.0287 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1237/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:39 441ms/step - loss: 1.0287 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1238/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:39 441ms/step - loss: 1.0286 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1239/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:38 441ms/step - loss: 1.0286 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1240/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:38 441ms/step - loss: 1.0286 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1241/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:38 441ms/step - loss: 1.0286 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1242/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:37 441ms/step - loss: 1.0286 - mean_absolute_error: 0.8103

<div class="k-default-codeblock">
```

```
</div>
 1243/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:37 441ms/step - loss: 1.0285 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1244/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:36 441ms/step - loss: 1.0285 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1245/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:36 441ms/step - loss: 1.0285 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1246/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:35 442ms/step - loss: 1.0285 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1247/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:35 442ms/step - loss: 1.0285 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1248/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:35 442ms/step - loss: 1.0284 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1249/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:34 442ms/step - loss: 1.0284 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1250/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:34 442ms/step - loss: 1.0284 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1251/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:33 442ms/step - loss: 1.0284 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1252/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:33 442ms/step - loss: 1.0284 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1253/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:32 442ms/step - loss: 1.0284 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1254/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:32 442ms/step - loss: 1.0283 - mean_absolute_error: 0.8102

<div class="k-default-codeblock">
```

```
</div>
 1255/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:32 442ms/step - loss: 1.0283 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1256/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:31 442ms/step - loss: 1.0283 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1257/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:31 442ms/step - loss: 1.0283 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1258/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:30 442ms/step - loss: 1.0283 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1259/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:30 442ms/step - loss: 1.0282 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1260/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:30 442ms/step - loss: 1.0282 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1261/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:29 443ms/step - loss: 1.0282 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1262/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:29 443ms/step - loss: 1.0282 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1263/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:28 443ms/step - loss: 1.0282 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1264/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:28 443ms/step - loss: 1.0282 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1265/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:27 443ms/step - loss: 1.0281 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1266/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:27 443ms/step - loss: 1.0281 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1267/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:27 443ms/step - loss: 1.0281 - mean_absolute_error: 0.8101

<div class="k-default-codeblock">
```

```
</div>
 1268/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:26 443ms/step - loss: 1.0281 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1269/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:26 443ms/step - loss: 1.0281 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1270/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:25 443ms/step - loss: 1.0280 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1271/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:25 443ms/step - loss: 1.0280 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1272/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:24 443ms/step - loss: 1.0280 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1273/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:24 443ms/step - loss: 1.0280 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1274/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:24 443ms/step - loss: 1.0280 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1275/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:23 443ms/step - loss: 1.0280 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1276/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:23 443ms/step - loss: 1.0279 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1277/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:22 443ms/step - loss: 1.0279 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1278/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:22 443ms/step - loss: 1.0279 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1279/1599 ━━━━━━━━━━━━━━━[37m━━━━━  2:21 443ms/step - loss: 1.0279 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1280/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:21 443ms/step - loss: 1.0279 - mean_absolute_error: 0.8100

<div class="k-default-codeblock">
```

```
</div>
 1281/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:20 443ms/step - loss: 1.0279 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1282/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:20 443ms/step - loss: 1.0278 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1283/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:20 443ms/step - loss: 1.0278 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1284/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:19 443ms/step - loss: 1.0278 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1285/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:19 443ms/step - loss: 1.0278 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1286/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:18 443ms/step - loss: 1.0278 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1287/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:18 443ms/step - loss: 1.0277 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1288/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:17 443ms/step - loss: 1.0277 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1289/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:17 443ms/step - loss: 1.0277 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1290/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:17 443ms/step - loss: 1.0277 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1291/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:16 443ms/step - loss: 1.0277 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1292/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:16 444ms/step - loss: 1.0277 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1293/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:15 444ms/step - loss: 1.0276 - mean_absolute_error: 0.8099

<div class="k-default-codeblock">
```

```
</div>
 1294/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:15 444ms/step - loss: 1.0276 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1295/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:14 444ms/step - loss: 1.0276 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1296/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:14 444ms/step - loss: 1.0276 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1297/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:14 444ms/step - loss: 1.0276 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1298/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:13 444ms/step - loss: 1.0276 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1299/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:13 444ms/step - loss: 1.0275 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1300/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:12 444ms/step - loss: 1.0275 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1301/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:12 444ms/step - loss: 1.0275 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1302/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:11 444ms/step - loss: 1.0275 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1303/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:11 444ms/step - loss: 1.0275 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1304/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:11 444ms/step - loss: 1.0275 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1305/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:10 444ms/step - loss: 1.0274 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1306/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:10 444ms/step - loss: 1.0274 - mean_absolute_error: 0.8098

<div class="k-default-codeblock">
```

```
</div>
 1307/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:09 444ms/step - loss: 1.0274 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1308/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:09 444ms/step - loss: 1.0274 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1309/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:08 444ms/step - loss: 1.0274 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1310/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:08 444ms/step - loss: 1.0274 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1311/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:07 444ms/step - loss: 1.0273 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1312/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:07 444ms/step - loss: 1.0273 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1313/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:06 444ms/step - loss: 1.0273 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1314/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:06 444ms/step - loss: 1.0273 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1315/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:06 444ms/step - loss: 1.0273 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1316/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:05 444ms/step - loss: 1.0273 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1317/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:05 444ms/step - loss: 1.0272 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1318/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:04 444ms/step - loss: 1.0272 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1319/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:04 444ms/step - loss: 1.0272 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1320/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:03 444ms/step - loss: 1.0272 - mean_absolute_error: 0.8097

<div class="k-default-codeblock">
```

```
</div>
 1321/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:03 444ms/step - loss: 1.0272 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1322/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:03 444ms/step - loss: 1.0272 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1323/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:02 444ms/step - loss: 1.0271 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1324/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:02 444ms/step - loss: 1.0271 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1325/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:01 444ms/step - loss: 1.0271 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1326/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:01 444ms/step - loss: 1.0271 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1327/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:00 444ms/step - loss: 1.0271 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1328/1599 ━━━━━━━━━━━━━━━━[37m━━━━  2:00 444ms/step - loss: 1.0271 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1329/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:59 444ms/step - loss: 1.0270 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1330/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:59 444ms/step - loss: 1.0270 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1331/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:59 444ms/step - loss: 1.0270 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1332/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:58 444ms/step - loss: 1.0270 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1333/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:58 444ms/step - loss: 1.0270 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1334/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:57 445ms/step - loss: 1.0270 - mean_absolute_error: 0.8096

<div class="k-default-codeblock">
```

```
</div>
 1335/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:57 445ms/step - loss: 1.0269 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1336/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:56 445ms/step - loss: 1.0269 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1337/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:56 445ms/step - loss: 1.0269 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1338/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:56 445ms/step - loss: 1.0269 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1339/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:55 445ms/step - loss: 1.0269 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1340/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:55 445ms/step - loss: 1.0269 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1341/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:54 445ms/step - loss: 1.0269 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1342/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:54 445ms/step - loss: 1.0268 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1343/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:53 445ms/step - loss: 1.0268 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1344/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:53 445ms/step - loss: 1.0268 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1345/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:52 445ms/step - loss: 1.0268 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1346/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:52 445ms/step - loss: 1.0268 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1347/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:52 445ms/step - loss: 1.0268 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1348/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:51 445ms/step - loss: 1.0267 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1349/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:51 445ms/step - loss: 1.0267 - mean_absolute_error: 0.8095

<div class="k-default-codeblock">
```

```
</div>
 1350/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:50 445ms/step - loss: 1.0267 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1351/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:50 445ms/step - loss: 1.0267 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1352/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:49 445ms/step - loss: 1.0267 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1353/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:49 445ms/step - loss: 1.0267 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1354/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:48 445ms/step - loss: 1.0267 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1355/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:48 445ms/step - loss: 1.0266 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1356/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:48 445ms/step - loss: 1.0266 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1357/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:47 445ms/step - loss: 1.0266 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1358/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:47 445ms/step - loss: 1.0266 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1359/1599 ━━━━━━━━━━━━━━━━[37m━━━━  1:46 445ms/step - loss: 1.0266 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1360/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:46 445ms/step - loss: 1.0266 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1361/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:45 445ms/step - loss: 1.0265 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1362/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:45 445ms/step - loss: 1.0265 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1363/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:44 445ms/step - loss: 1.0265 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1364/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:44 445ms/step - loss: 1.0265 - mean_absolute_error: 0.8094

<div class="k-default-codeblock">
```

```
</div>
 1365/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:44 445ms/step - loss: 1.0265 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1366/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:43 445ms/step - loss: 1.0265 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1367/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:43 445ms/step - loss: 1.0265 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1368/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:42 445ms/step - loss: 1.0264 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1369/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:42 445ms/step - loss: 1.0264 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1370/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:41 445ms/step - loss: 1.0264 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1371/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:41 445ms/step - loss: 1.0264 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1372/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:41 445ms/step - loss: 1.0264 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1373/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:40 445ms/step - loss: 1.0264 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1374/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:40 445ms/step - loss: 1.0264 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1375/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:39 445ms/step - loss: 1.0263 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1376/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:39 445ms/step - loss: 1.0263 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1377/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:38 445ms/step - loss: 1.0263 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1378/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:38 445ms/step - loss: 1.0263 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1379/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:37 445ms/step - loss: 1.0263 - mean_absolute_error: 0.8093

<div class="k-default-codeblock">
```

```
</div>
 1380/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:37 445ms/step - loss: 1.0263 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1381/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:37 445ms/step - loss: 1.0263 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1382/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:36 445ms/step - loss: 1.0262 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1383/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:36 445ms/step - loss: 1.0262 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1384/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:35 445ms/step - loss: 1.0262 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1385/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:35 445ms/step - loss: 1.0262 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1386/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:34 445ms/step - loss: 1.0262 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1387/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:34 445ms/step - loss: 1.0262 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1388/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:33 445ms/step - loss: 1.0261 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1389/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:33 445ms/step - loss: 1.0261 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1390/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:33 445ms/step - loss: 1.0261 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1391/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:32 445ms/step - loss: 1.0261 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1392/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:32 445ms/step - loss: 1.0261 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1393/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:31 445ms/step - loss: 1.0261 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1394/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:31 445ms/step - loss: 1.0261 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1395/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:30 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8092

<div class="k-default-codeblock">
```

```
</div>
 1396/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:30 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1397/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:29 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1398/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:29 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1399/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:28 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1400/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:28 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1401/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:28 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1402/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:27 445ms/step - loss: 1.0260 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1403/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:27 445ms/step - loss: 1.0259 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1404/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:26 445ms/step - loss: 1.0259 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1405/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:26 445ms/step - loss: 1.0259 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1406/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:25 445ms/step - loss: 1.0259 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1407/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:25 445ms/step - loss: 1.0259 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1408/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:25 445ms/step - loss: 1.0259 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1409/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:24 445ms/step - loss: 1.0259 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1410/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:24 445ms/step - loss: 1.0258 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1411/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:23 445ms/step - loss: 1.0258 - mean_absolute_error: 0.8091

<div class="k-default-codeblock">
```

```
</div>
 1412/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:23 445ms/step - loss: 1.0258 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1413/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:22 445ms/step - loss: 1.0258 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1414/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:22 445ms/step - loss: 1.0258 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1415/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:21 445ms/step - loss: 1.0258 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1416/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:21 445ms/step - loss: 1.0258 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1417/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:21 445ms/step - loss: 1.0257 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1418/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:20 445ms/step - loss: 1.0257 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1419/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:20 445ms/step - loss: 1.0257 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1420/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:19 445ms/step - loss: 1.0257 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1421/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:19 445ms/step - loss: 1.0257 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1422/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:18 445ms/step - loss: 1.0257 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1423/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:18 445ms/step - loss: 1.0257 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1424/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:17 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1425/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:17 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1426/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:17 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1427/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:16 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1428/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:16 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8090

<div class="k-default-codeblock">
```

```
</div>
 1429/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:15 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1430/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:15 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1431/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:14 445ms/step - loss: 1.0256 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1432/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:14 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1433/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:13 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1434/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:13 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1435/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:13 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1436/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:12 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1437/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:12 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1438/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:11 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1439/1599 ━━━━━━━━━━━━━━━━━[37m━━━  1:11 445ms/step - loss: 1.0255 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1440/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:10 445ms/step - loss: 1.0254 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1441/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:10 445ms/step - loss: 1.0254 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1442/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:09 445ms/step - loss: 1.0254 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1443/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:09 445ms/step - loss: 1.0254 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1444/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:09 445ms/step - loss: 1.0254 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1445/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:08 445ms/step - loss: 1.0254 - mean_absolute_error: 0.8089

<div class="k-default-codeblock">
```

```
</div>
 1446/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:08 445ms/step - loss: 1.0254 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1447/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:07 445ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1448/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:07 445ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1449/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:06 445ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1450/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:06 445ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1451/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:05 446ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1452/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:05 446ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1453/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:05 446ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1454/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:04 446ms/step - loss: 1.0253 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1455/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:04 446ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1456/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:03 446ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1457/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:03 445ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1458/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:02 445ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1459/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:02 445ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1460/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:01 445ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1461/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:01 445ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1462/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:01 445ms/step - loss: 1.0252 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1463/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:00 445ms/step - loss: 1.0251 - mean_absolute_error: 0.8088

<div class="k-default-codeblock">
```

```
</div>
 1464/1599 ━━━━━━━━━━━━━━━━━━[37m━━  1:00 445ms/step - loss: 1.0251 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1465/1599 ━━━━━━━━━━━━━━━━━━[37m━━  59s 445ms/step - loss: 1.0251 - mean_absolute_error: 0.8087 

<div class="k-default-codeblock">
```

```
</div>
 1466/1599 ━━━━━━━━━━━━━━━━━━[37m━━  59s 446ms/step - loss: 1.0251 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1467/1599 ━━━━━━━━━━━━━━━━━━[37m━━  58s 446ms/step - loss: 1.0251 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1468/1599 ━━━━━━━━━━━━━━━━━━[37m━━  58s 446ms/step - loss: 1.0251 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1469/1599 ━━━━━━━━━━━━━━━━━━[37m━━  57s 446ms/step - loss: 1.0251 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1470/1599 ━━━━━━━━━━━━━━━━━━[37m━━  57s 446ms/step - loss: 1.0251 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1471/1599 ━━━━━━━━━━━━━━━━━━[37m━━  57s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1472/1599 ━━━━━━━━━━━━━━━━━━[37m━━  56s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1473/1599 ━━━━━━━━━━━━━━━━━━[37m━━  56s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1474/1599 ━━━━━━━━━━━━━━━━━━[37m━━  55s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1475/1599 ━━━━━━━━━━━━━━━━━━[37m━━  55s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1476/1599 ━━━━━━━━━━━━━━━━━━[37m━━  54s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1477/1599 ━━━━━━━━━━━━━━━━━━[37m━━  54s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1478/1599 ━━━━━━━━━━━━━━━━━━[37m━━  53s 446ms/step - loss: 1.0250 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1479/1599 ━━━━━━━━━━━━━━━━━━[37m━━  53s 446ms/step - loss: 1.0249 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1480/1599 ━━━━━━━━━━━━━━━━━━[37m━━  53s 446ms/step - loss: 1.0249 - mean_absolute_error: 0.8087

<div class="k-default-codeblock">
```

```
</div>
 1481/1599 ━━━━━━━━━━━━━━━━━━[37m━━  52s 446ms/step - loss: 1.0249 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1482/1599 ━━━━━━━━━━━━━━━━━━[37m━━  52s 446ms/step - loss: 1.0249 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1483/1599 ━━━━━━━━━━━━━━━━━━[37m━━  51s 446ms/step - loss: 1.0249 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1484/1599 ━━━━━━━━━━━━━━━━━━[37m━━  51s 446ms/step - loss: 1.0249 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1485/1599 ━━━━━━━━━━━━━━━━━━[37m━━  50s 446ms/step - loss: 1.0249 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1486/1599 ━━━━━━━━━━━━━━━━━━[37m━━  50s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1487/1599 ━━━━━━━━━━━━━━━━━━[37m━━  49s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1488/1599 ━━━━━━━━━━━━━━━━━━[37m━━  49s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1489/1599 ━━━━━━━━━━━━━━━━━━[37m━━  49s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1490/1599 ━━━━━━━━━━━━━━━━━━[37m━━  48s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1491/1599 ━━━━━━━━━━━━━━━━━━[37m━━  48s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1492/1599 ━━━━━━━━━━━━━━━━━━[37m━━  47s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1493/1599 ━━━━━━━━━━━━━━━━━━[37m━━  47s 446ms/step - loss: 1.0248 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1494/1599 ━━━━━━━━━━━━━━━━━━[37m━━  46s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1495/1599 ━━━━━━━━━━━━━━━━━━[37m━━  46s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1496/1599 ━━━━━━━━━━━━━━━━━━[37m━━  45s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1497/1599 ━━━━━━━━━━━━━━━━━━[37m━━  45s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1498/1599 ━━━━━━━━━━━━━━━━━━[37m━━  45s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8086

<div class="k-default-codeblock">
```

```
</div>
 1499/1599 ━━━━━━━━━━━━━━━━━━[37m━━  44s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1500/1599 ━━━━━━━━━━━━━━━━━━[37m━━  44s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1501/1599 ━━━━━━━━━━━━━━━━━━[37m━━  43s 446ms/step - loss: 1.0247 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1502/1599 ━━━━━━━━━━━━━━━━━━[37m━━  43s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1503/1599 ━━━━━━━━━━━━━━━━━━[37m━━  42s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1504/1599 ━━━━━━━━━━━━━━━━━━[37m━━  42s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1505/1599 ━━━━━━━━━━━━━━━━━━[37m━━  41s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1506/1599 ━━━━━━━━━━━━━━━━━━[37m━━  41s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1507/1599 ━━━━━━━━━━━━━━━━━━[37m━━  41s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1508/1599 ━━━━━━━━━━━━━━━━━━[37m━━  40s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1509/1599 ━━━━━━━━━━━━━━━━━━[37m━━  40s 446ms/step - loss: 1.0246 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1510/1599 ━━━━━━━━━━━━━━━━━━[37m━━  39s 447ms/step - loss: 1.0245 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1511/1599 ━━━━━━━━━━━━━━━━━━[37m━━  39s 446ms/step - loss: 1.0245 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1512/1599 ━━━━━━━━━━━━━━━━━━[37m━━  38s 446ms/step - loss: 1.0245 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1513/1599 ━━━━━━━━━━━━━━━━━━[37m━━  38s 446ms/step - loss: 1.0245 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1514/1599 ━━━━━━━━━━━━━━━━━━[37m━━  37s 447ms/step - loss: 1.0245 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1515/1599 ━━━━━━━━━━━━━━━━━━[37m━━  37s 447ms/step - loss: 1.0245 - mean_absolute_error: 0.8085

<div class="k-default-codeblock">
```

```
</div>
 1516/1599 ━━━━━━━━━━━━━━━━━━[37m━━  37s 447ms/step - loss: 1.0245 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1517/1599 ━━━━━━━━━━━━━━━━━━[37m━━  36s 447ms/step - loss: 1.0245 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1518/1599 ━━━━━━━━━━━━━━━━━━[37m━━  36s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1519/1599 ━━━━━━━━━━━━━━━━━━[37m━━  35s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1520/1599 ━━━━━━━━━━━━━━━━━━━[37m━  35s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1521/1599 ━━━━━━━━━━━━━━━━━━━[37m━  34s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1522/1599 ━━━━━━━━━━━━━━━━━━━[37m━  34s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1523/1599 ━━━━━━━━━━━━━━━━━━━[37m━  33s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1524/1599 ━━━━━━━━━━━━━━━━━━━[37m━  33s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1525/1599 ━━━━━━━━━━━━━━━━━━━[37m━  33s 447ms/step - loss: 1.0244 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1526/1599 ━━━━━━━━━━━━━━━━━━━[37m━  32s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1527/1599 ━━━━━━━━━━━━━━━━━━━[37m━  32s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1528/1599 ━━━━━━━━━━━━━━━━━━━[37m━  31s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1529/1599 ━━━━━━━━━━━━━━━━━━━[37m━  31s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1530/1599 ━━━━━━━━━━━━━━━━━━━[37m━  30s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1531/1599 ━━━━━━━━━━━━━━━━━━━[37m━  30s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1532/1599 ━━━━━━━━━━━━━━━━━━━[37m━  29s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1533/1599 ━━━━━━━━━━━━━━━━━━━[37m━  29s 447ms/step - loss: 1.0243 - mean_absolute_error: 0.8084

<div class="k-default-codeblock">
```

```
</div>
 1534/1599 ━━━━━━━━━━━━━━━━━━━[37m━  29s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1535/1599 ━━━━━━━━━━━━━━━━━━━[37m━  28s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1536/1599 ━━━━━━━━━━━━━━━━━━━[37m━  28s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1537/1599 ━━━━━━━━━━━━━━━━━━━[37m━  27s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1538/1599 ━━━━━━━━━━━━━━━━━━━[37m━  27s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1539/1599 ━━━━━━━━━━━━━━━━━━━[37m━  26s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1540/1599 ━━━━━━━━━━━━━━━━━━━[37m━  26s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1541/1599 ━━━━━━━━━━━━━━━━━━━[37m━  25s 447ms/step - loss: 1.0242 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1542/1599 ━━━━━━━━━━━━━━━━━━━[37m━  25s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1543/1599 ━━━━━━━━━━━━━━━━━━━[37m━  25s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1544/1599 ━━━━━━━━━━━━━━━━━━━[37m━  24s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1545/1599 ━━━━━━━━━━━━━━━━━━━[37m━  24s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1546/1599 ━━━━━━━━━━━━━━━━━━━[37m━  23s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1547/1599 ━━━━━━━━━━━━━━━━━━━[37m━  23s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1548/1599 ━━━━━━━━━━━━━━━━━━━[37m━  22s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1549/1599 ━━━━━━━━━━━━━━━━━━━[37m━  22s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1550/1599 ━━━━━━━━━━━━━━━━━━━[37m━  21s 447ms/step - loss: 1.0241 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1551/1599 ━━━━━━━━━━━━━━━━━━━[37m━  21s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1552/1599 ━━━━━━━━━━━━━━━━━━━[37m━  21s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8083

<div class="k-default-codeblock">
```

```
</div>
 1553/1599 ━━━━━━━━━━━━━━━━━━━[37m━  20s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1554/1599 ━━━━━━━━━━━━━━━━━━━[37m━  20s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1555/1599 ━━━━━━━━━━━━━━━━━━━[37m━  19s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1556/1599 ━━━━━━━━━━━━━━━━━━━[37m━  19s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1557/1599 ━━━━━━━━━━━━━━━━━━━[37m━  18s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1558/1599 ━━━━━━━━━━━━━━━━━━━[37m━  18s 447ms/step - loss: 1.0240 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1559/1599 ━━━━━━━━━━━━━━━━━━━[37m━  17s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1560/1599 ━━━━━━━━━━━━━━━━━━━[37m━  17s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1561/1599 ━━━━━━━━━━━━━━━━━━━[37m━  16s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1562/1599 ━━━━━━━━━━━━━━━━━━━[37m━  16s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1563/1599 ━━━━━━━━━━━━━━━━━━━[37m━  16s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1564/1599 ━━━━━━━━━━━━━━━━━━━[37m━  15s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1565/1599 ━━━━━━━━━━━━━━━━━━━[37m━  15s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1566/1599 ━━━━━━━━━━━━━━━━━━━[37m━  14s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1567/1599 ━━━━━━━━━━━━━━━━━━━[37m━  14s 447ms/step - loss: 1.0239 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1568/1599 ━━━━━━━━━━━━━━━━━━━[37m━  13s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1569/1599 ━━━━━━━━━━━━━━━━━━━[37m━  13s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1570/1599 ━━━━━━━━━━━━━━━━━━━[37m━  12s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1571/1599 ━━━━━━━━━━━━━━━━━━━[37m━  12s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8082

<div class="k-default-codeblock">
```

```
</div>
 1572/1599 ━━━━━━━━━━━━━━━━━━━[37m━  12s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1573/1599 ━━━━━━━━━━━━━━━━━━━[37m━  11s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1574/1599 ━━━━━━━━━━━━━━━━━━━[37m━  11s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1575/1599 ━━━━━━━━━━━━━━━━━━━[37m━  10s 447ms/step - loss: 1.0238 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1576/1599 ━━━━━━━━━━━━━━━━━━━[37m━  10s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1577/1599 ━━━━━━━━━━━━━━━━━━━[37m━  9s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081 

<div class="k-default-codeblock">
```

```
</div>
 1578/1599 ━━━━━━━━━━━━━━━━━━━[37m━  9s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1579/1599 ━━━━━━━━━━━━━━━━━━━[37m━  8s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1580/1599 ━━━━━━━━━━━━━━━━━━━[37m━  8s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1581/1599 ━━━━━━━━━━━━━━━━━━━[37m━  8s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1582/1599 ━━━━━━━━━━━━━━━━━━━[37m━  7s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1583/1599 ━━━━━━━━━━━━━━━━━━━[37m━  7s 447ms/step - loss: 1.0237 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1584/1599 ━━━━━━━━━━━━━━━━━━━[37m━  6s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1585/1599 ━━━━━━━━━━━━━━━━━━━[37m━  6s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1586/1599 ━━━━━━━━━━━━━━━━━━━[37m━  5s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1587/1599 ━━━━━━━━━━━━━━━━━━━[37m━  5s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1588/1599 ━━━━━━━━━━━━━━━━━━━[37m━  4s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1589/1599 ━━━━━━━━━━━━━━━━━━━[37m━  4s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1590/1599 ━━━━━━━━━━━━━━━━━━━[37m━  4s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8081

<div class="k-default-codeblock">
```

```
</div>
 1591/1599 ━━━━━━━━━━━━━━━━━━━[37m━  3s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1592/1599 ━━━━━━━━━━━━━━━━━━━[37m━  3s 447ms/step - loss: 1.0236 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1593/1599 ━━━━━━━━━━━━━━━━━━━[37m━  2s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1594/1599 ━━━━━━━━━━━━━━━━━━━[37m━  2s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1595/1599 ━━━━━━━━━━━━━━━━━━━[37m━  1s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1596/1599 ━━━━━━━━━━━━━━━━━━━[37m━  1s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1597/1599 ━━━━━━━━━━━━━━━━━━━[37m━  0s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1598/1599 ━━━━━━━━━━━━━━━━━━━[37m━  0s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1599/1599 ━━━━━━━━━━━━━━━━━━━━ 0s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080

<div class="k-default-codeblock">
```

```
</div>
 1599/1599 ━━━━━━━━━━━━━━━━━━━━ 715s 447ms/step - loss: 1.0235 - mean_absolute_error: 0.8080


<div class="k-default-codeblock">
```
Test MAE: 0.773

```
</div>
You should achieve a Mean Absolute Error (MAE) at or around 0.7 on the test data.

---
## Conclusion

The BST model uses the Transformer layer in its architecture to capture the sequential signals underlying
users’ behavior sequences for recommendation.

You can try training this model with different configurations, for example, by increasing
the input sequence length and training the model for a larger number of epochs. In addition,
you can try including other features like movie release year and customer
zipcode, and including cross features like sex X genre.

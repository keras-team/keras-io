# Semantic Image Clustering

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/02/28<br>
**Last modified:** 2021/02/28<br>
**Description:** Semantic Clustering by Adopting Nearest neighbors (SCAN) algorithm.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/semantic_image_clustering.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/semantic_image_clustering.py)



---
## Introduction

This example demonstrates how to apply the [Semantic Clustering by Adopting Nearest neighbors
(SCAN)](https://arxiv.org/abs/2005.12320) algorithm (Van Gansbeke et al., 2020) on the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. The algorithm consists of
two phases:

1. Self-supervised visual representation learning of images, in which we use the
[simCLR](https://arxiv.org/abs/2002.05709) technique.
2. Clustering of the learned visual representation vectors to maximize the agreement
between the cluster assignments of neighboring vectors.

The example requires [TensorFlow Addons](https://www.tensorflow.org/addons),
which you can install using the following command:

```python
pip install tensorflow-addons
```

---
## Setup


```python
from collections import defaultdict
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm import tqdm
```

---
## Prepare the data


```python
num_classes = 10
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_data = np.concatenate([x_train, x_test])
y_data = np.concatenate([y_train, y_test])

print("x_data shape:", x_data.shape, "- y_data shape:", y_data.shape)

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
```

<div class="k-default-codeblock">
```
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170500096/170498071 [==============================] - 13s 0us/step
170508288/170498071 [==============================] - 13s 0us/step
x_data shape: (60000, 32, 32, 3) - y_data shape: (60000, 1)

```
</div>
---
## Define hyperparameters


```python
target_size = 32  # Resize the input images.
representation_dim = 512  # The dimensions of the features vector.
projection_units = 128  # The projection head of the representation learner.
num_clusters = 20  # Number of clusters.
k_neighbours = 5  # Number of neighbours to consider during cluster learning.
tune_encoder_during_clustering = False  # Freeze the encoder in the cluster learning.
```

---
## Implement data preprocessing

The data preprocessing step resizes the input images to the desired `target_size` and applies
feature-wise normalization. Note that, when using `keras.applications.ResNet50V2` as the
visual encoder, resizing the images into 255 x 255 inputs would lead to more accurate results
but require a longer time to train.


```python
data_preprocessing = keras.Sequential(
    [
        layers.Resizing(target_size, target_size),
        layers.Normalization(),
    ]
)
# Compute the mean and the variance from the data for normalization.
data_preprocessing.layers[-1].adapt(x_data)
```

---
## Data augmentation

Unlike simCLR, which randomly picks a single data augmentation function to apply to an input
image, we apply a set of data augmentation functions randomly to the input image.
(You can experiment with other image augmentation techniques by following
the [data augmentation tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation).)


```python
data_augmentation = keras.Sequential(
    [
        layers.RandomTranslation(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
        ),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.15, fill_mode="nearest"),
        layers.RandomZoom(
            height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
        ),
    ]
)
```

Display a random image


```python
image_idx = np.random.choice(range(x_data.shape[0]))
image = x_data[image_idx]
image_class = classes[y_data[image_idx][0]]
plt.figure(figsize=(3, 3))
plt.imshow(x_data[image_idx].astype("uint8"))
plt.title(image_class)
_ = plt.axis("off")
```


![png](/img/examples/vision/semantic_image_clustering/semantic_image_clustering_13_0.png)


Display a sample of augmented versions of the image


```python
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_images = data_augmentation(np.array([image]))
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
```


![png](/img/examples/vision/semantic_image_clustering/semantic_image_clustering_15_0.png)


---
## Self-supervised representation learning

### Implement the vision encoder


```python

def create_encoder(representation_dim):
    encoder = keras.Sequential(
        [
            keras.applications.ResNet50V2(
                include_top=False, weights=None, pooling="avg"
            ),
            layers.Dense(representation_dim),
        ]
    )
    return encoder

```

### Implement the unsupervised contrastive loss


```python

class RepresentationLearner(keras.Model):
    def __init__(
        self,
        encoder,
        projection_units,
        num_augmentations,
        temperature=1.0,
        dropout_rate=0.1,
        l2_normalize=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        # Create projection head.
        self.projector = keras.Sequential(
            [
                layers.Dropout(dropout_rate),
                layers.Dense(units=projection_units, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vectors, batch_size):
        num_augmentations = tf.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = tf.math.l2_normalize(feature_vectors, -1)
        # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
        logits = (
            tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
            / self.temperature
        )
        # Apply log-max trick for numerical stability.
        logits_max = tf.math.reduce_max(logits, axis=1)
        logits = logits - logits_max
        # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
        # targets is a matrix consits of num_augmentations submatrices of shape [batch_size * batch_size].
        # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
        targets = tf.tile(tf.eye(batch_size), [num_augmentations, num_augmentations])
        # Compute cross entropy loss
        return keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    def call(self, inputs):
        # Preprocess the input images.
        preprocessed = data_preprocessing(inputs)
        # Create augmented versions of the images.
        augmented = []
        for _ in range(self.num_augmentations):
            augmented.append(data_augmentation(preprocessed))
        augmented = layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(augmented)
        # Apply projection head.
        return self.projector(features)

    def train_step(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        batch_size = tf.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

```

### Train the model


```python
# Create vision encoder.
encoder = create_encoder(representation_dim)
# Create representation learner.
representation_learner = RepresentationLearner(
    encoder, projection_units, num_augmentations=2, temperature=0.1
)
# Create a a Cosine decay learning rate scheduler.
lr_scheduler = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=500, alpha=0.1
)
# Compile the model.
representation_learner.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001),
)
# Fit the model.
history = representation_learner.fit(
    x=x_data,
    batch_size=512,
    epochs=50,  # for better results, increase the number of epochs to 500.
)

```

<div class="k-default-codeblock">
```
Epoch 1/50
118/118 [==============================] - 70s 351ms/step - loss: 53.4089
Epoch 2/50
118/118 [==============================] - 39s 328ms/step - loss: 13.8591
Epoch 3/50
118/118 [==============================] - 39s 333ms/step - loss: 11.8397
Epoch 4/50
118/118 [==============================] - 40s 338ms/step - loss: 11.5879
Epoch 5/50
118/118 [==============================] - 40s 341ms/step - loss: 11.1749
Epoch 6/50
118/118 [==============================] - 40s 343ms/step - loss: 10.9583
Epoch 7/50
118/118 [==============================] - 41s 344ms/step - loss: 10.8544
Epoch 8/50
118/118 [==============================] - 41s 345ms/step - loss: 10.7517
Epoch 9/50
118/118 [==============================] - 41s 346ms/step - loss: 10.6248
Epoch 10/50
118/118 [==============================] - 41s 346ms/step - loss: 10.5156
Epoch 11/50
118/118 [==============================] - 41s 346ms/step - loss: 10.4036
Epoch 12/50
118/118 [==============================] - 41s 345ms/step - loss: 10.2672
Epoch 13/50
118/118 [==============================] - 41s 346ms/step - loss: 10.1477
Epoch 14/50
118/118 [==============================] - 41s 346ms/step - loss: 10.0444
Epoch 15/50
118/118 [==============================] - 41s 346ms/step - loss: 9.9758
Epoch 16/50
118/118 [==============================] - 41s 346ms/step - loss: 9.8623
Epoch 17/50
118/118 [==============================] - 41s 345ms/step - loss: 9.7079
Epoch 18/50
118/118 [==============================] - 41s 346ms/step - loss: 9.6141
Epoch 19/50
118/118 [==============================] - 41s 346ms/step - loss: 9.4421
Epoch 20/50
118/118 [==============================] - 41s 346ms/step - loss: 9.2634
Epoch 21/50
118/118 [==============================] - 41s 346ms/step - loss: 9.1574
Epoch 22/50
118/118 [==============================] - 41s 346ms/step - loss: 9.0650
Epoch 23/50
118/118 [==============================] - 41s 346ms/step - loss: 8.8151
Epoch 24/50
118/118 [==============================] - 41s 346ms/step - loss: 8.6706
Epoch 25/50
118/118 [==============================] - 41s 346ms/step - loss: 8.4993
Epoch 26/50
118/118 [==============================] - 41s 345ms/step - loss: 8.4586
Epoch 27/50
118/118 [==============================] - 41s 345ms/step - loss: 8.3577
Epoch 28/50
118/118 [==============================] - 41s 346ms/step - loss: 8.0840
Epoch 29/50
118/118 [==============================] - 41s 346ms/step - loss: 7.9753
Epoch 30/50
118/118 [==============================] - 41s 346ms/step - loss: 7.7742
Epoch 31/50
118/118 [==============================] - 41s 346ms/step - loss: 7.6332
Epoch 32/50
118/118 [==============================] - 41s 346ms/step - loss: 7.7878
Epoch 33/50
118/118 [==============================] - 41s 346ms/step - loss: 7.6894
Epoch 34/50
118/118 [==============================] - 41s 346ms/step - loss: 7.3130
Epoch 35/50
118/118 [==============================] - 41s 346ms/step - loss: 7.2549
Epoch 36/50
118/118 [==============================] - 41s 346ms/step - loss: 7.0269
Epoch 37/50
118/118 [==============================] - 41s 346ms/step - loss: 6.7713
Epoch 38/50
118/118 [==============================] - 41s 346ms/step - loss: 6.8245
Epoch 39/50
118/118 [==============================] - 41s 346ms/step - loss: 6.7953
Epoch 40/50
118/118 [==============================] - 41s 346ms/step - loss: 6.7573
Epoch 41/50
118/118 [==============================] - 41s 346ms/step - loss: 6.7621
Epoch 42/50
118/118 [==============================] - 41s 346ms/step - loss: 6.7473
Epoch 43/50
118/118 [==============================] - 41s 346ms/step - loss: 6.3506
Epoch 44/50
118/118 [==============================] - 41s 346ms/step - loss: 6.1783
Epoch 45/50
118/118 [==============================] - 41s 345ms/step - loss: 6.0123
Epoch 46/50
118/118 [==============================] - 41s 346ms/step - loss: 5.9238
Epoch 47/50
118/118 [==============================] - 41s 345ms/step - loss: 5.9278
Epoch 48/50
118/118 [==============================] - 41s 346ms/step - loss: 5.7985
Epoch 49/50
118/118 [==============================] - 41s 346ms/step - loss: 6.0905
Epoch 50/50
118/118 [==============================] - 41s 346ms/step - loss: 5.9406

```
</div>
Plot training loss


```python
plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
```


![png](/img/examples/vision/semantic_image_clustering/semantic_image_clustering_24_0.png)


---
## Compute the nearest neighbors

### Generate the embeddings for the images


```python
batch_size = 500
# Get the feature vector representations of the images.
feature_vectors = encoder.predict(x_data, batch_size=batch_size, verbose=1)
# Normalize the feature vectores.
feature_vectors = tf.math.l2_normalize(feature_vectors, -1)
```

<div class="k-default-codeblock">
```
120/120 [==============================] - 7s 35ms/step

```
</div>
### Find the *k* nearest neighbours for each embedding


```python
neighbours = []
num_batches = feature_vectors.shape[0] // batch_size
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    current_batch = feature_vectors[start_idx:end_idx]
    # Compute the dot similarity.
    similarities = tf.linalg.matmul(current_batch, feature_vectors, transpose_b=True)
    # Get the indices of most similar vectors.
    _, indices = tf.math.top_k(similarities, k=k_neighbours + 1, sorted=True)
    # Add the indices to the neighbours.
    neighbours.append(indices[..., 1:])

neighbours = np.reshape(np.array(neighbours), (-1, k_neighbours))
```

<div class="k-default-codeblock">
```
100%|██████████| 120/120 [00:01<00:00, 99.09it/s]

```
</div>
Let's display some neighbors on each row


```python
nrows = 4
ncols = k_neighbours + 1

plt.figure(figsize=(12, 12))
position = 1
for _ in range(nrows):
    anchor_idx = np.random.choice(range(x_data.shape[0]))
    neighbour_indicies = neighbours[anchor_idx]
    indices = [anchor_idx] + neighbour_indicies.tolist()
    for j in range(ncols):
        plt.subplot(nrows, ncols, position)
        plt.imshow(x_data[indices[j]].astype("uint8"))
        plt.title(classes[y_data[indices[j]][0]])
        plt.axis("off")
        position += 1
```


![png](/img/examples/vision/semantic_image_clustering/semantic_image_clustering_31_0.png)


You notice that images on each row are visually similar, and belong to similar classes.

---
## Semantic clustering with nearest neighbours

### Implement clustering consistency loss

This loss tries to make sure that neighbours have the same clustering assignments.


```python

class ClustersConsistencyLoss(keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, target, similarity, sample_weight=None):
        # Set targets to be ones.
        target = tf.ones_like(similarity)
        # Compute cross entropy loss.
        loss = keras.losses.binary_crossentropy(
            y_true=target, y_pred=similarity, from_logits=True
        )
        return tf.math.reduce_mean(loss)

```

### Implement the clusters entropy loss

This loss tries to make sure that cluster distribution is roughly uniformed, to avoid
assigning most of the instances to one cluster.


```python

class ClustersEntropyLoss(keras.losses.Loss):
    def __init__(self, entropy_loss_weight=1.0):
        super().__init__()
        self.entropy_loss_weight = entropy_loss_weight

    def __call__(self, target, cluster_probabilities, sample_weight=None):
        # Ideal entropy = log(num_clusters).
        num_clusters = tf.cast(tf.shape(cluster_probabilities)[-1], tf.dtypes.float32)
        target = tf.math.log(num_clusters)
        # Compute the overall clusters distribution.
        cluster_probabilities = tf.math.reduce_mean(cluster_probabilities, axis=0)
        # Replacing zero probabilities - if any - with a very small value.
        cluster_probabilities = tf.clip_by_value(
            cluster_probabilities, clip_value_min=1e-8, clip_value_max=1.0
        )
        # Compute the entropy over the clusters.
        entropy = -tf.math.reduce_sum(
            cluster_probabilities * tf.math.log(cluster_probabilities)
        )
        # Compute the difference between the target and the actual.
        loss = target - entropy
        return loss

```

### Implement clustering model

This model takes a raw image as an input, generated its feature vector using the trained
encoder, and produces a probability distribution of the clusters given the feature vector
as the cluster assignments.


```python

def create_clustering_model(encoder, num_clusters, name=None):
    inputs = keras.Input(shape=input_shape)
    # Preprocess the input images.
    preprocessed = data_preprocessing(inputs)
    # Apply data augmentation to the images.
    augmented = data_augmentation(preprocessed)
    # Generate embedding representations of the images.
    features = encoder(augmented)
    # Assign the images to clusters.
    outputs = layers.Dense(units=num_clusters, activation="softmax")(features)
    # Create the model.
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

```

### Implement clustering learner

This model receives the input `anchor` image and its `neighbours`, produces the clusters
assignments for them using the `clustering_model`, and produces two outputs:
1. `similarity`: the similarity between the cluster assignments of the `anchor` image and
its `neighbours`. This output is fed to the `ClustersConsistencyLoss`.
2. `anchor_clustering`: cluster assignments of the `anchor` images. This is fed to the `ClustersEntropyLoss`.


```python

def create_clustering_learner(clustering_model):
    anchor = keras.Input(shape=input_shape, name="anchors")
    neighbours = keras.Input(
        shape=tuple([k_neighbours]) + input_shape, name="neighbours"
    )
    # Changes neighbours shape to [batch_size * k_neighbours, width, height, channels]
    neighbours_reshaped = tf.reshape(neighbours, shape=tuple([-1]) + input_shape)
    # anchor_clustering shape: [batch_size, num_clusters]
    anchor_clustering = clustering_model(anchor)
    # neighbours_clustering shape: [batch_size * k_neighbours, num_clusters]
    neighbours_clustering = clustering_model(neighbours_reshaped)
    # Convert neighbours_clustering shape to [batch_size, k_neighbours, num_clusters]
    neighbours_clustering = tf.reshape(
        neighbours_clustering,
        shape=(-1, k_neighbours, tf.shape(neighbours_clustering)[-1]),
    )
    # similarity shape: [batch_size, 1, k_neighbours]
    similarity = tf.linalg.einsum(
        "bij,bkj->bik", tf.expand_dims(anchor_clustering, axis=1), neighbours_clustering
    )
    # similarity shape:  [batch_size, k_neighbours]
    similarity = layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="similarity")(
        similarity
    )
    # Create the model.
    model = keras.Model(
        inputs=[anchor, neighbours],
        outputs=[similarity, anchor_clustering],
        name="clustering_learner",
    )
    return model

```

### Train model


```python
# If tune_encoder_during_clustering is set to False,
# then freeze the encoder weights.
for layer in encoder.layers:
    layer.trainable = tune_encoder_during_clustering
# Create the clustering model and learner.
clustering_model = create_clustering_model(encoder, num_clusters, name="clustering")
clustering_learner = create_clustering_learner(clustering_model)
# Instantiate the model losses.
losses = [ClustersConsistencyLoss(), ClustersEntropyLoss(entropy_loss_weight=5)]
# Create the model inputs and labels.
inputs = {"anchors": x_data, "neighbours": tf.gather(x_data, neighbours)}
labels = tf.ones(shape=(x_data.shape[0]))
# Compile the model.
clustering_learner.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001),
    loss=losses,
)

# Begin training the model.
clustering_learner.fit(x=inputs, y=labels, batch_size=512, epochs=50)
```

<div class="k-default-codeblock">
```
Epoch 1/50
118/118 [==============================] - 41s 236ms/step - loss: 0.6638 - similarity_loss: 0.6631 - clustering_loss: 7.1000e-04
Epoch 2/50
118/118 [==============================] - 25s 209ms/step - loss: 0.6468 - similarity_loss: 0.6438 - clustering_loss: 0.0030
Epoch 3/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6348 - similarity_loss: 0.6303 - clustering_loss: 0.0046
Epoch 4/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6279 - similarity_loss: 0.6227 - clustering_loss: 0.0052
Epoch 5/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6235 - similarity_loss: 0.6177 - clustering_loss: 0.0058
Epoch 6/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6204 - similarity_loss: 0.6139 - clustering_loss: 0.0065
Epoch 7/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6188 - similarity_loss: 0.6113 - clustering_loss: 0.0076
Epoch 8/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6165 - similarity_loss: 0.6093 - clustering_loss: 0.0072
Epoch 9/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6151 - similarity_loss: 0.6077 - clustering_loss: 0.0074
Epoch 10/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6143 - similarity_loss: 0.6061 - clustering_loss: 0.0082
Epoch 11/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6133 - similarity_loss: 0.6050 - clustering_loss: 0.0083
Epoch 12/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6125 - similarity_loss: 0.6040 - clustering_loss: 0.0085
Epoch 13/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6115 - similarity_loss: 0.6033 - clustering_loss: 0.0082
Epoch 14/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6103 - similarity_loss: 0.6023 - clustering_loss: 0.0080
Epoch 15/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6103 - similarity_loss: 0.6017 - clustering_loss: 0.0086
Epoch 16/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6096 - similarity_loss: 0.6012 - clustering_loss: 0.0084
Epoch 17/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6090 - similarity_loss: 0.6006 - clustering_loss: 0.0084
Epoch 18/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6085 - similarity_loss: 0.6001 - clustering_loss: 0.0084
Epoch 19/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6083 - similarity_loss: 0.5997 - clustering_loss: 0.0086
Epoch 20/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6084 - similarity_loss: 0.5993 - clustering_loss: 0.0090
Epoch 21/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6081 - similarity_loss: 0.5990 - clustering_loss: 0.0092
Epoch 22/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6080 - similarity_loss: 0.5986 - clustering_loss: 0.0094
Epoch 23/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6071 - similarity_loss: 0.5985 - clustering_loss: 0.0086
Epoch 24/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6069 - similarity_loss: 0.5982 - clustering_loss: 0.0088
Epoch 25/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6071 - similarity_loss: 0.5977 - clustering_loss: 0.0094
Epoch 26/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6068 - similarity_loss: 0.5974 - clustering_loss: 0.0093
Epoch 27/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6057 - similarity_loss: 0.5971 - clustering_loss: 0.0086
Epoch 28/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6064 - similarity_loss: 0.5969 - clustering_loss: 0.0096
Epoch 29/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6063 - similarity_loss: 0.5971 - clustering_loss: 0.0092
Epoch 30/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6063 - similarity_loss: 0.5967 - clustering_loss: 0.0096
Epoch 31/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6061 - similarity_loss: 0.5966 - clustering_loss: 0.0095
Epoch 32/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6055 - similarity_loss: 0.5964 - clustering_loss: 0.0091
Epoch 33/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6051 - similarity_loss: 0.5960 - clustering_loss: 0.0091
Epoch 34/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6053 - similarity_loss: 0.5959 - clustering_loss: 0.0094
Epoch 35/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6048 - similarity_loss: 0.5960 - clustering_loss: 0.0088
Epoch 36/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6050 - similarity_loss: 0.5956 - clustering_loss: 0.0093
Epoch 37/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6047 - similarity_loss: 0.5955 - clustering_loss: 0.0092
Epoch 38/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6050 - similarity_loss: 0.5954 - clustering_loss: 0.0095
Epoch 39/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6044 - similarity_loss: 0.5952 - clustering_loss: 0.0091
Epoch 40/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6048 - similarity_loss: 0.5949 - clustering_loss: 0.0100
Epoch 41/50
118/118 [==============================] - 25s 211ms/step - loss: 0.6047 - similarity_loss: 0.5951 - clustering_loss: 0.0096
Epoch 42/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6045 - similarity_loss: 0.5950 - clustering_loss: 0.0096
Epoch 43/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6042 - similarity_loss: 0.5947 - clustering_loss: 0.0095
Epoch 44/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6050 - similarity_loss: 0.5949 - clustering_loss: 0.0100
Epoch 45/50
118/118 [==============================] - 25s 213ms/step - loss: 0.6037 - similarity_loss: 0.5947 - clustering_loss: 0.0090
Epoch 46/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6039 - similarity_loss: 0.5946 - clustering_loss: 0.0093
Epoch 47/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6041 - similarity_loss: 0.5945 - clustering_loss: 0.0096
Epoch 48/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6045 - similarity_loss: 0.5945 - clustering_loss: 0.0100
Epoch 49/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6039 - similarity_loss: 0.5944 - clustering_loss: 0.0095
Epoch 50/50
118/118 [==============================] - 25s 212ms/step - loss: 0.6039 - similarity_loss: 0.5943 - clustering_loss: 0.0097

<keras.callbacks.History at 0x7fe89414bfd0>

```
</div>
Plot training loss


```python
plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
```


![png](/img/examples/vision/semantic_image_clustering/semantic_image_clustering_45_0.png)


---
## Cluster analysis

### Assign images to clusters


```python
# Get the cluster probability distribution of the input images.
clustering_probs = clustering_model.predict(x_data, batch_size=batch_size, verbose=1)
# Get the cluster of the highest probability.
cluster_assignments = tf.math.argmax(clustering_probs, axis=-1).numpy()
# Store the clustering confidence.
# Images with the highest clustering confidence are considered the 'prototypes'
# of the clusters.
cluster_confidence = tf.math.reduce_max(clustering_probs, axis=-1).numpy()
```

<div class="k-default-codeblock">
```
120/120 [==============================] - 5s 35ms/step

```
</div>
Let's compute the cluster sizes


```python
clusters = defaultdict(list)
for idx, c in enumerate(cluster_assignments):
    clusters[c].append((idx, cluster_confidence[idx]))

for c in range(num_clusters):
    print("cluster", c, ":", len(clusters[c]))
```

<div class="k-default-codeblock">
```
cluster 0 : 3984
cluster 1 : 2029
cluster 2 : 2400
cluster 3 : 1851
cluster 4 : 2537
cluster 5 : 4901
cluster 6 : 2832
cluster 7 : 4165
cluster 8 : 2370
cluster 9 : 4054
cluster 10 : 3588
cluster 11 : 1469
cluster 12 : 3497
cluster 13 : 3030
cluster 14 : 2266
cluster 15 : 4296
cluster 16 : 2329
cluster 17 : 3335
cluster 18 : 1664
cluster 19 : 3403

```
</div>
Notice that the clusters have roughly balanced sizes.

### Visualize cluster images

Display the *prototypes*—instances with the highest clustering confidence—of each cluster:


```python
num_images = 8
plt.figure(figsize=(15, 15))
position = 1
for c in range(num_clusters):
    cluster_instances = sorted(clusters[c], key=lambda kv: kv[1], reverse=True)

    for j in range(num_images):
        image_idx = cluster_instances[j][0]
        plt.subplot(num_clusters, num_images, position)
        plt.imshow(x_data[image_idx].astype("uint8"))
        plt.title(classes[y_data[image_idx][0]])
        plt.axis("off")
        position += 1
```


![png](/img/examples/vision/semantic_image_clustering/semantic_image_clustering_53_0.png)


### Compute clustering accuracy

First, we assign a label for each cluster based on the majority label of its images.
Then, we compute the accuracy of each cluster by dividing the number of image with the
majority label by the size of the cluster.


```python
cluster_label_counts = dict()

for c in range(num_clusters):
    cluster_label_counts[c] = [0] * num_classes
    instances = clusters[c]
    for i, _ in instances:
        cluster_label_counts[c][y_data[i][0]] += 1

    cluster_label_idx = np.argmax(cluster_label_counts[c])
    correct_count = np.max(cluster_label_counts[c])
    cluster_size = len(clusters[c])
    accuracy = (
        np.round((correct_count / cluster_size) * 100, 2) if cluster_size > 0 else 0
    )
    cluster_label = classes[cluster_label_idx]
    print("cluster", c, "label is:", cluster_label, " -  accuracy:", accuracy, "%")
```

<div class="k-default-codeblock">
```
cluster 0 label is: frog  -  accuracy: 25.13 %
cluster 1 label is: bird  -  accuracy: 25.78 %
cluster 2 label is: dog  -  accuracy: 23.17 %
cluster 3 label is: bird  -  accuracy: 19.72 %
cluster 4 label is: ship  -  accuracy: 30.15 %
cluster 5 label is: truck  -  accuracy: 21.93 %
cluster 6 label is: airplane  -  accuracy: 34.82 %
cluster 7 label is: cat  -  accuracy: 16.69 %
cluster 8 label is: deer  -  accuracy: 24.47 %
cluster 9 label is: dog  -  accuracy: 19.26 %
cluster 10 label is: airplane  -  accuracy: 30.96 %
cluster 11 label is: bird  -  accuracy: 26.89 %
cluster 12 label is: horse  -  accuracy: 23.39 %
cluster 13 label is: automobile  -  accuracy: 29.9 %
cluster 14 label is: ship  -  accuracy: 27.23 %
cluster 15 label is: frog  -  accuracy: 21.76 %
cluster 16 label is: frog  -  accuracy: 28.98 %
cluster 17 label is: truck  -  accuracy: 33.13 %
cluster 18 label is: deer  -  accuracy: 18.51 %
cluster 19 label is: airplane  -  accuracy: 25.21 %

```
</div>
---
## Conclusion

To improve the accuracy results, you can: 1) increase the number
of epochs in the representation learning and the clustering phases; 2)
allow the encoder weights to be tuned during the clustering phase; and 3) perform a final
fine-tuning step through self-labeling, as described in the [original SCAN paper](https://arxiv.org/abs/2005.12320).
Note that unsupervised image clustering techniques are not expected to outperform the accuracy
of supervised image classification techniques, rather showing that they can learn the semantics
of the images and group them into clusters that are similar to their original classes.

**Example available on HuggingFace**
| Trained Model | Demo |
| :--: | :--: |
| [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Semantic%20Image%20Clustering-black.svg)](https://huggingface.co/keras-io/semantic-image-clustering) | [![Generic badge](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Semantic%20Image%20Clustering-black.svg)](https://huggingface.co/spaces/keras-io/semantic-image-clustering) |

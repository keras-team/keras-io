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
        layers.RandomRotation(
            factor=0.15, fill_mode="nearest"
        ),
        layers.RandomZoom(
            height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
        )
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
        super(RepresentationLearner, self).__init__(**kwargs)
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
118/118 [==============================] - 29s 135ms/step - loss: 41.1135
Epoch 2/50
118/118 [==============================] - 15s 125ms/step - loss: 11.7141
Epoch 3/50
118/118 [==============================] - 15s 125ms/step - loss: 11.1728
Epoch 4/50
118/118 [==============================] - 15s 125ms/step - loss: 10.9717
Epoch 5/50
118/118 [==============================] - 15s 125ms/step - loss: 10.8574
Epoch 6/50
118/118 [==============================] - 15s 125ms/step - loss: 10.9496
Epoch 7/50
118/118 [==============================] - 15s 124ms/step - loss: 10.7493
Epoch 8/50
118/118 [==============================] - 15s 124ms/step - loss: 10.5979
Epoch 9/50
118/118 [==============================] - 15s 124ms/step - loss: 10.4613
Epoch 10/50
118/118 [==============================] - 15s 125ms/step - loss: 10.2900
Epoch 11/50
118/118 [==============================] - 15s 124ms/step - loss: 10.1303
Epoch 12/50
118/118 [==============================] - 15s 124ms/step - loss: 9.9608
Epoch 13/50
118/118 [==============================] - 15s 124ms/step - loss: 9.7788
Epoch 14/50
118/118 [==============================] - 15s 124ms/step - loss: 9.5830
Epoch 15/50
118/118 [==============================] - 15s 124ms/step - loss: 9.4038
Epoch 16/50
118/118 [==============================] - 15s 124ms/step - loss: 9.1887
Epoch 17/50
118/118 [==============================] - 15s 124ms/step - loss: 9.0000
Epoch 18/50
118/118 [==============================] - 15s 124ms/step - loss: 8.7764
Epoch 19/50
118/118 [==============================] - 15s 124ms/step - loss: 8.5784
Epoch 20/50
118/118 [==============================] - 15s 124ms/step - loss: 8.3592
Epoch 21/50
118/118 [==============================] - 15s 124ms/step - loss: 8.2545
Epoch 22/50
118/118 [==============================] - 15s 124ms/step - loss: 8.1171
Epoch 23/50
118/118 [==============================] - 15s 124ms/step - loss: 7.9598
Epoch 24/50
118/118 [==============================] - 15s 124ms/step - loss: 7.8623
Epoch 25/50
118/118 [==============================] - 15s 124ms/step - loss: 7.7169
Epoch 26/50
118/118 [==============================] - 15s 124ms/step - loss: 7.5100
Epoch 27/50
118/118 [==============================] - 15s 124ms/step - loss: 7.5887
Epoch 28/50
118/118 [==============================] - 15s 124ms/step - loss: 7.3511
Epoch 29/50
118/118 [==============================] - 15s 124ms/step - loss: 7.1647
Epoch 30/50
118/118 [==============================] - 15s 124ms/step - loss: 7.1549
Epoch 31/50
118/118 [==============================] - 15s 124ms/step - loss: 7.0462
Epoch 32/50
118/118 [==============================] - 15s 124ms/step - loss: 6.8149
Epoch 33/50
118/118 [==============================] - 15s 124ms/step - loss: 6.6954
Epoch 34/50
118/118 [==============================] - 15s 124ms/step - loss: 6.5354
Epoch 35/50
118/118 [==============================] - 15s 124ms/step - loss: 6.3982
Epoch 36/50
118/118 [==============================] - 15s 124ms/step - loss: 6.4175
Epoch 37/50
118/118 [==============================] - 15s 124ms/step - loss: 6.3820
Epoch 38/50
118/118 [==============================] - 15s 124ms/step - loss: 6.2560
Epoch 39/50
118/118 [==============================] - 15s 124ms/step - loss: 6.1237
Epoch 40/50
118/118 [==============================] - 15s 124ms/step - loss: 6.0485
Epoch 41/50
118/118 [==============================] - 15s 124ms/step - loss: 5.8846
Epoch 42/50
118/118 [==============================] - 15s 124ms/step - loss: 5.7548
Epoch 43/50
118/118 [==============================] - 15s 124ms/step - loss: 6.0794
Epoch 44/50
118/118 [==============================] - 15s 124ms/step - loss: 5.9023
Epoch 45/50
118/118 [==============================] - 15s 124ms/step - loss: 5.9548
Epoch 46/50
118/118 [==============================] - 15s 124ms/step - loss: 6.0809
Epoch 47/50
118/118 [==============================] - 15s 124ms/step - loss: 5.6123
Epoch 48/50
118/118 [==============================] - 15s 124ms/step - loss: 5.5667
Epoch 49/50
118/118 [==============================] - 15s 124ms/step - loss: 5.4573
Epoch 50/50
118/118 [==============================] - 15s 124ms/step - loss: 5.4597

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
120/120 [==============================] - 4s 18ms/step

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 120/120 [00:00<00:00, 304.24it/s]

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
        super(ClustersConsistencyLoss, self).__init__()

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
        super(ClustersEntropyLoss, self).__init__()
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
118/118 [==============================] - 20s 95ms/step - loss: 0.6655 - similarity_loss: 0.6642 - clustering_loss: 0.0013
Epoch 2/50
118/118 [==============================] - 10s 86ms/step - loss: 0.6361 - similarity_loss: 0.6325 - clustering_loss: 0.0036
Epoch 3/50
118/118 [==============================] - 10s 85ms/step - loss: 0.6129 - similarity_loss: 0.6070 - clustering_loss: 0.0059
Epoch 4/50
118/118 [==============================] - 10s 85ms/step - loss: 0.6005 - similarity_loss: 0.5930 - clustering_loss: 0.0075
Epoch 5/50
118/118 [==============================] - 10s 85ms/step - loss: 0.5923 - similarity_loss: 0.5849 - clustering_loss: 0.0074
Epoch 6/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5879 - similarity_loss: 0.5795 - clustering_loss: 0.0084
Epoch 7/50
118/118 [==============================] - 10s 85ms/step - loss: 0.5841 - similarity_loss: 0.5754 - clustering_loss: 0.0087
Epoch 8/50
118/118 [==============================] - 10s 85ms/step - loss: 0.5817 - similarity_loss: 0.5733 - clustering_loss: 0.0084
Epoch 9/50
118/118 [==============================] - 10s 85ms/step - loss: 0.5811 - similarity_loss: 0.5717 - clustering_loss: 0.0094
Epoch 10/50
118/118 [==============================] - 10s 85ms/step - loss: 0.5797 - similarity_loss: 0.5697 - clustering_loss: 0.0100
Epoch 11/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5767 - similarity_loss: 0.5676 - clustering_loss: 0.0091
Epoch 12/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5771 - similarity_loss: 0.5667 - clustering_loss: 0.0104
Epoch 13/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5755 - similarity_loss: 0.5661 - clustering_loss: 0.0094
Epoch 14/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5746 - similarity_loss: 0.5653 - clustering_loss: 0.0093
Epoch 15/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5743 - similarity_loss: 0.5640 - clustering_loss: 0.0103
Epoch 16/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5738 - similarity_loss: 0.5636 - clustering_loss: 0.0102
Epoch 17/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5732 - similarity_loss: 0.5627 - clustering_loss: 0.0106
Epoch 18/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5723 - similarity_loss: 0.5621 - clustering_loss: 0.0102
Epoch 19/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5711 - similarity_loss: 0.5615 - clustering_loss: 0.0096
Epoch 20/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5693 - similarity_loss: 0.5596 - clustering_loss: 0.0097
Epoch 21/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5699 - similarity_loss: 0.5600 - clustering_loss: 0.0099
Epoch 22/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5694 - similarity_loss: 0.5592 - clustering_loss: 0.0102
Epoch 23/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5703 - similarity_loss: 0.5595 - clustering_loss: 0.0108
Epoch 24/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5687 - similarity_loss: 0.5587 - clustering_loss: 0.0101
Epoch 25/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5688 - similarity_loss: 0.5585 - clustering_loss: 0.0103
Epoch 26/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5690 - similarity_loss: 0.5583 - clustering_loss: 0.0108
Epoch 27/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5679 - similarity_loss: 0.5572 - clustering_loss: 0.0107
Epoch 28/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5681 - similarity_loss: 0.5573 - clustering_loss: 0.0108
Epoch 29/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5682 - similarity_loss: 0.5572 - clustering_loss: 0.0111
Epoch 30/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5675 - similarity_loss: 0.5571 - clustering_loss: 0.0104
Epoch 31/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5679 - similarity_loss: 0.5562 - clustering_loss: 0.0116
Epoch 32/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5663 - similarity_loss: 0.5554 - clustering_loss: 0.0109
Epoch 33/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5665 - similarity_loss: 0.5556 - clustering_loss: 0.0109
Epoch 34/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5679 - similarity_loss: 0.5568 - clustering_loss: 0.0111
Epoch 35/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5680 - similarity_loss: 0.5563 - clustering_loss: 0.0117
Epoch 36/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5665 - similarity_loss: 0.5553 - clustering_loss: 0.0112
Epoch 37/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5674 - similarity_loss: 0.5556 - clustering_loss: 0.0118
Epoch 38/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5648 - similarity_loss: 0.5543 - clustering_loss: 0.0105
Epoch 39/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5653 - similarity_loss: 0.5549 - clustering_loss: 0.0103
Epoch 40/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5656 - similarity_loss: 0.5544 - clustering_loss: 0.0113
Epoch 41/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5644 - similarity_loss: 0.5542 - clustering_loss: 0.0102
Epoch 42/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5658 - similarity_loss: 0.5540 - clustering_loss: 0.0118
Epoch 43/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5655 - similarity_loss: 0.5539 - clustering_loss: 0.0116
Epoch 44/50
118/118 [==============================] - 10s 87ms/step - loss: 0.5662 - similarity_loss: 0.5543 - clustering_loss: 0.0119
Epoch 45/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5651 - similarity_loss: 0.5537 - clustering_loss: 0.0114
Epoch 46/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5635 - similarity_loss: 0.5534 - clustering_loss: 0.0101
Epoch 47/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5633 - similarity_loss: 0.5529 - clustering_loss: 0.0103
Epoch 48/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5643 - similarity_loss: 0.5526 - clustering_loss: 0.0117
Epoch 49/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5653 - similarity_loss: 0.5532 - clustering_loss: 0.0121
Epoch 50/50
118/118 [==============================] - 10s 86ms/step - loss: 0.5641 - similarity_loss: 0.5525 - clustering_loss: 0.0117

<tensorflow.python.keras.callbacks.History at 0x7f1da373ea10>

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
120/120 [==============================] - 3s 20ms/step

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
cluster 0 : 4132
cluster 1 : 4057
cluster 2 : 1713
cluster 3 : 2801
cluster 4 : 2511
cluster 5 : 2655
cluster 6 : 2517
cluster 7 : 4493
cluster 8 : 3687
cluster 9 : 1716
cluster 10 : 3397
cluster 11 : 3606
cluster 12 : 3325
cluster 13 : 4010
cluster 14 : 2188
cluster 15 : 3278
cluster 16 : 1902
cluster 17 : 1858
cluster 18 : 3828
cluster 19 : 2326

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
cluster 0 label is: frog  -  accuracy: 23.11 %
cluster 1 label is: truck  -  accuracy: 23.56 %
cluster 2 label is: bird  -  accuracy: 29.01 %
cluster 3 label is: dog  -  accuracy: 16.67 %
cluster 4 label is: truck  -  accuracy: 27.8 %
cluster 5 label is: ship  -  accuracy: 36.91 %
cluster 6 label is: deer  -  accuracy: 27.89 %
cluster 7 label is: dog  -  accuracy: 23.84 %
cluster 8 label is: airplane  -  accuracy: 21.7 %
cluster 9 label is: bird  -  accuracy: 22.38 %
cluster 10 label is: automobile  -  accuracy: 24.76 %
cluster 11 label is: automobile  -  accuracy: 24.15 %
cluster 12 label is: cat  -  accuracy: 17.44 %
cluster 13 label is: truck  -  accuracy: 23.44 %
cluster 14 label is: ship  -  accuracy: 31.67 %
cluster 15 label is: airplane  -  accuracy: 41.06 %
cluster 16 label is: deer  -  accuracy: 22.77 %
cluster 17 label is: airplane  -  accuracy: 15.18 %
cluster 18 label is: frog  -  accuracy: 33.31 %
cluster 19 label is: deer  -  accuracy: 18.7 %

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

"""
Title: Siamese Network
Author: [hazemessamm](https://twitter.com/hazemessamm)
Date created: 2021/03/13
Last modified: 2021/03/16
Description: Siamese network with custom data generator and training loop.
"""
"""
### Setup
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from tensorflow.keras import losses, optimizers
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import applications
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import Sequence

"""
## Siamese Network

[Siamese Network](https://en.wikipedia.org/wiki/Siamese_neural_network) is used to solve
many problems like detecting question duplicates, face recognition by comparing the
similarity of the inputs by comparing their feature vectors.

First we need to have a dataset that contains 3 Images, 2 are similar and 1 is different,
they are called Anchor image, Positive Image and Negative image respectively, we need to
tell the network that the anchor image and the positive image are similar, we also need
to tell it that the anchor image and the negative image are NOT similar, we can do that
by the Triplet Loss Function.

Triplet Loss function:

L(Anchor, Positive, Negative) = max((distance(f(Anchor), f(Positive)) -
distance(f(Anchor), f(Negative)))**2, 0.0)

Note that the weights are shared which mean that we are only using one model for
prediction and training

You can find the dataset here:
https://drive.google.com/drive/folders/1qQJHA5m-vLMAkBfWEWgGW9n61gC_orHl

Also more info found here: https://sites.google.com/view/totally-looks-like-dataset

Image from:
https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942


![1_0E9104t29iMBmtvq7G1G6Q.png](attachment:1_0E9104t29iMBmtvq7G1G6Q.png)
"""

"""
First we get the paths of the datasets in siamese networks we usually have two folders
each folder has images and every image has a corresponding similar picture in the other
folder.
"""

"""
## Preparing data
"""

dataset = os.path.join("/mnt/media", "TotallyLooksLikeDataset")
anchor_images_path = os.path.join(dataset, "left")
positive_images_path = os.path.join(dataset, "right")
target_shape = (200, 200)

"""
## SiameseDatasetGenerator
"""

"""
This class inherits from Sequence which is used to generate images for training, the
reason of using a generator is that there are datasets which contain a lot of high
resolution images and we cannot load all of them in our memory so we just generate
batches of them while training we inherit it so we can use it in training

1) We override the __len__ method by returning our number of batches so keras can know
how many batches available.
2) We override __getitem__ method so we can access any index of an array
"""

"""
### Negative images:
"""

"""
Negative images are just random images we sample from our dataset. every example should
contain 3 images (Anchor, Positive and Negative). The negative image should NOT be the
same as the Anchor or the Positive images, We use a set() that stores the names of the
anchor and positive images so when we sample the negative images we avoid getting any
image that exist in the set()
"""

"""
### Batch shuffle:
"""

"""
We need to shuffle the batch so we can have random examples
"""

"""
### Image preprocessing:
"""

"""
After creating a list of paths for Anchor images, positive images, negative images we
pass these lists to the preprocess_img()
because we need to load the image given the path we have and we need to convert it into
tensor by using img_to_array()
"""


class SiameseDatasetGenerator(Sequence):
    def __init__(
        self,
        anchor_images_path,
        positive_images_path,
        target_shape,
        batch_size=128,
        shuffle=True,
    ):
        self.anchor_images_path = (
            anchor_images_path  # store the path of the anchor images
        )
        self.positive_images_path = (
            positive_images_path  # store the path of the positive images
        )
        self.target_shape = target_shape  ##store image shape
        # list the contents (images) of the specified directory
        self.anchor_images = np.array(os.listdir(positive_images_path))
        self.positive_images = np.array(os.listdir(positive_images_path))
        self.batch_size = batch_size
        self.num_examples = len(self.anchor_images)
        self.num_batches = self.num_examples // batch_size
        self.shuffle = shuffle

    """
    we use __len__ method that is called
    to get the length of the batches
    it is called when we call len()
    """

    def __len__(self):
        return self.num_batches

    """
    this method allows us to get batches when we
    access the instance the same way we access a list
    e.g. dataset[0] will call __getitem__(index=0)
    """

    def __getitem__(self, index):
        current_batch = index * self.batch_size
        # here we get batches of data by using slicing
        anchor_imgs = self.anchor_images[
            current_batch : current_batch + self.batch_size
        ]
        positive_imgs = self.positive_images[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        # store the loaded images to avoid reloading them in negative images
        # we store them in a set for faster access
        loaded_examples = set([i for i in anchor_imgs])

        negative_imgs = np.array(
            self.get_negative_imgs(
                from_anchor_dir=random.choice([True, False]),
                loaded_examples=loaded_examples,
            )
        )
        anchor_imgs = np.array(
            [os.path.join(self.anchor_images_path + "/", img) for img in anchor_imgs]
        )
        positive_imgs = np.array(
            [
                os.path.join(self.positive_images_path + "/", img)
                for img in positive_imgs
            ]
        )

        if self.shuffle:
            # create a list of random numbers to use it when we shuffle the batches
            random_shuffle = random.choices(
                [*range(0, len(anchor_imgs))], k=len(anchor_imgs)
            )
            anchor_imgs = anchor_imgs[random_shuffle]
            positive_imgs = positive_imgs[random_shuffle]
            negative_imgs = negative_imgs[random_shuffle]

        anchor_imgs = self.preprocess_img(anchor_imgs)
        positive_imgs = self.preprocess_img(positive_imgs)
        negative_imgs = self.preprocess_img(negative_imgs)

        # here if the batch size equal one we just convert the images into numpy
        # and expand the dimension of this batch by adding 1 in the first axis
        if self.batch_size == 1:
            return np.expand_dims(
                np.array([anchor_imgs, positive_imgs, negative_imgs]), axis=0
            )
        # Add the batch_size dimension in the first axis by using permute()
        return tf.keras.backend.permute_dimensions(
            np.array([anchor_imgs, positive_imgs, negative_imgs]), (1, 0, 2, 3, 4)
        )

    def get_negative_imgs(self, from_anchor_dir=True, loaded_examples={}):
        # load the negative_imgs by randomly loading it from anchor or positive images
        negative_imgs = []
        if from_anchor_dir:
            negative_imgs = random.choices(
                [img for img in self.anchor_images if img not in loaded_examples],
                k=self.batch_size,
            )
            negative_imgs = [
                os.path.join(self.anchor_images_path + "/", img)
                for img in negative_imgs
            ]
        else:
            negative_imgs = random.choices(
                [img for img in self.positive_images if img not in loaded_examples],
                k=self.batch_size,
            )
            negative_imgs = [
                os.path.join(self.positive_images_path + "/", img)
                for img in negative_imgs
            ]
        return negative_imgs

    def preprocess_img(self, imgs):
        output = []
        for img_path in imgs:
            img = preprocessing.image.load_img(img_path, target_size=self.target_shape)
            img = preprocessing.image.img_to_array(img)
            output.append(img)
        if len(output) == 1:
            return output[0]
        return tuple(output)


dataset = SiameseDatasetGenerator(
    anchor_images_path, positive_images_path, target_shape, 32, False
)

# this function just visalize each random 3 images (anchor, positive, negative)
def visualize():
    example = dataset[random.randint(0, dataset.batch_size)]
    img1, img2, img3 = (
        preprocessing.image.array_to_img(example[:, 0][0]),
        preprocessing.image.array_to_img(example[:, 1][0]),
        preprocessing.image.array_to_img(example[:, 2][0]),
    )
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(img3)
    plt.savefig("test.png")


visualize()  # as you see we have two similar images and one different image

"""
## Loading a pre-trained model
"""

"""
Here we use ResNet50 architecture, we use "imagenet" weights, also we pass the image shape
Note that include_top means that we do NOT want the top layers
"""

base_cnn = applications.ResNet50(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

"""
## Fine Tuning
"""

"""
Here we fine tune the ResNet50 we freeze all layers that exist before "conv5_block1_out"
layer, starting from "conv5_block2_2_relu" layer we unfreeze all the layers so we can
just train these layers
"""

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

"""
## Adding top layers
"""

"""
Here we customize the model by adding Dense layers and Batch Normalization layers. we
start with the image input then we pass the input to the base_cnn then we flatten it.
Finally we pass each layer as an input to the next layer the output layer is just a dense
layer which will act as an embedding for our images.
"""

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="SiameseNetwork")

embedding.summary()

"""
## Model for training
"""

"""
This model is just used for training we pass to it three input batches (anchor images,
positive images, negative images) and the output will be the output of the model we
defined above, it will be 1 output for each input.
"""

anchor_input = layers.Input(shape=target_shape + (3,))
positive_input = layers.Input(shape=target_shape + (3,))
negative_input = layers.Input(shape=target_shape + (3,))

anchor_output = embedding(anchor_input)
positive_output = embedding(positive_input)
negative_output = embedding(negative_input)

training_model = Model(
    [anchor_input, positive_input, negative_input],
    {
        "anchor_embedding": anchor_output,
        "positive_embedding": positive_output,
        "negative_embedding": negative_output,
    },
)


"""
## Cosine Similarity Layer
"""

"""

This layer just computes how similar to feature vectors are by computing it using the
Cosine Similarity
We override the call method and implement our own call method.

Check out https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity

We return the negative of the loss because we just need to know how similar they are we
do NOT need to know the loss
"""


class CosineDistance(layers.Layer):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def call(self, img1, img2):
        return -losses.CosineSimilarity(reduction=losses.Reduction.NONE)(img1, img2)


"""
## Model subclassing
"""

"""
Here we customize our training process and our model.

We override the train_step() method and apply our own loss and our own training process

We also use Triplet loss function as we specified above.

Loss function explaination:

we calculate the distance between the anchor embedding and the positive embedding the
axis = -1 because we want the distance over the features of every example. We also add
alpha which act as extra margin.
"""


class SiameseModel(Model):
    def __init__(self, model, alpha=0.5):
        super(SiameseModel, self).__init__()
        self.embedding = model  # we pass the model to the class
        self.alpha = alpha

    def call(self, inputs):
        pass

    def train_step(self, data):
        # here we create a tape to record our operations so we can get the gradients
        with tf.GradientTape() as tape:
            embeddings = training_model((data[:, 0], data[:, 1], data[:, 2]))

            # Euclidean Distance between anchor and positive
            # axis=-1 so we can get distances over examples
            anchor_positive_dist = tf.reduce_sum(
                tf.square(
                    embeddings["anchor_embedding"] - embeddings["positive_embedding"]
                ),
                -1,
            )

            # Euclidean Distance between anchor and negative
            anchor_negative_dist = tf.reduce_sum(
                tf.square(
                    embeddings["anchor_embedding"] - embeddings["negative_embedding"]
                ),
                -1,
            )

            # getting the loss by subtracting the distances
            loss = anchor_positive_dist - anchor_negative_dist
            # getting the max because we don't want negative loss
            loss = tf.reduce_mean(tf.maximum(loss + self.alpha, 0.0))
        # getting the gradients [loss with respect to trainable weights]
        grads = tape.gradient(loss, training_model.trainable_weights)
        # applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(zip(grads, training_model.trainable_weights))
        return {"Loss": loss}


"""
### Training
"""

siamese_model = SiameseModel(embedding)

siamese_model.compile(optimizer=optimizers.Adam(0.0001))

siamese_model.fit(dataset, epochs=3)

"""
### Inference
"""

# here we just load from the dataset an example
# we should NOT test the performace of the model
# using training data but here we are just see how did it learn
example_prediction = dataset[3]
anchor_example = preprocessing.image.array_to_img(example_prediction[:, 0][0])
positive_example = preprocessing.image.array_to_img(example_prediction[:, 1][0])
negative_example = preprocessing.image.array_to_img(example_prediction[:, 2][0])

# here we just plotting the example that we loaded
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(anchor_example)
ax2.imshow(positive_example)
ax3.imshow(negative_example)

# we add an extra dimension (batch_size dimension) in the first axis by using expand dims.
anchor_tensor = np.expand_dims(example_prediction[:, 0][0], axis=0)
positive_tensor = np.expand_dims(example_prediction[:, 1][0], axis=0)
negative_tensor = np.expand_dims(example_prediction[:, 2][0], axis=0)

anchor_embedding, positive_embedding = (
    embedding(anchor_tensor),
    embedding(positive_tensor),
)
positive_similarity = CosineDistance()(anchor_embedding, positive_embedding)
print("Similarity between similar images:", positive_similarity)

anchor_embedding, negative_embedding = (
    embedding(anchor_tensor),
    embedding(negative_tensor),
)
negative_similarity = CosineDistance()(anchor_embedding, negative_embedding)
print("Similarity between dissimilar images:", negative_similarity)

"""
### Key Takeaways

1) You can create your custom data generator by creating a class that inherits from
tf.keras.utils.Sequence, as we saw, this is really helpful if we want to generate data in
different forms like Anchor, Positive and negative in our case, you just need to
implement the __len__() and __getitem__(). Check out the documentation
https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

2) If you don't have the computation power to train large models like ResNet-50 or if you
don't have the time to re-write really big models you can just download it with one line,
e.g. tf.keras.applications.ResNet50().

3) Every layer has a name, this is really helpful for fine tuning, If you want to fine
tune specific layers, in our example we loop over the layers until we find specific layer
by it's name and we made it trainable, this allows the weights of this layer to change
during training

4) In our example we have only one embedding network that we need to train it but we need
3 outputs to compare them with each other, we can do that by creating a model that have 3
input layers and each input will pass through the embedding network and then we will have
3 outputs embeddings, we did that in the "Model for Training section".

5) You can name your output layers like we did in the "Model for Training section", you
just need to create a dictionary with keys as the name of your output layer and the
output layers as values.

6) We used cosine similarity to measure how to 2 output embeddings are similar to each
other.

6) You can create your custom Layers by just creating a class that inherits from
tf.keras.layers.Layer, you just need to implement the call function. check out the
documentation https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

7) If you want to have custom training loop you can create your own model class that
inherits from tf.keras.Model, you just need to override the train_step function and add
you implementation.

8) You can get your model gradients by using tf.GradientTape(), GradientTape records the
operations that you do inside it, so you can get the predictions and write your custom
loss function inside the GradientTape as we did.

9) You can get the gradients using tape.gradient(loss, model.trainable_weights), this
means that we need the gradients of the loss with respect to the model trainable weights,
where "tape" is the name of our tf.GradientTape() in our example.

10) you can just pass the gradients and the model weights to the optimizer to update them.

For more info about GradientTape check out
https://keras.io/getting_started/intro_to_keras_for_researchers/ and
https://www.tensorflow.org/api_docs/python/tf/GradientTape
"""

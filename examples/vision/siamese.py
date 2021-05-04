
"""
# Siamese Network with Contrastive Loss

Author: [Mehdi](https://github.com/s-mrb)<br>
Date created: 2021/05/02<br>
Last modified: 2020/05/03<br>
Description: Similarity Learning Using Siamese Network with Contrastive Loss
"""

"""
## Introduction

The [Siamese Network](https://papers.nips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf) introduced by Bromley and LeCun to solve signature
verification as an image matching problem is now widely used for finding the
level of similarity between two images. This algorithm evolved to have many
different variants, although the one we are going to use is Duplet Network.
A duplet network involves two sister networks for finding embeddings/encodings
and then a distance heuristic to find out how much these embeddings differ from
each other.

"""

"""
## Necessary imports
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Lambda, Input
from tensorflow.keras.layers import Conv2D, Activation,AveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from keras import backend as K

"""
## Load data
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# We change data type to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

"""
## Create pairs of images

We want our model to be able to differentiate 0 from 1,2,3, .., 9 to do so we
should teach it how each digit differs from each of the other digit, for this
purpose we would select N random images from class A (lets say class of
digit 0) and would pair it with N random images from class B (lets say class of
digit 1), and would repeat this process untill class of digit 9. In this way we
successfuly pair 0 with each other digit, now we would do the same for class of
1,2,3 upto that of 9.
"""

def make_pairs(x, y):
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]
        
        pairs += [[x1, x2]]
        labels += [1]
    
        # add a not matching example
        label2 = random.randint(0, num_classes-1)
        while label2 == label1:
            label2 = random.randint(0, num_classes-1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]
        
        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype('float32')

pairs_train, labels_train = make_pairs(x_train, y_train)
pairs_test, labels_test = make_pairs(x_test, y_test)

"""
## Create dataset
"""

# 0 index of pairs_train contain one image of pair
# and at index 1 the other image of pair is present
# we can split entire vector of pairs into two havles,
# later we can generate tf.data.Dataset using these halves
# we will do the same for labels

x1 = pairs_train[:,0]
x2 = pairs_train[:,1]
# x1.shape = (120000, 28, 28)

y1, y2 = pairs_test[:,0],pairs_test[:,1]

train_pair = tf.data.Dataset.from_tensor_slices((x1, x2))
train_label = tf.data.Dataset.from_tensor_slices(labels_train)
train_ds = tf.data.Dataset.zip((train_pair, train_label)).batch(16)

test_pair = tf.data.Dataset.from_tensor_slices((y1, y2))
test_label = tf.data.Dataset.from_tensor_slices(labels_test)
test_ds = tf.data.Dataset.zip((test_pair, test_label)).batch(16)

"""
## Visualize
"""

def visualize(dataset, to_show=10, num_col=5, predictions=None, test=False):
  num_row = to_show//num_col

  # plot images
  fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
  for images, labels in dataset.take(1):
    for i in range(to_show):
        ax = axes[i//num_col, i%num_col]
        # images[0][i][:,:,0] -> because it is (28,28,1) and imshow takes (28,28)
        # ax.imshow(tf.concat([images[0][i][:,:,0],images[1][i][:,:,0]],axis=1))

        ax.imshow(tf.concat([images[0][i],images[1][i]],axis=1))
        if test:
          ax.set_title('y:{}  |  y^:{}'.format(labels[i],predictions[i]))
        else:
          ax.set_title('Label: {}'.format(labels[i]))
  if test:
    plt.tight_layout(rect = (0,0,2,2 ), w_pad=0.0)
  else:
    plt.tight_layout()
  plt.show()

visualize(train_ds)

"""
## Model

There are two input layers, each lead to its own dense network which produces
embeddings. Lambda layer will merge them using euclidean distance and merged
layer will be fed to final network.
"""

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))



input = Input((28,28,1))
x = tf.keras.layers.BatchNormalization()(input)
x = Conv2D(4, (5,5), activation = 'tanh')(x)
x = AveragePooling2D(pool_size = (2,2))(x)
x = Conv2D(16, (5,5), activation = 'tanh')(x)
x = AveragePooling2D(pool_size = (2,2))(x)
x = Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = Dense(10, activation = 'tanh')(x)
dense = Model(input, x)


input1 = Input((28,28,1))
input2 = Input((28,28,1))

dense1 = dense(input1)
dense2 = dense(input2)

merge_layer = Lambda(euclidean_distance)([dense1,dense2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
dense_layer = Dense(1, activation="sigmoid")(normal_layer)
model = Model(inputs=[input1, input2], outputs=dense_layer)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(1-y_pred)
    margin_square = K.square(K.maximum(margin - (1-y_pred), 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

model.compile(loss = contrastive_loss, optimizer=RMSprop(), metrics=["accuracy"])

model.summary()

# Rarely it stucks at local optima, in that case just try again
model.fit(train_ds, epochs= 10)

results = model.evaluate(test_ds)
print("test loss, test acc:", results)

"""
## Visualize predictions
"""

predictions = model.predict(test_ds)

visualize(dataset=test_ds, predictions=predictions, test=True)


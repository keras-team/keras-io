"""
Title: Classification using Attention-based Deep Multiple Instance Learning (MIL).
Author: [Mohamad Jaber](https://www.linkedin.com/in/mohamadjaber1/)
Date created: 2021/08/16
Last modified: 2021/09/09
Description: MIL approach to classify bags of instances and get their individual instance score.
"""
"""
Suppose there is a locked door and several key chains with each chain containing few keys.
The goal here is to be able to know which key chain (in general) and key (in specific) can
open up the locked door.

In order to solve this task, we shall find the key that can opens the door. If the key was
correctly found, then the entire key chain will be classified as positive. If the key chain
doesn't contain any compatible key, it will be classified as negative.

In the context of MIL, each of the key chains are known as **bags**, in which each can be
labelled as positive or negative. Each key within the key chains are known as **instances**.

## Introduction

Usually for image classification tasks using **supervised learning**  algorithms, the learner
receives a single label for a single image. In case of **MIL**, the learner receives a single
label for a set of images.

The motivation behind this model is to:

- know which instance (in this case: image) has contributed the most to the class label
prediction.
- leverage from weakly labelled samples.

### MIL Classifier: Attention-based approach

The MIL classifier is modelled using neural networks. The attention mechanism as MIL pooling:

- Trainable MIL operator.
- Interpretable results using attention scores.

Using attention scores or weights could be very useful for results interpretability as it will
not merely predict the final output class but the instance scores that led to those
classification results.

Another reason why this method could be helpful is: weakly labelled data. Since labelling
individual samples is often performed manually, it is considered to be a time consuming task.
This task is also prone to errors (especially in computer vision problems).

Therefore, instead of having individual labelled data points or instances, this algorithm
receives a single label for each bag of instances. The bag is labelled positive if it contains
atleast one positive instance and negative if it does not contain any.

The attention mechanism provides insight into the contribution of each instance to the bag label.

Besides the implementation of this mechanism, some **regularization** techniques,
**ensemble averaging** (for stability) and dealing with **imbalanced data** will be covered.

## References

- [Attention-based Deep Multiple Instance Learning](https://arxiv.org/pdf/1802.04712.pdf).
- Some of attention operator code implementation was inspired from https://github.com/utayao/Atten_Deep_MIL.
- Imbalanced data [tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
by TensorFlow.
"""
"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.style.use("ggplot")

"""
## Create dataset
`mnist` and `fashion_mnist` datasets have been tested using this implementation. In this
example, I will be using `mnist`.

At first we will set up the configurations and then prepare the datasets.

### Configurations

- `POS_CLASS`: The desired class to be kept in the positive bag.
- `BAG_COUNT`: The number of training bags.
- `VAL_BAG_COUNT`: The number of validation bags.
- `BAG_SIZE`: The number of instances in a bag.
- `PLOT_SIZE`: The number of bags to plot.
- `ENSEMBLE_AVG_COUNT`: The number of models to create and get their average. (OPTIONAL:
often results in better performance - set to 1 for single model)
"""

POS_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 5

"""
### Prepare bags
Since the attention operator is a permutation-invariant operator, an instance with a
positive class label is randomly placed among the instances in the positive bag.
"""


def create_bags(input_data, input_labels, pos_class, bag_count, instance_count):

    # Set up bags.
    bags = []
    bag_labels = []

    # Normalize input data.
    input_data = np.divide(input_data, 255.0)

    # Count positive samples.
    count = 0

    # Take out the filter for specific class.
    filter_class = np.where(input_labels == pos_class)[0]

    # Assign new variables consisting of this class.
    data_pos_class = input_data[filter_class]
    labels_pos_class = input_labels[filter_class]

    # From overall data, remove this class.
    data_neg_classes = np.delete(input_data, filter_class, 0)
    labels_neg_classes = np.delete(input_labels, filter_class, 0)

    # Merge both inputs and labels to each another.
    data = np.concatenate([data_pos_class, data_neg_classes], axis=0)
    labels = np.concatenate([labels_pos_class, labels_neg_classes], axis=0)

    # Data are ordered in such a way: [pos_class... neg_classes].
    # Shuffle the data randomly.
    order = np.arange(len(data))
    np.random.shuffle(order)
    data = data[order]
    labels = labels[order]

    for _ in range(bag_count):

        # Pick a fixed size random subset of samples.
        index = np.random.choice(data.shape[0], instance_count, replace=False)
        instances_data = data[index]
        instances_labels = labels[index]

        # By default, all bags are labelled as 0.
        bag_label = 0

        # Check if there is at least a positive class in the bag.
        if pos_class in instances_labels:

            # Positive bag will be labelled as 1.
            bag_label = 1

            # Increment count by 1.
            count += 1

        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))

    print(f"Positive bags: {count}")
    print(f"Negative bags: {bag_count - count}")

    return (list(np.swapaxes(bags, 0, 1)), np.array(bag_labels))


# Load desired data.
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()

# Create training data.
train_data, train_labels = create_bags(x_train, y_train, POS_CLASS, BAG_COUNT, BAG_SIZE)

# Create validation data.
val_data, val_labels = create_bags(x_val, y_val, POS_CLASS, VAL_BAG_COUNT, BAG_SIZE)

"""
# Create and train neural networks
We will now build the attention layer, prepare some utilities, build and train the
entire model.

## Attention operator implementation

The output size of this layer is decided by the size of a single bag.

The attention mechanism uses a weighted average of instances in a bag, in which the sum
of the weights must equal to 1 (invariant of the bag size).

The weight matrices (parameters) are **w** and **v**. To include positive and negative
values, hyperbolic tangent element-wise non-linearity is utilized.

**Gated attention mechanism** can be used to deal with complex relations. Another weight
matrix, **u**, is added to the computation.
Sigmoid non-linearity is used to overcome approximately linear behavior for *x* ∈ [−1, 1]
by hyperbolic tangent non-linearity.
"""


class AttentionLayer(layers.Layer):
    def __init__(
        self,
        l_dim,
        output_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):
        self.l_dim = l_dim
        self.output_dim = output_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise TypeError("Input should be of type of list")

        # Input Shape
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.l_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.l_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.l_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = [self.compute_weights(instance) for instance in inputs]

        # such that each row summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_weights(self, instance):

        # in-case "gated mechanism" used.
        ori_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            # sigmoid(u*h_k^T).
            gate_instance = tf.math.sigmoid(
                tf.tensordot(ori_instance, self.u_weight_params, axes=1)
            )

            # element-wise multiplication.
            ac_instance = instance * gate_instance

        else:
            ac_instance = instance

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(ac_instance, self.w_weight_params, axes=1)


"""
## Visualizer tool

Plot the number of bags (given by `PLOT_SIZE`) with respect to the class.

Moreover, if activated, the class label prediction with its associated instance score
for each bag (after the model has been trained) can be seen.
"""

# Function for plotting.
def plot(data, labels, bag_class, pred=None, attention_weights=None):

    labels_re = np.array(labels).reshape(-1)

    if bag_class == "pos":
        if pred is not None:
            labels = np.where(pred.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

        else:
            labels = np.where(labels_re == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    elif bag_class == "neg":
        if pred is not None:
            labels = np.where(pred.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        else:
            labels = np.where(labels_re == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]

    else:
        print(f"There is no class as {bag_class}")
        return

    for i in range(PLOT_SIZE):
        fig = plt.figure(figsize=(8, 8))
        print(f"Bag number: {labels[i]}")
        for j in range(BAG_SIZE):
            img = bags[j][i]
            fig.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 2))
            plt.imshow(img)
        plt.show()


# Plot some of validation data bags per class.
plot(val_data, val_labels, "pos")
plot(val_data, val_labels, "neg")

"""
## Create model

First we will create some embeddings per instance, invoke the attention operator and then
use the softmax function to output the class probabilities.
"""


def create_model(instance_shape):
    def dense_layers():

        inputs, embeddings = [], []
        for _ in range(BAG_SIZE):
            inp = layers.Input(instance_shape)
            flatten = layers.Flatten()(inp)
            dense_1 = layers.Dense(128, activation="relu")(flatten)
            dense_2 = layers.Dense(64, activation="relu")(dense_1)
            inputs.append(inp)
            embeddings.append(dense_2)

        return inputs, embeddings

    # Create layers that form embeddings.
    inputs, embeddings = dense_layers()

    # Invoke the attention layer.
    alpha = AttentionLayer(
        l_dim=256,
        output_dim=1,
        kernel_regularizer=keras.regularizers.l2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    mul_layers = [layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))]

    # Concatenate layers.
    concat = layers.concatenate(mul_layers, axis=1)

    # Classification output node.
    output = layers.Dense(2, activation="softmax")(concat)

    return keras.Model(inputs, output)


"""
## Class weights

Since this kind of problem could simply turn into imbalanced data classification problem,
class weights should be considered.

Let's say there are 1000 bags. There often could be cases were ~90 % of the bags does not
contain any positive label and ~10 % does.
Such data can be referred to as **Imbalanced data**.

Using class weights, the model will tend to consider the rare class more as compared to
that of the abundant one.
"""


def calculate_weights(labels):

    # Count number of postive and negative bags.
    neg_count = len(np.where(labels == 0)[0])
    pos_count = len(np.where(labels == 1)[0])
    total_count = neg_count + pos_count

    # Build class weight dictionary.
    return {
        0: (1 / neg_count) * (total_count / 2),
        1: (1 / pos_count) * (total_count / 2),
    }


"""
## Build and train model

The model is built and trained in this section.

Some regularization techniques are considered to avoid overfitting the model which ensures
minimal generalization error.
"""


def train(train_data, train_labels, val_data, val_labels, model):

    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # Take the file name from the wrapper.
    file_path = "/tmp/best_model_weights.h5"

    # Initialize model checkpoint callback.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Initialze early stopping callback.
    # The model performance is monitored across the unseen data and stops training
    # when the generalization error cease to decrease.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )

    # Compile model.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fit model.
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=20,
        class_weight=calculate_weights(train_labels),
        batch_size=1,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0,
    )

    # Load best weights.
    model.load_weights(file_path)

    return model


# Building model(s).
instance_shape = train_data[0][0].shape
models = [create_model(instance_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

# Show single model architecture.
print(models[0].summary())

# Training model(s).
trained_models = [
    train(train_data, train_labels, val_data, val_labels, model)
    for model in tqdm(models)
]

"""
## Model evaluation

The models are in their deterministic state now and ready for evaluation.
With each model we also create an associated intermediate model to get the
weights from the attention layer.

Based on the number of models (`ENSEMBLE_AVG_COUNT`), the models predict the results
and then averaged out (equal contribution per model).
"""


def predict(data, labels, trained_models):

    # Collect info per model.
    predictions = []
    attention_weights = []
    losses = []
    accuracies = []

    for model in trained_models:

        # Predict output classes on validation data.
        pred = model.predict(data)
        predictions.append(pred)

        # Create intermediate model to get attention layer weights.
        interm_model = keras.Model(model.input, model.get_layer("alpha").output)

        # Predict attention layer weights.
        pred_interm = interm_model.predict(data)

        # Reshape list of arrays.
        model_attention_weights = np.squeeze(np.swapaxes(pred_interm, 1, 0))
        attention_weights.append(model_attention_weights)

        loss, acc = model.evaluate(data, labels, verbose=0)
        losses.append(loss)
        accuracies.append(acc)

    print(
        f"The average loss and accuracy are {np.sum(losses, axis=0) / ENSEMBLE_AVG_COUNT:.2f}"
        f" and {100 * np.sum(accuracies, axis=0) / ENSEMBLE_AVG_COUNT:.2f} % resp."
    )

    return (
        np.sum(predictions, axis=0) / ENSEMBLE_AVG_COUNT,
        np.sum(attention_weights, axis=0) / ENSEMBLE_AVG_COUNT,
    )


pred_class, atten_weights = predict(val_data, val_labels, trained_models)

# Plot some results of validation data bags per class.
plot(val_data, val_labels, "pos", pred=pred_class, attention_weights=atten_weights)
plot(val_data, val_labels, "neg", pred=pred_class, attention_weights=atten_weights)

"""
## Conclusion

From the above plot, you can notice that the weights are always summing to 1. If it is a
negative predicted bag, the weights will somehow be equally distributed. However, in a
positive predict bag, the instance which resulted to the positve labelling, will have
substantial higher attention score among that bag.

## Remarks

- If the model is overfitted, the weights will be equally distributed for all bags. Hence,
the regularization techniques are necessary.
- In the paper, the bags' sizes can differ from one bag to another. For simplicity, the
bags' sizes are fixed here.
- In order not to rely on the random initial weights of a single model, averaging ensemble
methods are considered.
"""

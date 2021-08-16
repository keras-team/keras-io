"""
Title: Classification using Attention-based Deep MIL
Author: [Mohamad Jaber](https://www.linkedin.com/in/mohamadjaber1/)
Date created: 2021/08/16
Last modified: 2021/08/16
Description: A neural network model that uses attention weights on instance-level to decide the output class.
"""
"""
## Introduction

Instead of having individual labelled instances, this neural network model receives a single label for each bag of instances.
The bag is labelled positive if it contains atleast one positive instance and negative if it does not contain any.
The MIL problem is stated as the learning Bernoulli distribution of the bag label where output probability is parametermized by the algorithm.
The attention mechanism provides insight into the contribution of each instance to the bag label.

Besides the implementation of this mechanism, some **regularization** techniques, **ensemble averaging** (for stability)
and dealing **imbalanced data** will be covered.

## References

- The paper [Attention-based Deep Multiple Instance Learning](https://arxiv.org/pdf/1802.04712.pdf) is authored by Maximilian Ilse,
Jakub M. Tomczak and Max Welling.
- Some of attention operator code implementation was inspired from https://github.com/utayao/Atten_Deep_MIL.
- Imbalanced data [tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) by TensorFlow.
"""
"""
## Setup
"""

import numpy as np
import os
from tqdm import tqdm
import tempfile
from tensorflow.python.keras.datasets import mnist, fashion_mnist
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers, regularizers
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
plt.style.use("ggplot")

"""
## Create Dataset

In this example, either of `mnist` or `fashion_mnist` data can be used.
At first we will set up the configurations, create the function for preparing them and then load the datasets that will be *used*.

### Configurations
- DATASET: Either mnist digits or fashion mnist data.
- POS_CLASS: The desired class to be kept in the positive bag.
- BAG_COUNT: The number of training bags.
- VAL_BAG_COUNT: The number of validation bags.
- BAG_SIZE: The number of instances in a bag.
- PLOT_SIZE: The number of bags to plot.
- ENSEMBLE_AVG_COUNT: The number of models to create and get their average. (OPTIONAL: often results in better performance - set to 1 for single model)
"""

DATASET = {"mnist": mnist, "fashion_mnist": fashion_mnist}
POS_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 5

"""
### Prepare Bags
This function returns the bags along with their corresponding labels.

Since the attention operator is a permutation-invariant operator,
the desired/positive class is randomly placed among the instances
in the positive bag.
"""


def create_bags(x, y, pos_class, bag_count, instance_count):

    # Set up bags.
    bags = []
    bag_labels = []

    # Normalize input data.
    x = np.divide(x, 255.0)

    # Count positive samples.
    count = 0

    # Take out the filter for specific class.
    filter_class = np.where(y == pos_class)[0]

    # Assign new variables consisting of this class.
    x_pos_class = x[filter_class]
    y_pos_class = y[filter_class]

    # From overall data, remove this class.
    x_neg_classes = np.delete(x, filter_class, 0)
    y_neg_classes = np.delete(y, filter_class, 0)

    # Merge both inputs and labels to each another.
    x_merged = np.concatenate([x_pos_class, x_neg_classes], axis=0)
    y_merged = np.concatenate([y_pos_class, y_neg_classes], axis=0)

    # Data are ordered in such a way: [pos_class... neg_classes]
    # Shuffle the data randomly.
    order = np.arange(len(x_merged))
    np.random.shuffle(order)
    x = x_merged[order]
    y = y_merged[order]

    for _ in range(bag_count):

        # Pick a fixed size random subset of samples.
        index = np.random.choice(x.shape[0], instance_count, replace=False)
        images = x[index]
        labels = y[index]

        # By default, all bags are labelled as 0.
        label = 0

        # Check if there is atleast a positive class in the bag.
        if pos_class in labels:

            # Positive bag will be labelled as 1.
            label = 1

            # Increment count by 1.
            count += 1

        bags.append(images)
        bag_labels.append(np.array([label]))

    print(
        "Positive bags: " + str(count),
        "Negative bags: " + str(bag_count - count),
    )
    return (list(np.swapaxes(bags, 0, 1)), np.array(bag_labels))


# Load data.
(x, y), (x_val, y_val) = DATASET["mnist"].load_data()

# Create training data.
train_data, train_labels = create_bags(x, y, POS_CLASS, BAG_COUNT, BAG_SIZE)

# Create validation data.
val_data, val_labels = create_bags(x_val, y_val, POS_CLASS, VAL_BAG_COUNT, BAG_SIZE)

"""
# Create and Train Neural Networks.
In this section we will build the attention layer, write some function for training and visualizing, build and train the entire model.

## Attention Operator Implementation

The output size of this layer is decided by the size of a single bag.

The attention mechanism uses a weighted average of instances in a bag, in which the sum of the weights must equal to 1 (invariant of the bag size).

The weight matrices (parameters) are **w** and **V**. To include positive and negative values, hyperbolic tangent element-wise non-linearity is utilized.

**Gated attention mechanism** can be used to deal with complex relations. Another weight matrix, **U**, is added to the computation.
Sigmoid non-linearity is used to overcome approximately linear behavior for *x* ∈ [−1, 1] by hpyerbolic tangent non-linearity.
"""


class AttentionLayer(layers.Layer):
    def __init__(
        self,
        L_dim,
        output_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_bias=True,
        use_gated=False,
        **kwargs
    ):
        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Input Shape
        # List of 2D tensors with shape: (batch_size, input_dim)
        input_dim = input_shape[0][1]

        self.V = self.add_weight(
            shape=(input_dim, self.L_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w = self.add_weight(
            shape=(self.L_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.U = self.add_weight(
                shape=(input_dim, self.L_dim),
                initializer=self.u_init,
                name="U",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.U = None

        self.input_built = True

    def call(self, x, mask=None):
        def compute(x):

            # in-case "gated mechanism" used
            ori_x = x

            # tanh(V*h_k^T)
            # (input_dim, L_dim) e.g (N,64)
            x = K.tanh(K.dot(x, self.V))

            # for learning non-linear relations efficiently
            if self.use_gated:

                # sigmoid(U*h_k^T)
                # (input_dim, L_dim) e.g (N,64)
                gate_x = K.sigmoid(K.dot(ori_x, self.U))

                # element-wise multiplication
                ac_x = x * gate_x

            else:
                ac_x = x

            # w^T*(tanh(V*h_k^T)) / w^T*(tanh(V*h_k^T)*sigmoid(U*h_k^T))
            # (N,64) * (64, 1) = (N,1)
            soft_x = K.dot(ac_x, self.w)

            return soft_x

        # Assigning four variables from the input x
        instances = [compute(instance) for instance in x]

        # apply softmax over N
        # such that each row summation is equal to 1
        alpha = K.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_output_shape(self, input_shapes):

        assert isinstance(input_shapes, list)

        return [(shape[0], self.output_dim) for shape in input_shapes]


"""
## Visualizer Tool

Plot the number of bags (given by PLOT_SIZE) with respect to the class.

Moreover, it returns the associated instance weights for each bag (after the model has been trained).
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
        print("There is no class as {}".format(bag_class))
        return

    for i in range(PLOT_SIZE):
        fig = plt.figure(figsize=(8, 8))
        for j in range(BAG_SIZE):
            img = bags[j][i]
            fig.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            plt.imshow(img)
        print("Bag number: {}".format(labels[i]))
        if attention_weights is not None:
            print(
                "Weight for each instance: {}".format(
                    np.around(attention_weights[labels[i]], 2)
                )
            )
        plt.show()


# Plot some of validation data bags per class.
plot(val_data, val_labels, "pos")
plot(val_data, val_labels, "neg")

"""
## Create Model
This function creates the deep learning model which consists of feature extractors per instance, attention operator, and the classifier. Inner function yields embeddings from raw input data.

"""


def create_model(instance_shape):
    def dense_layers():

        inputs, embeddings = [], []
        for i in range(BAG_SIZE):
            inp = layers.Input(instance_shape)
            x = layers.Flatten()(inp)
            x = layers.Dense(126, activation="relu")(x)
            x = layers.Dense(64, activation="relu")(x)
            inputs.append(inp)
            embeddings.append(x)

        return inputs, embeddings

    # Create layers that form embeddings.
    inputs, embeddings = dense_layers()

    # Invoke the attention layer.
    alpha = AttentionLayer(
        L_dim=256,
        output_dim=1,
        kernel_regularizer=regularizers.l2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    mul_layers = [layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))]

    # Concatenate layers.
    concat = layers.concatenate(mul_layers, axis=1)

    # Dropout layers.
    drop = layers.Dropout(0.3)(concat)

    # Classification output node.
    output = layers.Dense(2, activation="softmax")(concat)

    return Model(inputs, output)


"""
## Class Weights

Since this kind of problem could simply turn into imbalanced data classification problem, class weights should be considered.

Let's say there are 1000 bags. There often could be cases were ~90 % of the bags does not contain any positive label and ~10 % does. Such data can be referred to as **Imbalanced data**.

Using class weights, the model will tend to consider the rare class more as compared to that of the abundant one.
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
## Build and Train Model

The model is built and trained in this section.

Some regularization techniques are considered to avoid overfitting the model which ensures minimal generalization error.
"""


def train(train_data, train_labels, val_data, val_labels, model):

    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:

        # Take the file name from the wrapper.
        file_path = tmp.name

        # Initialze model checkpoint callback.
        mc = callbacks.ModelCheckpoint(
            file_path,
            monitor="val_accuracy",
            verbose=0,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )

        # Initialze early stopping callback.
        # The model performance is monitored across the unseen data and stops training
        # when the generalization error cease to decrease
        es = callbacks.EarlyStopping(monitor="val_accuracy", patience=10, mode="max")

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
            callbacks=[es, mc],
            verbose=0,
        )

        # Load best weights.
        model.load_weights(file_path)

        return model


# Building model(s).
instance_shape = train_data[0][0].shape
models = [create_model(instance_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

# Training model(s).
trained_models = [
    train(train_data, train_labels, val_data, val_labels, model)
    for model in tqdm(models)
]

"""
# Model Evaluation

The models are in their deterministic state now and ready for evaluation.

Based on the number of models (ENSEMBLE_AVG_COUNT), the models predict the results
and then averaged out (equal contribution per model).
"""


def predict(data, labels, trained_models):
    predictions = []
    attention_weights = []
    losses = []
    accuracies = []

    for model in trained_models:

        # Predict output classes on validation data.
        pred = model.predict(data)
        predictions.append(pred)

        # Create intermediate model to get attention layer weights.
        interm_model = Model(model.input, model.get_layer("alpha").output)

        # Predict attention layer weights.
        pred_interm = interm_model.predict(data)

        # Reshape list of arrays.
        A_W = np.squeeze(np.swapaxes(pred_interm, 1, 0))
        attention_weights.append(A_W)

        loss, acc = model.evaluate(data, labels, verbose=0)
        losses.append(loss)
        accuracies.append(acc)

    def average(data_list):
        return np.sum(data_list, axis=0) / ENSEMBLE_AVG_COUNT

    print(
        "The average loss and accuracy are {0:.2f} and {1:.2f} % resp.".format(
            average(losses), 100 * average(accuracies)
        )
    )

    return average(predictions), average(attention_weights)


pred_class, atten_weights = predict(val_data, val_labels, trained_models)

# Plot some results of validation data bags per class.
plot(val_data, val_labels, "pos", pred=pred_class, attention_weights=atten_weights)
plot(val_data, val_labels, "neg", pred=pred_class, attention_weights=atten_weights)

"""
# Conclusion
From the above plot, you can notice that the weights are always summing to 1. If it is a negative predicted bag,
the weights will somehow be equally distributed. However, in a positive predict bag, the instance which resulted
to the positve labelling, will have the highest attention weight among that bag.

## Remarks
- If the model is overfitted, the weights will be equally distributed for all bags. Hence, the regularization techniques
are necessary.

- In the paper, the bags' sizes can differ from one bag to another. For simplicity,the bags' sizes are fixed here.

- In order not to rely on the random initial weights of a single model, averaging ensemble methods are considered.
"""

# Visualize the hyperparameter tuning process

**Author:** Haifeng Jin<br>
**Date created:** 2021/06/25<br>
**Last modified:** 2021/06/05<br>
**Description:** Using TensorBoard to visualize the hyperparameter tuning process in KerasTuner.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/visualize_tuning.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/visualize_tuning.py)




```python
!pip install keras-tuner -q
```

---
## Introduction

KerasTuner prints the logs to screen including the values of the
hyperparameters in each trial for the user to monitor the progress. However,
reading the logs is not intuitive enough to sense the influences of
hyperparameters have on the results, Therefore, we provide a method to
visualize the hyperparameter values and the corresponding evaluation results
with interactive figures using TensorBaord.

[TensorBoard](https://www.tensorflow.org/tensorboard) is a useful tool for
visualizing the machine learning experiments.  It can monitor the losses and
metrics during the model training and visualize the model architectures.
Running KerasTuner with TensorBoard will give you additional features for
visualizing hyperparameter tuning results using its HParams plugin.

We will use a simple example of tuning a model for the MNIST image
classification dataset to show how to use KerasTuner with TensorBoard.

The first step is to download and format the data.


```python
import numpy as np
import keras_tuner
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Normalize the pixel values to the range of [0, 1].
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Add the channel dimension to the images.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# Print the shapes of the data.
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```

<div class="k-default-codeblock">
```
2022-04-28 04:13:37.237059: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-04-28 04:13:37.237117: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

(60000, 28, 28, 1)
(60000,)
(10000, 28, 28, 1)
(10000,)

```
</div>
Then, we write a `build_model` function to build the model with hyperparameters
and return the model. The hyperparameters include the type of model to use
(multi-layer perceptron or convolutional neural network), the number of layers,
the number of units or filters, whether to use dropout.


```python

def build_model(hp):
    inputs = keras.Input(shape=(28, 28, 1))
    # Model type can be MLP or CNN.
    model_type = hp.Choice("model_type", ["mlp", "cnn"])
    x = inputs
    if model_type == "mlp":
        x = layers.Flatten()(x)
        # Number of layers of the MLP is a hyperparameter.
        for i in range(hp.Int("mlp_layers", 1, 3)):
            # Number of units of each layer are
            # different hyperparameters with different names.
            output_node = layers.Dense(
                units=hp.Int(f"units_{i}", 32, 128, step=32), activation="relu",
            )(x)
    else:
        # Number of layers of the CNN is also a hyperparameter.
        for i in range(hp.Int("cnn_layers", 1, 3)):
            x = layers.Conv2D(
                hp.Int(f"filters_{i}", 32, 128, step=32),
                kernel_size=(3, 3),
                activation="relu",
            )(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)

    # A hyperparamter for whether to use dropout layer.
    if hp.Boolean("dropout"):
        x = layers.Dropout(0.5)(x)

    # The last layer contains 10 units,
    # which is the same as the number of classes.
    outputs = layers.Dense(units=10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model.
    model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam",
    )
    return model

```

We can do a quick test of the models to check if it build successfully for both
CNN and MLP.


```python

# Initialize the `HyperParameters` and set the values.
hp = keras_tuner.HyperParameters()
hp.values["model_type"] = "cnn"
# Build the model using the `HyperParameters`.
model = build_model(hp)
# Test if the model runs with our data.
model(x_train[:100])
# Print a summary of the model.
model.summary()

# Do the same for MLP model.
hp.values["model_type"] = "mlp"
model = build_model(hp)
model(x_train[:100])
model.summary()
```

<div class="k-default-codeblock">
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 5408)              0         
                                                                 
 dense (Dense)               (None, 10)                54090     
                                                                 
=================================================================
Total params: 54,410
Trainable params: 54,410
Non-trainable params: 0
_________________________________________________________________
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 28, 28, 1)]       0         
                                                                 
 flatten_1 (Flatten)         (None, 784)               0         
                                                                 
 dense_2 (Dense)             (None, 10)                7850      
                                                                 
=================================================================
Total params: 7,850
Trainable params: 7,850
Non-trainable params: 0
_________________________________________________________________

2022-04-28 04:13:39.859939: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2022-04-28 04:13:39.860040: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-04-28 04:13:39.860073: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (haifengj.c.googlers.com): /proc/driver/nvidia/version does not exist
2022-04-28 04:13:39.860436: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

```
</div>
Initialize the `RandomSearch` tuner with 10 trials and using validation
accuracy as the metric for selecting models.


```python
tuner = keras_tuner.RandomSearch(
    build_model,
    max_trials=10,
    # Do not resume the previous search in the same directory.
    overwrite=True,
    objective="val_accuracy",
    # Set a directory to store the intermediate results.
    directory="/tmp/tb",
)
```

Start the search by calling `tuner.search(...)`. To use TensorBoard, we need
to pass a `keras.callbacks.TensorBoard` instance to the callbacks.


```python
tuner.search(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=2,
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[keras.callbacks.TensorBoard("/tmp/tb_logs")],
)
```

<div class="k-default-codeblock">
```
Trial 10 Complete [00h 00m 08s]
val_accuracy: 0.9115833044052124
```
</div>
    
<div class="k-default-codeblock">
```
Best val_accuracy So Far: 0.984749972820282
Total elapsed time: 00h 10m 28s
INFO:tensorflow:Oracle triggered exit

```
</div>
If running in Colab, the following two commands will show you the TensorBoard
inside Colab.

`%load_ext tensorboard`

`%tensorboard --logdir /tmp/tb_logs`

You have access to all the common features of the TensorBoard. For example, you
can view the loss and metrics curves and visualize the computational graph of
the models in different trials.

![Loss and metrics curves](https://i.imgur.com/ShulDtI.png)
![Computational graphs](https://i.imgur.com/8sRiT1I.png)

In addition to these features, we also have a HParams tab, in which there are
three views.  In the table view, you can view the 10 different trials in a
table with the different hyperparameter values and evaluation metrics.

![Table view](https://i.imgur.com/OMcQdOw.png)

On the left side, you can specify the filters for certain hyperparameters. For
example, you can specify to only view the MLP models without the dropout layer
and with 1 to 2 dense layers.

![Filtered table view](https://i.imgur.com/yZpfaxN.png)

Besides the table view, it also provides two other views, parallel coordinates
view and scatter plot matrix view. They are just different visualization
methods for the same data. You can still use the panel on the left to filter
the results.

In the parallel coordinates view, each colored line is a trial.
The axes are the hyperparameters and evaluation metrics.

![Parallel coordinates view](https://i.imgur.com/PJ7HQUQ.png)

In the scatter plot matrix view, each dot is a trial. The plots are projections
of the trials on planes with different hyperparameter and metrics as the axes.

![Scatter plot matrix view](https://i.imgur.com/zjPjh6o.png)

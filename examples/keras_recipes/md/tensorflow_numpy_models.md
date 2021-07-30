# Writing Keras Models With TensorFlow NumPy

**Author:** lukewood<br>
**Date created:** 2021/08/28<br>
**Last modified:** 2021/08/28<br>
**Description:** Overview of how to use the TensorFlow NumPy API to write Keras models.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/tensorflow_numpy_models.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/tensorflow_numpy_models.py)



---
## Introduction

[NumPy](https://numpy.org/) is a hugely successful Python linear algebra library.

TensorFlow recently launched [tf_numpy](https://www.tensorflow.org/guide/tf_numpy), a
TensorFlow implementation of a large subset of the NumPy API.
Thanks to `tf_numpy`, you can write Keras layers or models in the NumPy style!

The TensorFlow NumPy API has full integration with the TensorFlow ecosystem.
Features such as automatic differentiation, TensorBoard, Keras model callbacks,
TPU distribution and model exporting are all supported.

Let's run through a few examples.

---
## Setup
TensorFlow NumPy requires TensorFlow 2.5 or later.


```python
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import keras
import keras.layers as layers
import numpy as np
```

<div class="k-default-codeblock">
```
2021-07-29 17:57:42.026605: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.026626: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

```
</div>
Optionally, you can call `tnp.experimental_enable_numpy_behavior()` to enable type promotion in TensorFlow.
This allows TNP to more closely follow the NumPy standard.


```python
tnp.experimental_enable_numpy_behavior()
```

To test our models we will use the Boston housing prices regression dataset.


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)


def evaluate_model(model: keras.Model):
    [loss, percent_error] = model.evaluate(x_test, y_test, verbose=0)
    print("Mean absolute percent error before training: ", percent_error)
    model.fit(x_train, y_train, epochs=200, verbose=0)
    [loss, percent_error] = model.evaluate(x_test, y_test, verbose=0)
    print("Mean absolute percent error after training:", percent_error)

```

---
## Subclassing keras.Model with TNP

The most flexible way to make use of the Keras API is to subclass the
[`keras.Model`](https://keras.io/api/models/model/) class.  Subclassing the Model class
gives you the ability to fully customize what occurs in the training loop.  This makes
subclassing Model a popular option for researchers.

In this example, we will implement a `Model` subclass that performs regression over the
boston housing dataset using the TNP API.  Note that differentiation and gradient
descent is handled automatically when using the TNP API alongside keras.

First let's define a simple `TNPForwardFeedRegressionNetwork` class.


```python

class TNPForwardFeedRegressionNetwork(keras.Model):
    def __init__(self, blocks=None, **kwargs):
        super(TNPForwardFeedRegressionNetwork, self).__init__(**kwargs)
        if not isinstance(blocks, list):
            raise ValueError(f"blocks must be a list, got blocks={blocks}")
        self.blocks = blocks
        self.block_weights = None
        self.biases = None

    def build(self, input_shape):
        current_shape = input_shape[1]
        self.block_weights = []
        self.biases = []
        for i, block in enumerate(self.blocks):
            self.block_weights.append(
                self.add_weight(
                    shape=(current_shape, block), trainable=True, name=f"block-{i}"
                )
            )
            self.biases.append(
                self.add_weight(shape=(block,), trainable=True, name=f"bias-{i}")
            )
            current_shape = block

        self.linear_layer = self.add_weight(
            shape=(current_shape, 1), name="linear_projector", trainable=True
        )

    def call(self, inputs):
        activations = inputs
        for w, b in zip(self.block_weights, self.biases):
            activations = tnp.matmul(activations, w) + b
            # ReLu activation function
            activations = tnp.maximum(activations, 0.0)

        return tnp.matmul(activations, self.linear_layer)

```

Just like with any other Keras model we can utilize any supported optimizer, loss,
metrics or callbacks that we want.

Let's see how the model performs!


```python
model = TNPForwardFeedRegressionNetwork(blocks=[3, 3])
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
evaluate_model(model)
```

<div class="k-default-codeblock">
```
2021-07-29 17:57:42.943880: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1
2021-07-29 17:57:42.985693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1733] Found device 0 with properties: 
pciBusID: 0000:65:00.0 name: Quadro P1000 computeCapability: 6.1
coreClock: 1.4805GHz coreCount: 5 deviceMemorySize: 3.94GiB deviceMemoryBandwidth: 74.65GiB/s
2021-07-29 17:57:42.985856: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.985949: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcublas.so.11'; dlerror: libcublas.so.11: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.986035: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcublasLt.so.11'; dlerror: libcublasLt.so.11: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.986135: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.986218: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.986300: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.986382: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcusparse.so.11'; dlerror: libcusparse.so.11: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.986465: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory
2021-07-29 17:57:42.986479: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1766] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2021-07-29 17:57:42.986946: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-07-29 17:57:42.988383: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1258] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-07-29 17:57:42.988403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1264]      
2021-07-29 17:57:43.339481: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
2021-07-29 17:57:43.339883: I tensorflow/core/platform/profile_utils/cpu_utils.cc:114] CPU Frequency: 3699850000 Hz

Mean absolute percent error before training:  422.45343017578125
Mean absolute percent error after training: 97.24715423583984

```
</div>
Great!  Our model seems to be effectively learning to solve the problem at hand.

We can also write our own custom loss function using TNP.


```python

def tnp_mse(y_true, y_pred):
    return tnp.mean(tnp.square(y_true - y_pred), axis=0)


keras.backend.clear_session()
model = TNPForwardFeedRegressionNetwork(blocks=[3, 3])
model.compile(
    optimizer="adam",
    loss=tnp_mse,
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
evaluate_model(model)
```

<div class="k-default-codeblock">
```
Mean absolute percent error before training:  79.84039306640625
Mean absolute percent error after training: 28.658035278320312

```
</div>
---
## Implementing a Keras Layer Based Model with TNP

If desired, TNP can also be used in layer oriented Keras code structure.  Let's
implement the same model, but using a layered approach!


```python

def tnp_relu(x):
    return tnp.maximum(x, 0)


class TNPDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights",
            shape=(input_shape[1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        outputs = tnp.matmul(inputs, self.w) + self.bias
        if self.activation:
            return self.activation(outputs)
        return outputs


def create_layered_tnp_model():
    return keras.Sequential(
        [
            TNPDense(3, activation=tnp_relu),
            TNPDense(3, activation=tnp_relu),
            TNPDense(1),
        ]
    )


model = create_layered_tnp_model()
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
model.build((None, 13,))
model.summary()

evaluate_model(model)
```

<div class="k-default-codeblock">
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
tnp_dense (TNPDense)         (None, 3)                 42        
_________________________________________________________________
tnp_dense_1 (TNPDense)       (None, 3)                 12        
_________________________________________________________________
tnp_dense_2 (TNPDense)       (None, 1)                 4         
=================================================================
Total params: 58
Trainable params: 58
Non-trainable params: 0
_________________________________________________________________
Mean absolute percent error before training:  101.17143249511719
Mean absolute percent error after training: 23.479856491088867

```
</div>
You can also seamlessly switch between TNP layers and native Keras layers!


```python

def create_mixed_model():
    return keras.Sequential(
        [
            TNPDense(3, activation=tnp_relu),
            # The model will have no issue using a normal Dense layer
            layers.Dense(3, activation="relu"),
            # ... or switching back to tnp layers!
            TNPDense(1),
        ]
    )


model = create_mixed_model()
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=[keras.metrics.MeanAbsolutePercentageError()],
)
model.build((None, 13,))
model.summary()

evaluate_model(model)
```

<div class="k-default-codeblock">
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
tnp_dense_3 (TNPDense)       (None, 3)                 42        
_________________________________________________________________
dense (Dense)                (None, 3)                 12        
_________________________________________________________________
tnp_dense_4 (TNPDense)       (None, 1)                 4         
=================================================================
Total params: 58
Trainable params: 58
Non-trainable params: 0
_________________________________________________________________
Mean absolute percent error before training:  104.59967041015625
Mean absolute percent error after training: 27.712949752807617

```
</div>
The Keras API offers a wide variety of layers.  The ability to use them alongside NumPy
code can be a huge time saver in projects.

---
## Distribution Strategy

TensorFlow NumPy and Keras integrate with
[TensorFlow Distribution Strategies](https://www.tensorflow.org/guide/distributed_training).
This makes it simple to perform distributed training across multiple GPUs,
or even an entire TPU Pod.


```python
gpus = tf.config.list_logical_devices("GPU")
if gpus:
    strategy = tf.distribute.MirroredStrategy(gpus)
else:
    # We can fallback to a no-op CPU strategy.
    strategy = tf.distribute.get_strategy()
print("Running with strategy:", str(strategy.__class__.__name__))

with strategy.scope():
    model = create_layered_tnp_model()
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanAbsolutePercentageError()],
    )
    model.build((None, 13,))
    model.summary()
    evaluate_model(model)
```

<div class="k-default-codeblock">
```
Running with strategy: _DefaultDistributionStrategy
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
tnp_dense_5 (TNPDense)       (None, 3)                 42        
_________________________________________________________________
tnp_dense_6 (TNPDense)       (None, 3)                 12        
_________________________________________________________________
tnp_dense_7 (TNPDense)       (None, 1)                 4         
=================================================================
Total params: 58
Trainable params: 58
Non-trainable params: 0
_________________________________________________________________
Mean absolute percent error before training:  100.5331039428711
Mean absolute percent error after training: 20.71842384338379

```
</div>
---
## TensorBoard Integration

One of the many benefits of using the Keras API is the ability to monitor training
through TensorBoard.  Using the TensorFlow NumPy API alongside Keras allows you to easily
leverage TensorBoard.


```python
keras.backend.clear_session()
```

To load the TensorBoard from a Jupyter notebook, you can run the following magic:
```
%load_ext tensorboard
```


```python
models = [
    (TNPForwardFeedRegressionNetwork(blocks=[3, 3]), "TNPForwardFeedRegressionNetwork"),
    (create_layered_tnp_model(), "layered_tnp_model"),
    (create_mixed_model(), "mixed_model"),
]
for model, model_name in models:
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanAbsolutePercentageError()],
    )
    model.fit(
        x_train,
        y_train,
        epochs=200,
        verbose=0,
        callbacks=[keras.callbacks.TensorBoard(log_dir=f"logs/{model_name}")],
    )
```

<div class="k-default-codeblock">
```
2021-07-29 17:58:14.405316: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
2021-07-29 17:58:14.405358: I tensorflow/core/profiler/lib/profiler_session.cc:141] Profiler session started.
2021-07-29 17:58:14.405412: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1611] Profiler found 1 GPUs
2021-07-29 17:58:14.405708: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcupti.so.11.2'; dlerror: libcupti.so.11.2: cannot open shared object file: No such file or directory
2021-07-29 17:58:14.405798: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcupti.so'; dlerror: libcupti.so: cannot open shared object file: No such file or directory
2021-07-29 17:58:14.405812: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1661] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:14.406093: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session tear down.
2021-07-29 17:58:14.406121: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1752] function cupti_interface_->Finalize()failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:14.958623: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
2021-07-29 17:58:14.958654: I tensorflow/core/profiler/lib/profiler_session.cc:141] Profiler session started.
2021-07-29 17:58:14.958761: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1661] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:14.960648: I tensorflow/core/profiler/lib/profiler_session.cc:66] Profiler session collecting data.
2021-07-29 17:58:14.960699: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1752] function cupti_interface_->Finalize()failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:14.961077: I tensorflow/core/profiler/internal/gpu/cupti_collector.cc:673]  GpuTracer has collected 0 callback api events and 0 activity events. 
2021-07-29 17:58:14.961770: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session tear down.
2021-07-29 17:58:14.963236: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14
2021-07-29 17:58:14.963851: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for trace.json.gz to logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14/yeti.sfo.corp.google.com.trace.json.gz
2021-07-29 17:58:14.965368: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14
2021-07-29 17:58:14.965447: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for memory_profile.json.gz to logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14/yeti.sfo.corp.google.com.memory_profile.json.gz
2021-07-29 17:58:14.965608: I tensorflow/core/profiler/rpc/client/capture_profile.cc:251] Creating directory: logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14Dumped tool data for xplane.pb to logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14/yeti.sfo.corp.google.com.xplane.pb
Dumped tool data for overview_page.pb to logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14/yeti.sfo.corp.google.com.overview_page.pb
Dumped tool data for input_pipeline.pb to logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14/yeti.sfo.corp.google.com.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14/yeti.sfo.corp.google.com.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to logs/TNPForwardFeedRegressionNetwork/train/plugins/profile/2021_07_29_17_58_14/yeti.sfo.corp.google.com.kernel_stats.pb
```
</div>
    


<div class="k-default-codeblock">
```
WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... Layer TNPDense has arguments in `__init__` and therefore must override `get_config`.

2021-07-29 17:58:19.449655: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
2021-07-29 17:58:19.449734: I tensorflow/core/profiler/lib/profiler_session.cc:141] Profiler session started.
2021-07-29 17:58:19.449892: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1661] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:19.449962: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session tear down.
2021-07-29 17:58:19.449998: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1752] function cupti_interface_->Finalize()failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:22.011179: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
2021-07-29 17:58:22.011211: I tensorflow/core/profiler/lib/profiler_session.cc:141] Profiler session started.
2021-07-29 17:58:22.011241: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1661] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:22.022699: I tensorflow/core/profiler/lib/profiler_session.cc:66] Profiler session collecting data.
2021-07-29 17:58:22.022762: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1752] function cupti_interface_->Finalize()failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:22.023090: I tensorflow/core/profiler/internal/gpu/cupti_collector.cc:673]  GpuTracer has collected 0 callback api events and 0 activity events. 
2021-07-29 17:58:22.023764: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session tear down.
2021-07-29 17:58:22.024763: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22
2021-07-29 17:58:22.025353: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for trace.json.gz to logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22/yeti.sfo.corp.google.com.trace.json.gz
2021-07-29 17:58:22.026151: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22
2021-07-29 17:58:22.026214: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for memory_profile.json.gz to logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22/yeti.sfo.corp.google.com.memory_profile.json.gz
2021-07-29 17:58:22.026407: I tensorflow/core/profiler/rpc/client/capture_profile.cc:251] Creating directory: logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22Dumped tool data for xplane.pb to logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22/yeti.sfo.corp.google.com.xplane.pb
Dumped tool data for overview_page.pb to logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22/yeti.sfo.corp.google.com.overview_page.pb
Dumped tool data for input_pipeline.pb to logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22/yeti.sfo.corp.google.com.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22/yeti.sfo.corp.google.com.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to logs/layered_tnp_model/train/plugins/profile/2021_07_29_17_58_22/yeti.sfo.corp.google.com.kernel_stats.pb
```
</div>
    


<div class="k-default-codeblock">
```
WARNING:tensorflow:Model failed to serialize as JSON. Ignoring... Layer TNPDense has arguments in `__init__` and therefore must override `get_config`.

2021-07-29 17:58:26.608574: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
2021-07-29 17:58:26.608649: I tensorflow/core/profiler/lib/profiler_session.cc:141] Profiler session started.
2021-07-29 17:58:26.608706: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1661] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:26.608841: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session tear down.
2021-07-29 17:58:26.608876: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1752] function cupti_interface_->Finalize()failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:27.255248: I tensorflow/core/profiler/lib/profiler_session.cc:126] Profiler session initializing.
2021-07-29 17:58:27.255282: I tensorflow/core/profiler/lib/profiler_session.cc:141] Profiler session started.
2021-07-29 17:58:27.255389: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1661] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:27.266298: I tensorflow/core/profiler/lib/profiler_session.cc:66] Profiler session collecting data.
2021-07-29 17:58:27.266360: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1752] function cupti_interface_->Finalize()failed with error CUPTI could not be loaded or symbol could not be found.
2021-07-29 17:58:27.266683: I tensorflow/core/profiler/internal/gpu/cupti_collector.cc:673]  GpuTracer has collected 0 callback api events and 0 activity events. 
2021-07-29 17:58:27.267354: I tensorflow/core/profiler/lib/profiler_session.cc:159] Profiler session tear down.
2021-07-29 17:58:27.268329: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27
2021-07-29 17:58:27.268915: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for trace.json.gz to logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27/yeti.sfo.corp.google.com.trace.json.gz
2021-07-29 17:58:27.269671: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27
2021-07-29 17:58:27.269736: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for memory_profile.json.gz to logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27/yeti.sfo.corp.google.com.memory_profile.json.gz
2021-07-29 17:58:27.269905: I tensorflow/core/profiler/rpc/client/capture_profile.cc:251] Creating directory: logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27Dumped tool data for xplane.pb to logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27/yeti.sfo.corp.google.com.xplane.pb
Dumped tool data for overview_page.pb to logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27/yeti.sfo.corp.google.com.overview_page.pb
Dumped tool data for input_pipeline.pb to logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27/yeti.sfo.corp.google.com.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27/yeti.sfo.corp.google.com.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to logs/mixed_model/train/plugins/profile/2021_07_29_17_58_27/yeti.sfo.corp.google.com.kernel_stats.pb
```
</div>
    


To load the TensorBoard from a Jupyter notebook you can use the `%tensorboard` magic:

```
%tensorboard --logdir logs
```

The TensorBoard monitor metrics and examine the training curve.

![Tensorboard training graph](https://i.imgur.com/wsOuFnz.png)

The TensorBoard also allows you to explore the computation graph used in your models.

![Tensorboard graph exploration](https://i.imgur.com/tOrezDL.png)

The ability to introspect into your models can be valuable during debugging.

---
## Conclusion

Porting existing NumPy code to Keras models using the `tensorflow_numpy` API is easy!
By integrating with Keras you gain the ability to use existing Keras callbacks, metrics
and optimizers, easily distribute your training and use Tensorboard.

Migrating a more complex model, such as a ResNet, to the TensorFlow NumPy API would be a
great follow up learning exercise.

Several open source NumPy ResNet implementations are available online.

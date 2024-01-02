# Image classification with Perceiver

**Author:** [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)<br>
**Date created:** 2021/04/30<br>
**Last modified:** 2023/12/30<br>
**Description:** Implementing the Perceiver model for image classification.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/perceiver_image_classification.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/vision/perceiver_image_classification.py)



---
## Introduction

This example implements the
[Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)
model by Andrew Jaegle et al. for image classification,
and demonstrates it on the CIFAR-100 dataset.

The Perceiver model leverages an asymmetric attention mechanism to iteratively
distill inputs into a tight latent bottleneck,
allowing it to scale to handle very large inputs.

In other words: let's assume that your input data array (e.g. image) has `M` elements (i.e. patches), where `M` is large.
In a standard Transformer model, a self-attention operation is performed for the `M` elements.
The complexity of this operation is `O(M^2)`.
However, the Perceiver model creates a latent array of size `N` elements, where `N << M`,
and performs two operations iteratively:

1. Cross-attention Transformer between the latent array and the data array - The complexity of this operation is `O(M.N)`.
2. Self-attention Transformer on the latent array -  The complexity of this operation is `O(N^2)`.

This example requires Keras 3.0 or higher.

---
## Setup


```python
import keras
from keras import layers, activations, ops
```

---
## Prepare the data


```python
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
```

<div class="k-default-codeblock">
```
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
x_test shape: (10000, 32, 32, 3) - y_test shape: (10000, 1)

```
</div>
---
## Configure the hyperparameters


```python
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_epochs = 2 # It is recommended to run 50 epochs to observe improvements in accuracy
dropout_rate = 0.2
image_size = 64  # We'll resize input images to this size.
patch_size = 2  # Size of the patches to be extract from the input images.
num_patches = (image_size // patch_size) ** 2  # Size of the data array.
latent_dim = 256  # Size of the latent array.
projection_dim = 256  # Embedding size of each element in the data and latent arrays.
num_heads = 8  # Number of Transformer heads.
ffn_units = [
    projection_dim,
    projection_dim,
]  # Size of the Transformer Feedforward network.
num_transformer_blocks = 4
num_iterations = 2  # Repetitions of the cross-attention and Transformer modules.
classifier_units = [
    projection_dim,
    num_classes,
]  # Size of the Feedforward network of the final classifier.

print(f"Image size: {image_size} X {image_size} = {image_size ** 2}")
print(f"Patch size: {patch_size} X {patch_size} = {patch_size ** 2} ")
print(f"Patches per image: {num_patches}")
print(f"Elements per patch (3 channels): {(patch_size ** 2) * 3}")
print(f"Latent array shape: {latent_dim} X {projection_dim}")
print(f"Data array shape: {num_patches} X {projection_dim}")
```

<div class="k-default-codeblock">
```
Image size: 64 X 64 = 4096
Patch size: 2 X 2 = 4 
Patches per image: 1024
Elements per patch (3 channels): 12
Latent array shape: 256 X 256
Data array shape: 1024 X 256

```
</div>
Note that, in order to use each pixel as an individual input in the data array,
set `patch_size` to 1.

---
## Use data augmentation


```python
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)
```

---
## Implement Feedforward network (FFN)


```python

def create_ffn(hidden_units, dropout_rate):
    ffn_layers = []
    for units in hidden_units[:-1]:
        ffn_layers.append(layers.Dense(units, activation=activations.gelu))

    ffn_layers.append(layers.Dense(units=hidden_units[-1]))
    ffn_layers.append(layers.Dropout(dropout_rate))

    ffn = keras.Sequential(ffn_layers)
    return ffn

```

---
## Implement patch creation as a layer


```python

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = ops.shape(images)[0]
        patches = ops.image.extract_patches(
            image=images,
            size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            dilation_rate=1,
            padding="valid",
        )
        patch_dims = patches.shape[-1]
        patches = ops.reshape(patches, [batch_size, -1, patch_dims])
        return patches

```

---
## Implement the patch encoding layer

The `PatchEncoder` layer will linearly transform a patch by projecting it into
a vector of size `latent_dim`. In addition, it adds a learnable position embedding
to the projected vector.

Note that the orginal Perceiver paper uses the Fourier feature positional encodings.


```python

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = ops.arange(start=0, stop=self.num_patches, step=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

```

---
## Build the Perceiver model

The Perceiver consists of two modules: a cross-attention
module and a standard Transformer with self-attention.

### Cross-attention module

The cross-attention expects a `(latent_dim, projection_dim)` latent array,
and the `(data_dim,  projection_dim)` data array as inputs,
to produce a `(latent_dim, projection_dim)` latent array as an output.
To apply cross-attention, the `query` vectors are generated from the latent array,
while the `key` and `value` vectors are generated from the encoded image.

Note that the data array in this example is the image,
where the `data_dim` is set to the `num_patches`.


```python

def create_cross_attention_module(
    latent_dim, data_dim, projection_dim, ffn_units, dropout_rate
):
    inputs = {
        # Recieve the latent array as an input of shape [1, latent_dim, projection_dim].
        "latent_array": layers.Input(
            shape=(latent_dim, projection_dim), name="latent_array"
        ),
        # Recieve the data_array (encoded image) as an input of shape [batch_size, data_dim, projection_dim].
        "data_array": layers.Input(shape=(data_dim, projection_dim), name="data_array"),
    }

    # Apply layer norm to the inputs
    latent_array = layers.LayerNormalization(epsilon=1e-6)(inputs["latent_array"])
    data_array = layers.LayerNormalization(epsilon=1e-6)(inputs["data_array"])

    # Create query tensor: [1, latent_dim, projection_dim].
    query = layers.Dense(units=projection_dim)(latent_array)
    # Create key tensor: [batch_size, data_dim, projection_dim].
    key = layers.Dense(units=projection_dim)(data_array)
    # Create value tensor: [batch_size, data_dim, projection_dim].
    value = layers.Dense(units=projection_dim)(data_array)

    # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
    attention_output = layers.Attention(use_scale=True, dropout=0.1)(
        [query, key, value], return_attention_scores=False
    )
    # Skip connection 1.
    attention_output = layers.Add()([attention_output, latent_array])

    # Apply layer norm.
    attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
    # Apply Feedforward network.
    ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
    outputs = ffn(attention_output)
    # Skip connection 2.
    outputs = layers.Add()([outputs, attention_output])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

```

### Transformer module

The Transformer expects the output latent vector from the cross-attention module
as an input, applies multi-head self-attention to its `latent_dim` elements,
followed by feedforward network, to produce another `(latent_dim, projection_dim)` latent array.


```python

def create_transformer_module(
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
):
    # input_shape: [1, latent_dim, projection_dim]
    inputs = layers.Input(shape=(latent_dim, projection_dim))

    x0 = inputs
    # Create multiple layers of the Transformer block.
    for _ in range(num_transformer_blocks):
        # Apply layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x0)
        # Create a multi-head self-attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x0])
        # Apply layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Apply Feedforward network.
        ffn = create_ffn(hidden_units=ffn_units, dropout_rate=dropout_rate)
        x3 = ffn(x3)
        # Skip connection 2.
        x0 = layers.Add()([x3, x2])

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x0)
    return model

```

### Perceiver model

The Perceiver model repeats the cross-attention and Transformer modules
`num_iterations` times—with shared weights and skip connections—to allow
the latent array to iteratively extract information from the input image as it is needed.


```python

class Perceiver(keras.Model):
    def __init__(
        self,
        patch_size,
        data_dim,
        latent_dim,
        projection_dim,
        num_heads,
        num_transformer_blocks,
        ffn_units,
        dropout_rate,
        num_iterations,
        classifier_units,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.num_iterations = num_iterations
        self.classifier_units = classifier_units

    def build(self, input_shape):
        # Create latent array.
        self.latent_array = self.add_weight(
            shape=(self.latent_dim, self.projection_dim),
            initializer="random_normal",
            trainable=True,
        )

        # Create patching module.war
        self.patch_encoder = PatchEncoder(self.data_dim, self.projection_dim)

        # Create cross-attenion module.
        self.cross_attention = create_cross_attention_module(
            self.latent_dim,
            self.data_dim,
            self.projection_dim,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create Transformer module.
        self.transformer = create_transformer_module(
            self.latent_dim,
            self.projection_dim,
            self.num_heads,
            self.num_transformer_blocks,
            self.ffn_units,
            self.dropout_rate,
        )

        # Create global average pooling layer.
        self.global_average_pooling = layers.GlobalAveragePooling1D()

        # Create a classification head.
        self.classification_head = create_ffn(
            hidden_units=self.classifier_units, dropout_rate=self.dropout_rate
        )

        super().build(input_shape)

    def call(self, inputs):
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = self.patcher(augmented)
        # Encode patches.
        encoded_patches = self.patch_encoder(patches)
        # Prepare cross-attention inputs.
        cross_attention_inputs = {
            "latent_array": ops.expand_dims(self.latent_array, 0),
            "data_array": encoded_patches,
        }
        # Apply the cross-attention and the Transformer modules iteratively.
        for _ in range(self.num_iterations):
            # Apply cross-attention from the latent array to the data array.
            latent_array = self.cross_attention(cross_attention_inputs)
            # Apply self-attention Transformer to the latent array.
            latent_array = self.transformer(latent_array)
            # Set the latent array of the next iteration.
            cross_attention_inputs["latent_array"] = latent_array

        # Apply global average pooling to generate a [batch_size, projection_dim] repesentation tensor.
        representation = self.global_average_pooling(latent_array)
        # Generate logits.
        logits = self.classification_head(representation)
        return logits

```

---
## Compile, train, and evaluate the mode


```python

def run_experiment(model):
    # Create ADAM instead of LAMB optimizer with weight decay. (LAMB isn't supported yet)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[early_stopping, reduce_lr],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history

```

Note that training the perceiver model with the current settings on a V100 GPUs takes
around 200 seconds.


```python
perceiver_classifier = Perceiver(
    patch_size,
    num_patches,
    latent_dim,
    projection_dim,
    num_heads,
    num_transformer_blocks,
    ffn_units,
    dropout_rate,
    num_iterations,
    classifier_units,
)


history = run_experiment(perceiver_classifier)
```

<div class="k-default-codeblock">
```
Epoch 1/2

```
</div>
    
   1/704 [37m━━━━━━━━━━━━━━━━━━━━  2:02:41 10s/step - acc: 0.0000e+00 - loss: 5.1023 - top5-acc: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   2/704 [37m━━━━━━━━━━━━━━━━━━━━  45:18 4s/step - acc: 0.0000e+00 - loss: 5.4356 - top5-acc: 0.0820   

<div class="k-default-codeblock">
```

```
</div>
   3/704 [37m━━━━━━━━━━━━━━━━━━━━  44:58 4s/step - acc: 0.0000e+00 - loss: 5.5133 - top5-acc: 0.0825

<div class="k-default-codeblock">
```

```
</div>
   4/704 [37m━━━━━━━━━━━━━━━━━━━━  44:54 4s/step - acc: 0.0000e+00 - loss: 5.5671 - top5-acc: 0.0794

<div class="k-default-codeblock">
```

```
</div>
   5/704 [37m━━━━━━━━━━━━━━━━━━━━  44:49 4s/step - acc: 0.0000e+00 - loss: 5.5885 - top5-acc: 0.0773

<div class="k-default-codeblock">
```

```
</div>
   6/704 [37m━━━━━━━━━━━━━━━━━━━━  44:44 4s/step - acc: 0.0000e+00 - loss: 5.5852 - top5-acc: 0.0757

<div class="k-default-codeblock">
```

```
</div>
   7/704 [37m━━━━━━━━━━━━━━━━━━━━  44:41 4s/step - acc: 3.1888e-04 - loss: 5.5720 - top5-acc: 0.0751

<div class="k-default-codeblock">
```

```
</div>
   8/704 [37m━━━━━━━━━━━━━━━━━━━━  44:37 4s/step - acc: 7.6730e-04 - loss: 5.5559 - top5-acc: 0.0747

<div class="k-default-codeblock">
```

```
</div>
   9/704 [37m━━━━━━━━━━━━━━━━━━━━  44:35 4s/step - acc: 0.0015 - loss: 5.5370 - top5-acc: 0.0741    

<div class="k-default-codeblock">
```

```
</div>
  10/704 [37m━━━━━━━━━━━━━━━━━━━━  44:32 4s/step - acc: 0.0021 - loss: 5.5160 - top5-acc: 0.0741

<div class="k-default-codeblock">
```

```
</div>
  11/704 [37m━━━━━━━━━━━━━━━━━━━━  44:31 4s/step - acc: 0.0028 - loss: 5.4939 - top5-acc: 0.0741

<div class="k-default-codeblock">
```

```
</div>
  12/704 [37m━━━━━━━━━━━━━━━━━━━━  44:26 4s/step - acc: 0.0037 - loss: 5.4719 - top5-acc: 0.0741

<div class="k-default-codeblock">
```

```
</div>
  13/704 [37m━━━━━━━━━━━━━━━━━━━━  44:21 4s/step - acc: 0.0043 - loss: 5.4510 - top5-acc: 0.0739

<div class="k-default-codeblock">
```

```
</div>
  14/704 [37m━━━━━━━━━━━━━━━━━━━━  44:17 4s/step - acc: 0.0048 - loss: 5.4307 - top5-acc: 0.0738

<div class="k-default-codeblock">
```

```
</div>
  15/704 [37m━━━━━━━━━━━━━━━━━━━━  44:13 4s/step - acc: 0.0052 - loss: 5.4106 - top5-acc: 0.0736

<div class="k-default-codeblock">
```

```
</div>
  16/704 [37m━━━━━━━━━━━━━━━━━━━━  44:10 4s/step - acc: 0.0056 - loss: 5.3911 - top5-acc: 0.0735

<div class="k-default-codeblock">
```

```
</div>
  17/704 [37m━━━━━━━━━━━━━━━━━━━━  44:07 4s/step - acc: 0.0059 - loss: 5.3730 - top5-acc: 0.0733

<div class="k-default-codeblock">
```

```
</div>
  18/704 [37m━━━━━━━━━━━━━━━━━━━━  44:02 4s/step - acc: 0.0062 - loss: 5.3560 - top5-acc: 0.0730

<div class="k-default-codeblock">
```

```
</div>
  19/704 [37m━━━━━━━━━━━━━━━━━━━━  43:59 4s/step - acc: 0.0065 - loss: 5.3398 - top5-acc: 0.0727

<div class="k-default-codeblock">
```

```
</div>
  20/704 [37m━━━━━━━━━━━━━━━━━━━━  43:56 4s/step - acc: 0.0068 - loss: 5.3248 - top5-acc: 0.0724

<div class="k-default-codeblock">
```

```
</div>
  21/704 [37m━━━━━━━━━━━━━━━━━━━━  43:52 4s/step - acc: 0.0071 - loss: 5.3106 - top5-acc: 0.0720

<div class="k-default-codeblock">
```

```
</div>
  22/704 [37m━━━━━━━━━━━━━━━━━━━━  43:48 4s/step - acc: 0.0073 - loss: 5.2972 - top5-acc: 0.0717

<div class="k-default-codeblock">
```

```
</div>
  23/704 [37m━━━━━━━━━━━━━━━━━━━━  43:45 4s/step - acc: 0.0075 - loss: 5.2842 - top5-acc: 0.0714

<div class="k-default-codeblock">
```

```
</div>
  24/704 [37m━━━━━━━━━━━━━━━━━━━━  43:42 4s/step - acc: 0.0076 - loss: 5.2717 - top5-acc: 0.0712

<div class="k-default-codeblock">
```

```
</div>
  25/704 [37m━━━━━━━━━━━━━━━━━━━━  43:38 4s/step - acc: 0.0078 - loss: 5.2597 - top5-acc: 0.0709

<div class="k-default-codeblock">
```

```
</div>
  26/704 [37m━━━━━━━━━━━━━━━━━━━━  43:34 4s/step - acc: 0.0079 - loss: 5.2480 - top5-acc: 0.0707

<div class="k-default-codeblock">
```

```
</div>
  27/704 [37m━━━━━━━━━━━━━━━━━━━━  43:31 4s/step - acc: 0.0080 - loss: 5.2368 - top5-acc: 0.0705

<div class="k-default-codeblock">
```

```
</div>
  28/704 [37m━━━━━━━━━━━━━━━━━━━━  43:28 4s/step - acc: 0.0082 - loss: 5.2262 - top5-acc: 0.0703

<div class="k-default-codeblock">
```

```
</div>
  29/704 [37m━━━━━━━━━━━━━━━━━━━━  43:24 4s/step - acc: 0.0083 - loss: 5.2161 - top5-acc: 0.0700

<div class="k-default-codeblock">
```

```
</div>
  30/704 [37m━━━━━━━━━━━━━━━━━━━━  43:20 4s/step - acc: 0.0084 - loss: 5.2063 - top5-acc: 0.0698

<div class="k-default-codeblock">
```

```
</div>
  31/704 [37m━━━━━━━━━━━━━━━━━━━━  43:17 4s/step - acc: 0.0084 - loss: 5.1971 - top5-acc: 0.0695

<div class="k-default-codeblock">
```

```
</div>
  32/704 [37m━━━━━━━━━━━━━━━━━━━━  43:13 4s/step - acc: 0.0085 - loss: 5.1882 - top5-acc: 0.0691

<div class="k-default-codeblock">
```

```
</div>
  33/704 [37m━━━━━━━━━━━━━━━━━━━━  43:09 4s/step - acc: 0.0085 - loss: 5.1796 - top5-acc: 0.0688

<div class="k-default-codeblock">
```

```
</div>
  34/704 [37m━━━━━━━━━━━━━━━━━━━━  43:06 4s/step - acc: 0.0086 - loss: 5.1713 - top5-acc: 0.0685

<div class="k-default-codeblock">
```

```
</div>
  35/704 [37m━━━━━━━━━━━━━━━━━━━━  43:02 4s/step - acc: 0.0086 - loss: 5.1634 - top5-acc: 0.0682

<div class="k-default-codeblock">
```

```
</div>
  36/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:58 4s/step - acc: 0.0086 - loss: 5.1558 - top5-acc: 0.0679

<div class="k-default-codeblock">
```

```
</div>
  37/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:55 4s/step - acc: 0.0086 - loss: 5.1483 - top5-acc: 0.0677

<div class="k-default-codeblock">
```

```
</div>
  38/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:51 4s/step - acc: 0.0086 - loss: 5.1412 - top5-acc: 0.0674

<div class="k-default-codeblock">
```

```
</div>
  39/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:47 4s/step - acc: 0.0087 - loss: 5.1343 - top5-acc: 0.0672

<div class="k-default-codeblock">
```

```
</div>
  40/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:44 4s/step - acc: 0.0087 - loss: 5.1275 - top5-acc: 0.0670

<div class="k-default-codeblock">
```

```
</div>
  41/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:41 4s/step - acc: 0.0087 - loss: 5.1209 - top5-acc: 0.0668

<div class="k-default-codeblock">
```

```
</div>
  42/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:37 4s/step - acc: 0.0088 - loss: 5.1145 - top5-acc: 0.0667

<div class="k-default-codeblock">
```

```
</div>
  43/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:35 4s/step - acc: 0.0088 - loss: 5.1083 - top5-acc: 0.0665

<div class="k-default-codeblock">
```

```
</div>
  44/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:32 4s/step - acc: 0.0089 - loss: 5.1023 - top5-acc: 0.0663

<div class="k-default-codeblock">
```

```
</div>
  45/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:28 4s/step - acc: 0.0089 - loss: 5.0964 - top5-acc: 0.0662

<div class="k-default-codeblock">
```

```
</div>
  46/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:25 4s/step - acc: 0.0089 - loss: 5.0907 - top5-acc: 0.0660

<div class="k-default-codeblock">
```

```
</div>
  47/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:20 4s/step - acc: 0.0090 - loss: 5.0852 - top5-acc: 0.0659

<div class="k-default-codeblock">
```

```
</div>
  48/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:17 4s/step - acc: 0.0090 - loss: 5.0798 - top5-acc: 0.0658

<div class="k-default-codeblock">
```

```
</div>
  49/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:13 4s/step - acc: 0.0091 - loss: 5.0746 - top5-acc: 0.0657

<div class="k-default-codeblock">
```

```
</div>
  50/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:09 4s/step - acc: 0.0091 - loss: 5.0695 - top5-acc: 0.0656

<div class="k-default-codeblock">
```

```
</div>
  51/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:05 4s/step - acc: 0.0091 - loss: 5.0645 - top5-acc: 0.0655

<div class="k-default-codeblock">
```

```
</div>
  52/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:01 4s/step - acc: 0.0092 - loss: 5.0597 - top5-acc: 0.0653

<div class="k-default-codeblock">
```

```
</div>
  53/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:57 4s/step - acc: 0.0092 - loss: 5.0550 - top5-acc: 0.0652

<div class="k-default-codeblock">
```

```
</div>
  54/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:53 4s/step - acc: 0.0093 - loss: 5.0504 - top5-acc: 0.0652

<div class="k-default-codeblock">
```

```
</div>
  55/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:50 4s/step - acc: 0.0093 - loss: 5.0459 - top5-acc: 0.0651

<div class="k-default-codeblock">
```

```
</div>
  56/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:45 4s/step - acc: 0.0094 - loss: 5.0416 - top5-acc: 0.0650

<div class="k-default-codeblock">
```

```
</div>
  57/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:42 4s/step - acc: 0.0094 - loss: 5.0373 - top5-acc: 0.0649

<div class="k-default-codeblock">
```

```
</div>
  58/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:38 4s/step - acc: 0.0094 - loss: 5.0331 - top5-acc: 0.0648

<div class="k-default-codeblock">
```

```
</div>
  59/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:34 4s/step - acc: 0.0094 - loss: 5.0291 - top5-acc: 0.0647

<div class="k-default-codeblock">
```

```
</div>
  60/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:30 4s/step - acc: 0.0095 - loss: 5.0252 - top5-acc: 0.0646

<div class="k-default-codeblock">
```

```
</div>
  61/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:26 4s/step - acc: 0.0095 - loss: 5.0213 - top5-acc: 0.0645

<div class="k-default-codeblock">
```

```
</div>
  62/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:22 4s/step - acc: 0.0095 - loss: 5.0176 - top5-acc: 0.0644

<div class="k-default-codeblock">
```

```
</div>
  63/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:18 4s/step - acc: 0.0096 - loss: 5.0139 - top5-acc: 0.0643

<div class="k-default-codeblock">
```

```
</div>
  64/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:14 4s/step - acc: 0.0096 - loss: 5.0104 - top5-acc: 0.0642

<div class="k-default-codeblock">
```

```
</div>
  65/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:10 4s/step - acc: 0.0096 - loss: 5.0068 - top5-acc: 0.0641

<div class="k-default-codeblock">
```

```
</div>
  66/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:06 4s/step - acc: 0.0096 - loss: 5.0034 - top5-acc: 0.0640

<div class="k-default-codeblock">
```

```
</div>
  67/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:02 4s/step - acc: 0.0097 - loss: 5.0000 - top5-acc: 0.0639

<div class="k-default-codeblock">
```

```
</div>
  68/704 ━[37m━━━━━━━━━━━━━━━━━━━  40:58 4s/step - acc: 0.0097 - loss: 4.9967 - top5-acc: 0.0638

<div class="k-default-codeblock">
```

```
</div>
  69/704 ━[37m━━━━━━━━━━━━━━━━━━━  40:55 4s/step - acc: 0.0097 - loss: 4.9935 - top5-acc: 0.0637

<div class="k-default-codeblock">
```

```
</div>
  70/704 ━[37m━━━━━━━━━━━━━━━━━━━  40:51 4s/step - acc: 0.0097 - loss: 4.9904 - top5-acc: 0.0636

<div class="k-default-codeblock">
```

```
</div>
  71/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:47 4s/step - acc: 0.0098 - loss: 4.9873 - top5-acc: 0.0635

<div class="k-default-codeblock">
```

```
</div>
  72/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:43 4s/step - acc: 0.0098 - loss: 4.9843 - top5-acc: 0.0634

<div class="k-default-codeblock">
```

```
</div>
  73/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:39 4s/step - acc: 0.0098 - loss: 4.9813 - top5-acc: 0.0633

<div class="k-default-codeblock">
```

```
</div>
  74/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:35 4s/step - acc: 0.0098 - loss: 4.9784 - top5-acc: 0.0633

<div class="k-default-codeblock">
```

```
</div>
  75/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:31 4s/step - acc: 0.0098 - loss: 4.9755 - top5-acc: 0.0632

<div class="k-default-codeblock">
```

```
</div>
  76/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:27 4s/step - acc: 0.0098 - loss: 4.9727 - top5-acc: 0.0631

<div class="k-default-codeblock">
```

```
</div>
  77/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:23 4s/step - acc: 0.0099 - loss: 4.9699 - top5-acc: 0.0630

<div class="k-default-codeblock">
```

```
</div>
  78/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:19 4s/step - acc: 0.0099 - loss: 4.9672 - top5-acc: 0.0629

<div class="k-default-codeblock">
```

```
</div>
  79/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:15 4s/step - acc: 0.0099 - loss: 4.9645 - top5-acc: 0.0628

<div class="k-default-codeblock">
```

```
</div>
  80/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:11 4s/step - acc: 0.0099 - loss: 4.9618 - top5-acc: 0.0627

<div class="k-default-codeblock">
```

```
</div>
  81/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:07 4s/step - acc: 0.0099 - loss: 4.9592 - top5-acc: 0.0626

<div class="k-default-codeblock">
```

```
</div>
  82/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:03 4s/step - acc: 0.0099 - loss: 4.9567 - top5-acc: 0.0625

<div class="k-default-codeblock">
```

```
</div>
  83/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:59 4s/step - acc: 0.0099 - loss: 4.9542 - top5-acc: 0.0624

<div class="k-default-codeblock">
```

```
</div>
  84/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:55 4s/step - acc: 0.0099 - loss: 4.9518 - top5-acc: 0.0623

<div class="k-default-codeblock">
```

```
</div>
  85/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:52 4s/step - acc: 0.0099 - loss: 4.9494 - top5-acc: 0.0622

<div class="k-default-codeblock">
```

```
</div>
  86/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:48 4s/step - acc: 0.0099 - loss: 4.9470 - top5-acc: 0.0621

<div class="k-default-codeblock">
```

```
</div>
  87/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:44 4s/step - acc: 0.0099 - loss: 4.9446 - top5-acc: 0.0620

<div class="k-default-codeblock">
```

```
</div>
  88/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:40 4s/step - acc: 0.0099 - loss: 4.9423 - top5-acc: 0.0620

<div class="k-default-codeblock">
```

```
</div>
  89/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:36 4s/step - acc: 0.0099 - loss: 4.9401 - top5-acc: 0.0619

<div class="k-default-codeblock">
```

```
</div>
  90/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:32 4s/step - acc: 0.0100 - loss: 4.9379 - top5-acc: 0.0618

<div class="k-default-codeblock">
```

```
</div>
  91/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:28 4s/step - acc: 0.0100 - loss: 4.9357 - top5-acc: 0.0617

<div class="k-default-codeblock">
```

```
</div>
  92/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:24 4s/step - acc: 0.0100 - loss: 4.9335 - top5-acc: 0.0616

<div class="k-default-codeblock">
```

```
</div>
  93/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:20 4s/step - acc: 0.0100 - loss: 4.9314 - top5-acc: 0.0616

<div class="k-default-codeblock">
```

```
</div>
  94/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:16 4s/step - acc: 0.0100 - loss: 4.9293 - top5-acc: 0.0615

<div class="k-default-codeblock">
```

```
</div>
  95/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:13 4s/step - acc: 0.0100 - loss: 4.9272 - top5-acc: 0.0614

<div class="k-default-codeblock">
```

```
</div>
  96/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:09 4s/step - acc: 0.0100 - loss: 4.9252 - top5-acc: 0.0614

<div class="k-default-codeblock">
```

```
</div>
  97/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:05 4s/step - acc: 0.0100 - loss: 4.9232 - top5-acc: 0.0613

<div class="k-default-codeblock">
```

```
</div>
  98/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:01 4s/step - acc: 0.0100 - loss: 4.9212 - top5-acc: 0.0612

<div class="k-default-codeblock">
```

```
</div>
  99/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:57 4s/step - acc: 0.0100 - loss: 4.9193 - top5-acc: 0.0611

<div class="k-default-codeblock">
```

```
</div>
 100/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:53 4s/step - acc: 0.0100 - loss: 4.9174 - top5-acc: 0.0610

<div class="k-default-codeblock">
```

```
</div>
 101/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:49 4s/step - acc: 0.0100 - loss: 4.9155 - top5-acc: 0.0610

<div class="k-default-codeblock">
```

```
</div>
 102/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:45 4s/step - acc: 0.0100 - loss: 4.9136 - top5-acc: 0.0609

<div class="k-default-codeblock">
```

```
</div>
 103/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:42 4s/step - acc: 0.0100 - loss: 4.9118 - top5-acc: 0.0608

<div class="k-default-codeblock">
```

```
</div>
 104/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:38 4s/step - acc: 0.0100 - loss: 4.9100 - top5-acc: 0.0608

<div class="k-default-codeblock">
```

```
</div>
 105/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:34 4s/step - acc: 0.0100 - loss: 4.9082 - top5-acc: 0.0607

<div class="k-default-codeblock">
```

```
</div>
 106/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:30 4s/step - acc: 0.0100 - loss: 4.9064 - top5-acc: 0.0606

<div class="k-default-codeblock">
```

```
</div>
 107/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:26 4s/step - acc: 0.0100 - loss: 4.9047 - top5-acc: 0.0606

<div class="k-default-codeblock">
```

```
</div>
 108/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:22 4s/step - acc: 0.0100 - loss: 4.9029 - top5-acc: 0.0605

<div class="k-default-codeblock">
```

```
</div>
 109/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:18 4s/step - acc: 0.0100 - loss: 4.9012 - top5-acc: 0.0604

<div class="k-default-codeblock">
```

```
</div>
 110/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:15 4s/step - acc: 0.0100 - loss: 4.8996 - top5-acc: 0.0604

<div class="k-default-codeblock">
```

```
</div>
 111/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:11 4s/step - acc: 0.0100 - loss: 4.8979 - top5-acc: 0.0603

<div class="k-default-codeblock">
```

```
</div>
 112/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:07 4s/step - acc: 0.0100 - loss: 4.8963 - top5-acc: 0.0602

<div class="k-default-codeblock">
```

```
</div>
 113/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:03 4s/step - acc: 0.0100 - loss: 4.8947 - top5-acc: 0.0602

<div class="k-default-codeblock">
```

```
</div>
 114/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:59 4s/step - acc: 0.0101 - loss: 4.8931 - top5-acc: 0.0601

<div class="k-default-codeblock">
```

```
</div>
 115/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:55 4s/step - acc: 0.0101 - loss: 4.8915 - top5-acc: 0.0600

<div class="k-default-codeblock">
```

```
</div>
 116/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:51 4s/step - acc: 0.0101 - loss: 4.8899 - top5-acc: 0.0600

<div class="k-default-codeblock">
```

```
</div>
 117/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:47 4s/step - acc: 0.0101 - loss: 4.8884 - top5-acc: 0.0599

<div class="k-default-codeblock">
```

```
</div>
 118/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:44 4s/step - acc: 0.0101 - loss: 4.8869 - top5-acc: 0.0598

<div class="k-default-codeblock">
```

```
</div>
 119/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:40 4s/step - acc: 0.0101 - loss: 4.8854 - top5-acc: 0.0598

<div class="k-default-codeblock">
```

```
</div>
 120/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:36 4s/step - acc: 0.0101 - loss: 4.8839 - top5-acc: 0.0597

<div class="k-default-codeblock">
```

```
</div>
 121/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:32 4s/step - acc: 0.0101 - loss: 4.8824 - top5-acc: 0.0597

<div class="k-default-codeblock">
```

```
</div>
 122/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:28 4s/step - acc: 0.0101 - loss: 4.8810 - top5-acc: 0.0596

<div class="k-default-codeblock">
```

```
</div>
 123/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:24 4s/step - acc: 0.0101 - loss: 4.8796 - top5-acc: 0.0595

<div class="k-default-codeblock">
```

```
</div>
 124/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:21 4s/step - acc: 0.0101 - loss: 4.8781 - top5-acc: 0.0595

<div class="k-default-codeblock">
```

```
</div>
 125/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:17 4s/step - acc: 0.0101 - loss: 4.8767 - top5-acc: 0.0594

<div class="k-default-codeblock">
```

```
</div>
 126/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:13 4s/step - acc: 0.0101 - loss: 4.8754 - top5-acc: 0.0593

<div class="k-default-codeblock">
```

```
</div>
 127/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:09 4s/step - acc: 0.0101 - loss: 4.8740 - top5-acc: 0.0593

<div class="k-default-codeblock">
```

```
</div>
 128/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:05 4s/step - acc: 0.0101 - loss: 4.8727 - top5-acc: 0.0592

<div class="k-default-codeblock">
```

```
</div>
 129/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:01 4s/step - acc: 0.0101 - loss: 4.8713 - top5-acc: 0.0592

<div class="k-default-codeblock">
```

```
</div>
 130/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:57 4s/step - acc: 0.0101 - loss: 4.8700 - top5-acc: 0.0591

<div class="k-default-codeblock">
```

```
</div>
 131/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:53 4s/step - acc: 0.0101 - loss: 4.8687 - top5-acc: 0.0590

<div class="k-default-codeblock">
```

```
</div>
 132/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:49 4s/step - acc: 0.0101 - loss: 4.8674 - top5-acc: 0.0590

<div class="k-default-codeblock">
```

```
</div>
 133/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:46 4s/step - acc: 0.0102 - loss: 4.8661 - top5-acc: 0.0589

<div class="k-default-codeblock">
```

```
</div>
 134/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:42 4s/step - acc: 0.0102 - loss: 4.8649 - top5-acc: 0.0589

<div class="k-default-codeblock">
```

```
</div>
 135/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:38 4s/step - acc: 0.0102 - loss: 4.8636 - top5-acc: 0.0588

<div class="k-default-codeblock">
```

```
</div>
 136/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:34 4s/step - acc: 0.0102 - loss: 4.8624 - top5-acc: 0.0588

<div class="k-default-codeblock">
```

```
</div>
 137/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:30 4s/step - acc: 0.0102 - loss: 4.8612 - top5-acc: 0.0587

<div class="k-default-codeblock">
```

```
</div>
 138/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:26 4s/step - acc: 0.0102 - loss: 4.8599 - top5-acc: 0.0587

<div class="k-default-codeblock">
```

```
</div>
 139/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:22 4s/step - acc: 0.0102 - loss: 4.8588 - top5-acc: 0.0586

<div class="k-default-codeblock">
```

```
</div>
 140/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:19 4s/step - acc: 0.0102 - loss: 4.8576 - top5-acc: 0.0585

<div class="k-default-codeblock">
```

```
</div>
 141/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:15 4s/step - acc: 0.0102 - loss: 4.8564 - top5-acc: 0.0585

<div class="k-default-codeblock">
```

```
</div>
 142/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:11 4s/step - acc: 0.0102 - loss: 4.8552 - top5-acc: 0.0584

<div class="k-default-codeblock">
```

```
</div>
 143/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:07 4s/step - acc: 0.0102 - loss: 4.8541 - top5-acc: 0.0584

<div class="k-default-codeblock">
```

```
</div>
 144/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:03 4s/step - acc: 0.0102 - loss: 4.8530 - top5-acc: 0.0583

<div class="k-default-codeblock">
```

```
</div>
 145/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:59 4s/step - acc: 0.0102 - loss: 4.8518 - top5-acc: 0.0583

<div class="k-default-codeblock">
```

```
</div>
 146/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:56 4s/step - acc: 0.0102 - loss: 4.8507 - top5-acc: 0.0582

<div class="k-default-codeblock">
```

```
</div>
 147/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:52 4s/step - acc: 0.0102 - loss: 4.8496 - top5-acc: 0.0582

<div class="k-default-codeblock">
```

```
</div>
 148/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:48 4s/step - acc: 0.0102 - loss: 4.8485 - top5-acc: 0.0581

<div class="k-default-codeblock">
```

```
</div>
 149/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:44 4s/step - acc: 0.0102 - loss: 4.8475 - top5-acc: 0.0581

<div class="k-default-codeblock">
```

```
</div>
 150/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:40 4s/step - acc: 0.0102 - loss: 4.8464 - top5-acc: 0.0580

<div class="k-default-codeblock">
```

```
</div>
 151/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:36 4s/step - acc: 0.0102 - loss: 4.8453 - top5-acc: 0.0580

<div class="k-default-codeblock">
```

```
</div>
 152/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:33 4s/step - acc: 0.0102 - loss: 4.8443 - top5-acc: 0.0580

<div class="k-default-codeblock">
```

```
</div>
 153/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:29 4s/step - acc: 0.0102 - loss: 4.8432 - top5-acc: 0.0579

<div class="k-default-codeblock">
```

```
</div>
 154/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:25 4s/step - acc: 0.0102 - loss: 4.8422 - top5-acc: 0.0579

<div class="k-default-codeblock">
```

```
</div>
 155/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:21 4s/step - acc: 0.0102 - loss: 4.8412 - top5-acc: 0.0578

<div class="k-default-codeblock">
```

```
</div>
 156/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:17 4s/step - acc: 0.0102 - loss: 4.8402 - top5-acc: 0.0578

<div class="k-default-codeblock">
```

```
</div>
 157/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:13 4s/step - acc: 0.0102 - loss: 4.8392 - top5-acc: 0.0577

<div class="k-default-codeblock">
```

```
</div>
 158/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:10 4s/step - acc: 0.0102 - loss: 4.8382 - top5-acc: 0.0577

<div class="k-default-codeblock">
```

```
</div>
 159/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:06 4s/step - acc: 0.0102 - loss: 4.8372 - top5-acc: 0.0576

<div class="k-default-codeblock">
```

```
</div>
 160/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:02 4s/step - acc: 0.0102 - loss: 4.8363 - top5-acc: 0.0576

<div class="k-default-codeblock">
```

```
</div>
 161/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:58 4s/step - acc: 0.0102 - loss: 4.8353 - top5-acc: 0.0575

<div class="k-default-codeblock">
```

```
</div>
 162/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:54 4s/step - acc: 0.0102 - loss: 4.8343 - top5-acc: 0.0575

<div class="k-default-codeblock">
```

```
</div>
 163/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:51 4s/step - acc: 0.0102 - loss: 4.8334 - top5-acc: 0.0575

<div class="k-default-codeblock">
```

```
</div>
 164/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:47 4s/step - acc: 0.0102 - loss: 4.8325 - top5-acc: 0.0574

<div class="k-default-codeblock">
```

```
</div>
 165/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:43 4s/step - acc: 0.0102 - loss: 4.8315 - top5-acc: 0.0574

<div class="k-default-codeblock">
```

```
</div>
 166/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:39 4s/step - acc: 0.0102 - loss: 4.8306 - top5-acc: 0.0573

<div class="k-default-codeblock">
```

```
</div>
 167/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:35 4s/step - acc: 0.0102 - loss: 4.8297 - top5-acc: 0.0573

<div class="k-default-codeblock">
```

```
</div>
 168/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:31 4s/step - acc: 0.0102 - loss: 4.8288 - top5-acc: 0.0573

<div class="k-default-codeblock">
```

```
</div>
 169/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:27 4s/step - acc: 0.0102 - loss: 4.8279 - top5-acc: 0.0572

<div class="k-default-codeblock">
```

```
</div>
 170/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:23 4s/step - acc: 0.0102 - loss: 4.8270 - top5-acc: 0.0572

<div class="k-default-codeblock">
```

```
</div>
 171/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:20 4s/step - acc: 0.0102 - loss: 4.8262 - top5-acc: 0.0572

<div class="k-default-codeblock">
```

```
</div>
 172/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:16 4s/step - acc: 0.0102 - loss: 4.8253 - top5-acc: 0.0571

<div class="k-default-codeblock">
```

```
</div>
 173/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:12 4s/step - acc: 0.0102 - loss: 4.8244 - top5-acc: 0.0571

<div class="k-default-codeblock">
```

```
</div>
 174/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:08 4s/step - acc: 0.0102 - loss: 4.8236 - top5-acc: 0.0571

<div class="k-default-codeblock">
```

```
</div>
 175/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:04 4s/step - acc: 0.0102 - loss: 4.8227 - top5-acc: 0.0570

<div class="k-default-codeblock">
```

```
</div>
 176/704 ━━━━━[37m━━━━━━━━━━━━━━━  34:00 4s/step - acc: 0.0102 - loss: 4.8219 - top5-acc: 0.0570

<div class="k-default-codeblock">
```

```
</div>
 177/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:56 4s/step - acc: 0.0102 - loss: 4.8211 - top5-acc: 0.0570

<div class="k-default-codeblock">
```

```
</div>
 178/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:52 4s/step - acc: 0.0102 - loss: 4.8202 - top5-acc: 0.0569

<div class="k-default-codeblock">
```

```
</div>
 179/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:49 4s/step - acc: 0.0102 - loss: 4.8194 - top5-acc: 0.0569

<div class="k-default-codeblock">
```

```
</div>
 180/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:45 4s/step - acc: 0.0102 - loss: 4.8186 - top5-acc: 0.0569

<div class="k-default-codeblock">
```

```
</div>
 181/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:41 4s/step - acc: 0.0102 - loss: 4.8178 - top5-acc: 0.0568

<div class="k-default-codeblock">
```

```
</div>
 182/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:37 4s/step - acc: 0.0102 - loss: 4.8170 - top5-acc: 0.0568

<div class="k-default-codeblock">
```

```
</div>
 183/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:33 4s/step - acc: 0.0102 - loss: 4.8162 - top5-acc: 0.0568

<div class="k-default-codeblock">
```

```
</div>
 184/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:29 4s/step - acc: 0.0102 - loss: 4.8154 - top5-acc: 0.0568

<div class="k-default-codeblock">
```

```
</div>
 185/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:25 4s/step - acc: 0.0102 - loss: 4.8147 - top5-acc: 0.0567

<div class="k-default-codeblock">
```

```
</div>
 186/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:21 4s/step - acc: 0.0102 - loss: 4.8139 - top5-acc: 0.0567

<div class="k-default-codeblock">
```

```
</div>
 187/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:17 4s/step - acc: 0.0102 - loss: 4.8131 - top5-acc: 0.0567

<div class="k-default-codeblock">
```

```
</div>
 188/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:14 4s/step - acc: 0.0102 - loss: 4.8124 - top5-acc: 0.0567

<div class="k-default-codeblock">
```

```
</div>
 189/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:10 4s/step - acc: 0.0102 - loss: 4.8116 - top5-acc: 0.0566

<div class="k-default-codeblock">
```

```
</div>
 190/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:06 4s/step - acc: 0.0102 - loss: 4.8109 - top5-acc: 0.0566

<div class="k-default-codeblock">
```

```
</div>
 191/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:02 4s/step - acc: 0.0102 - loss: 4.8101 - top5-acc: 0.0566

<div class="k-default-codeblock">
```

```
</div>
 192/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:58 4s/step - acc: 0.0102 - loss: 4.8094 - top5-acc: 0.0566

<div class="k-default-codeblock">
```

```
</div>
 193/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:54 4s/step - acc: 0.0102 - loss: 4.8087 - top5-acc: 0.0565

<div class="k-default-codeblock">
```

```
</div>
 194/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:50 4s/step - acc: 0.0102 - loss: 4.8080 - top5-acc: 0.0565

<div class="k-default-codeblock">
```

```
</div>
 195/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:46 4s/step - acc: 0.0102 - loss: 4.8072 - top5-acc: 0.0565

<div class="k-default-codeblock">
```

```
</div>
 196/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:42 4s/step - acc: 0.0102 - loss: 4.8065 - top5-acc: 0.0565

<div class="k-default-codeblock">
```

```
</div>
 197/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:39 4s/step - acc: 0.0102 - loss: 4.8058 - top5-acc: 0.0564

<div class="k-default-codeblock">
```

```
</div>
 198/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:35 4s/step - acc: 0.0102 - loss: 4.8051 - top5-acc: 0.0564

<div class="k-default-codeblock">
```

```
</div>
 199/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:31 4s/step - acc: 0.0102 - loss: 4.8044 - top5-acc: 0.0564

<div class="k-default-codeblock">
```

```
</div>
 200/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:27 4s/step - acc: 0.0102 - loss: 4.8038 - top5-acc: 0.0564

<div class="k-default-codeblock">
```

```
</div>
 201/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:23 4s/step - acc: 0.0102 - loss: 4.8031 - top5-acc: 0.0563

<div class="k-default-codeblock">
```

```
</div>
 202/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:19 4s/step - acc: 0.0102 - loss: 4.8024 - top5-acc: 0.0563

<div class="k-default-codeblock">
```

```
</div>
 203/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:15 4s/step - acc: 0.0102 - loss: 4.8017 - top5-acc: 0.0563

<div class="k-default-codeblock">
```

```
</div>
 204/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:12 4s/step - acc: 0.0102 - loss: 4.8011 - top5-acc: 0.0563

<div class="k-default-codeblock">
```

```
</div>
 205/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:08 4s/step - acc: 0.0102 - loss: 4.8004 - top5-acc: 0.0562

<div class="k-default-codeblock">
```

```
</div>
 206/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:04 4s/step - acc: 0.0102 - loss: 4.7997 - top5-acc: 0.0562

<div class="k-default-codeblock">
```

```
</div>
 207/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:00 4s/step - acc: 0.0102 - loss: 4.7991 - top5-acc: 0.0562

<div class="k-default-codeblock">
```

```
</div>
 208/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:56 4s/step - acc: 0.0102 - loss: 4.7984 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 209/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:52 4s/step - acc: 0.0102 - loss: 4.7978 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 210/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:48 4s/step - acc: 0.0102 - loss: 4.7972 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 211/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:44 4s/step - acc: 0.0102 - loss: 4.7965 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 212/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:41 4s/step - acc: 0.0102 - loss: 4.7959 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 213/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:37 4s/step - acc: 0.0102 - loss: 4.7953 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 214/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:33 4s/step - acc: 0.0102 - loss: 4.7946 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 215/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:29 4s/step - acc: 0.0102 - loss: 4.7940 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 216/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:25 4s/step - acc: 0.0102 - loss: 4.7934 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 217/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:21 4s/step - acc: 0.0102 - loss: 4.7928 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 218/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:18 4s/step - acc: 0.0102 - loss: 4.7922 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 219/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:14 4s/step - acc: 0.0102 - loss: 4.7916 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 220/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:10 4s/step - acc: 0.0102 - loss: 4.7910 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 221/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:06 4s/step - acc: 0.0102 - loss: 4.7904 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 222/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:02 4s/step - acc: 0.0102 - loss: 4.7899 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 223/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:58 4s/step - acc: 0.0102 - loss: 4.7893 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 224/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:54 4s/step - acc: 0.0102 - loss: 4.7887 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 225/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:50 4s/step - acc: 0.0102 - loss: 4.7881 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 226/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:46 4s/step - acc: 0.0102 - loss: 4.7876 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 227/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:43 4s/step - acc: 0.0102 - loss: 4.7870 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 228/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:39 4s/step - acc: 0.0102 - loss: 4.7864 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 229/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:35 4s/step - acc: 0.0102 - loss: 4.7859 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 230/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:31 4s/step - acc: 0.0102 - loss: 4.7853 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 231/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:27 4s/step - acc: 0.0102 - loss: 4.7848 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 232/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:23 4s/step - acc: 0.0102 - loss: 4.7842 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 233/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:19 4s/step - acc: 0.0102 - loss: 4.7837 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 234/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:16 4s/step - acc: 0.0102 - loss: 4.7832 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 235/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:12 4s/step - acc: 0.0102 - loss: 4.7826 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 236/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:08 4s/step - acc: 0.0102 - loss: 4.7821 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 237/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:04 4s/step - acc: 0.0102 - loss: 4.7816 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 238/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:00 4s/step - acc: 0.0102 - loss: 4.7810 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 239/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:56 4s/step - acc: 0.0102 - loss: 4.7805 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 240/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:52 4s/step - acc: 0.0102 - loss: 4.7800 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 241/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:48 4s/step - acc: 0.0102 - loss: 4.7795 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 242/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:44 4s/step - acc: 0.0102 - loss: 4.7790 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 243/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:41 4s/step - acc: 0.0102 - loss: 4.7785 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 244/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:37 4s/step - acc: 0.0102 - loss: 4.7780 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 245/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:33 4s/step - acc: 0.0102 - loss: 4.7775 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 246/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:29 4s/step - acc: 0.0102 - loss: 4.7770 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 247/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:25 4s/step - acc: 0.0102 - loss: 4.7765 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 248/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:21 4s/step - acc: 0.0102 - loss: 4.7760 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 249/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:17 4s/step - acc: 0.0102 - loss: 4.7755 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 250/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:13 4s/step - acc: 0.0102 - loss: 4.7750 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 251/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:10 4s/step - acc: 0.0102 - loss: 4.7745 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 252/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:06 4s/step - acc: 0.0102 - loss: 4.7741 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 253/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:02 4s/step - acc: 0.0102 - loss: 4.7736 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 254/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:58 4s/step - acc: 0.0102 - loss: 4.7731 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 255/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:54 4s/step - acc: 0.0102 - loss: 4.7727 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 256/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:50 4s/step - acc: 0.0102 - loss: 4.7722 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 257/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:46 4s/step - acc: 0.0102 - loss: 4.7717 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 258/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:42 4s/step - acc: 0.0102 - loss: 4.7713 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 259/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:39 4s/step - acc: 0.0102 - loss: 4.7708 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 260/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:35 4s/step - acc: 0.0102 - loss: 4.7704 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 261/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:31 4s/step - acc: 0.0101 - loss: 4.7699 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 262/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:27 4s/step - acc: 0.0101 - loss: 4.7695 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 263/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:23 4s/step - acc: 0.0101 - loss: 4.7690 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 264/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:19 4s/step - acc: 0.0101 - loss: 4.7686 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 265/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:15 4s/step - acc: 0.0101 - loss: 4.7681 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 266/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:11 4s/step - acc: 0.0101 - loss: 4.7677 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 267/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:08 4s/step - acc: 0.0101 - loss: 4.7673 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 268/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:04 4s/step - acc: 0.0101 - loss: 4.7668 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 269/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:00 4s/step - acc: 0.0101 - loss: 4.7664 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 270/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:56 4s/step - acc: 0.0101 - loss: 4.7660 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 271/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:52 4s/step - acc: 0.0101 - loss: 4.7655 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 272/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:48 4s/step - acc: 0.0101 - loss: 4.7651 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 273/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:44 4s/step - acc: 0.0101 - loss: 4.7647 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 274/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:41 4s/step - acc: 0.0101 - loss: 4.7643 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 275/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:37 4s/step - acc: 0.0101 - loss: 4.7639 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 276/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:33 4s/step - acc: 0.0101 - loss: 4.7635 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 277/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:29 4s/step - acc: 0.0101 - loss: 4.7630 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 278/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:25 4s/step - acc: 0.0101 - loss: 4.7626 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 279/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:21 4s/step - acc: 0.0101 - loss: 4.7622 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 280/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:17 4s/step - acc: 0.0101 - loss: 4.7618 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 281/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:14 4s/step - acc: 0.0101 - loss: 4.7614 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 282/704 ━━━━━━━━[37m━━━━━━━━━━━━  27:10 4s/step - acc: 0.0101 - loss: 4.7610 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 283/704 ━━━━━━━━[37m━━━━━━━━━━━━  27:06 4s/step - acc: 0.0101 - loss: 4.7606 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 284/704 ━━━━━━━━[37m━━━━━━━━━━━━  27:02 4s/step - acc: 0.0101 - loss: 4.7602 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 285/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:58 4s/step - acc: 0.0101 - loss: 4.7598 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 286/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:54 4s/step - acc: 0.0101 - loss: 4.7595 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 287/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:50 4s/step - acc: 0.0101 - loss: 4.7591 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 288/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:47 4s/step - acc: 0.0101 - loss: 4.7587 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 289/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:43 4s/step - acc: 0.0101 - loss: 4.7583 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 290/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:39 4s/step - acc: 0.0101 - loss: 4.7579 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 291/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:35 4s/step - acc: 0.0101 - loss: 4.7575 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 292/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:31 4s/step - acc: 0.0101 - loss: 4.7572 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 293/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:27 4s/step - acc: 0.0101 - loss: 4.7568 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 294/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:23 4s/step - acc: 0.0101 - loss: 4.7564 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 295/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:19 4s/step - acc: 0.0101 - loss: 4.7560 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 296/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:16 4s/step - acc: 0.0101 - loss: 4.7557 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 297/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:12 4s/step - acc: 0.0101 - loss: 4.7553 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 298/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:08 4s/step - acc: 0.0101 - loss: 4.7549 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 299/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:04 4s/step - acc: 0.0101 - loss: 4.7546 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 300/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:00 4s/step - acc: 0.0101 - loss: 4.7542 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 301/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:56 4s/step - acc: 0.0101 - loss: 4.7539 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 302/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:52 4s/step - acc: 0.0101 - loss: 4.7535 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 303/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:49 4s/step - acc: 0.0101 - loss: 4.7532 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 304/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:45 4s/step - acc: 0.0101 - loss: 4.7528 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 305/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:41 4s/step - acc: 0.0101 - loss: 4.7524 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 306/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:37 4s/step - acc: 0.0101 - loss: 4.7521 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 307/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:33 4s/step - acc: 0.0101 - loss: 4.7518 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 308/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:29 4s/step - acc: 0.0101 - loss: 4.7514 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 309/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:25 4s/step - acc: 0.0101 - loss: 4.7511 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 310/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:21 4s/step - acc: 0.0101 - loss: 4.7507 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 311/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:18 4s/step - acc: 0.0101 - loss: 4.7504 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 312/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:14 4s/step - acc: 0.0101 - loss: 4.7500 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 313/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:10 4s/step - acc: 0.0101 - loss: 4.7497 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 314/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:06 4s/step - acc: 0.0101 - loss: 4.7494 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 315/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:02 4s/step - acc: 0.0101 - loss: 4.7490 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 316/704 ━━━━━━━━[37m━━━━━━━━━━━━  24:58 4s/step - acc: 0.0101 - loss: 4.7487 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 317/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:54 4s/step - acc: 0.0101 - loss: 4.7484 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 318/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:51 4s/step - acc: 0.0101 - loss: 4.7480 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 319/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:47 4s/step - acc: 0.0101 - loss: 4.7477 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 320/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:43 4s/step - acc: 0.0101 - loss: 4.7474 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 321/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:39 4s/step - acc: 0.0101 - loss: 4.7471 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 322/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:35 4s/step - acc: 0.0101 - loss: 4.7468 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 323/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:31 4s/step - acc: 0.0101 - loss: 4.7464 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 324/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:27 4s/step - acc: 0.0101 - loss: 4.7461 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 325/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:23 4s/step - acc: 0.0101 - loss: 4.7458 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 326/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:20 4s/step - acc: 0.0101 - loss: 4.7455 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 327/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:16 4s/step - acc: 0.0101 - loss: 4.7452 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 328/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:12 4s/step - acc: 0.0101 - loss: 4.7449 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 329/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:08 4s/step - acc: 0.0101 - loss: 4.7445 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 330/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:04 4s/step - acc: 0.0101 - loss: 4.7442 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 331/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:00 4s/step - acc: 0.0101 - loss: 4.7439 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 332/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:56 4s/step - acc: 0.0101 - loss: 4.7436 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 333/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:53 4s/step - acc: 0.0101 - loss: 4.7433 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 334/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:49 4s/step - acc: 0.0101 - loss: 4.7430 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 335/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:45 4s/step - acc: 0.0101 - loss: 4.7427 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 336/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:41 4s/step - acc: 0.0101 - loss: 4.7424 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 337/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:37 4s/step - acc: 0.0101 - loss: 4.7421 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 338/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:33 4s/step - acc: 0.0101 - loss: 4.7418 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 339/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:29 4s/step - acc: 0.0101 - loss: 4.7415 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 340/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:25 4s/step - acc: 0.0101 - loss: 4.7412 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 341/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:22 4s/step - acc: 0.0101 - loss: 4.7409 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 342/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:18 4s/step - acc: 0.0101 - loss: 4.7407 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 343/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:14 4s/step - acc: 0.0101 - loss: 4.7404 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 344/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:10 4s/step - acc: 0.0101 - loss: 4.7401 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 345/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:06 4s/step - acc: 0.0101 - loss: 4.7398 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 346/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:02 4s/step - acc: 0.0101 - loss: 4.7395 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 347/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:58 4s/step - acc: 0.0101 - loss: 4.7392 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 348/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:54 4s/step - acc: 0.0101 - loss: 4.7389 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 349/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:51 4s/step - acc: 0.0101 - loss: 4.7387 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 350/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:47 4s/step - acc: 0.0101 - loss: 4.7384 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 351/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:43 4s/step - acc: 0.0101 - loss: 4.7381 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 352/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:39 4s/step - acc: 0.0101 - loss: 4.7378 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 353/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:35 4s/step - acc: 0.0101 - loss: 4.7375 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 354/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:31 4s/step - acc: 0.0101 - loss: 4.7373 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 355/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:27 4s/step - acc: 0.0101 - loss: 4.7370 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 356/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:24 4s/step - acc: 0.0101 - loss: 4.7367 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 357/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:20 4s/step - acc: 0.0100 - loss: 4.7364 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 358/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:16 4s/step - acc: 0.0100 - loss: 4.7362 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 359/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:12 4s/step - acc: 0.0100 - loss: 4.7359 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 360/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:08 4s/step - acc: 0.0100 - loss: 4.7356 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 361/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:04 4s/step - acc: 0.0100 - loss: 4.7354 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 362/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:00 4s/step - acc: 0.0100 - loss: 4.7351 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 363/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:57 4s/step - acc: 0.0100 - loss: 4.7348 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 364/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:53 4s/step - acc: 0.0100 - loss: 4.7346 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 365/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:49 4s/step - acc: 0.0100 - loss: 4.7343 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 366/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:45 4s/step - acc: 0.0100 - loss: 4.7341 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 367/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:41 4s/step - acc: 0.0100 - loss: 4.7338 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 368/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:37 4s/step - acc: 0.0100 - loss: 4.7335 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 369/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:33 4s/step - acc: 0.0100 - loss: 4.7333 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 370/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:29 4s/step - acc: 0.0100 - loss: 4.7330 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 371/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:26 4s/step - acc: 0.0100 - loss: 4.7328 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 372/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:22 4s/step - acc: 0.0100 - loss: 4.7325 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 373/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:18 4s/step - acc: 0.0100 - loss: 4.7323 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 374/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:14 4s/step - acc: 0.0100 - loss: 4.7320 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 375/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:10 4s/step - acc: 0.0100 - loss: 4.7318 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 376/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:06 4s/step - acc: 0.0100 - loss: 4.7315 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 377/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:02 4s/step - acc: 0.0100 - loss: 4.7313 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 378/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:59 4s/step - acc: 0.0100 - loss: 4.7310 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 379/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:55 4s/step - acc: 0.0100 - loss: 4.7308 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 380/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:51 4s/step - acc: 0.0100 - loss: 4.7305 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 381/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:47 4s/step - acc: 0.0100 - loss: 4.7303 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 382/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:43 4s/step - acc: 0.0100 - loss: 4.7300 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 383/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:40 4s/step - acc: 0.0100 - loss: 4.7298 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 384/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:36 4s/step - acc: 0.0100 - loss: 4.7296 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 385/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:32 4s/step - acc: 0.0100 - loss: 4.7293 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 386/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:28 4s/step - acc: 0.0100 - loss: 4.7291 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 387/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:24 4s/step - acc: 0.0100 - loss: 4.7288 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 388/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:20 4s/step - acc: 0.0100 - loss: 4.7286 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 389/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:16 4s/step - acc: 0.0100 - loss: 4.7284 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 390/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:12 4s/step - acc: 0.0100 - loss: 4.7281 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 391/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:09 4s/step - acc: 0.0100 - loss: 4.7279 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 392/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:05 4s/step - acc: 0.0100 - loss: 4.7277 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 393/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:01 4s/step - acc: 0.0100 - loss: 4.7274 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 394/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:57 4s/step - acc: 0.0100 - loss: 4.7272 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 395/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:53 4s/step - acc: 0.0100 - loss: 4.7270 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 396/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:49 4s/step - acc: 0.0100 - loss: 4.7268 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 397/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:45 4s/step - acc: 0.0100 - loss: 4.7265 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 398/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:41 4s/step - acc: 0.0100 - loss: 4.7263 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 399/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:38 4s/step - acc: 0.0100 - loss: 4.7261 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 400/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:34 4s/step - acc: 0.0100 - loss: 4.7259 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 401/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:30 4s/step - acc: 0.0100 - loss: 4.7256 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 402/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:26 4s/step - acc: 0.0100 - loss: 4.7254 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 403/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:22 4s/step - acc: 0.0100 - loss: 4.7252 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 404/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:18 4s/step - acc: 0.0100 - loss: 4.7250 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 405/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:14 4s/step - acc: 0.0100 - loss: 4.7247 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 406/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:10 4s/step - acc: 0.0100 - loss: 4.7245 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 407/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:07 4s/step - acc: 0.0100 - loss: 4.7243 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 408/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:03 4s/step - acc: 0.0100 - loss: 4.7241 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 409/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:59 4s/step - acc: 0.0100 - loss: 4.7239 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 410/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:55 4s/step - acc: 0.0100 - loss: 4.7237 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 411/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:51 4s/step - acc: 0.0100 - loss: 4.7234 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 412/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:47 4s/step - acc: 0.0100 - loss: 4.7232 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 413/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:43 4s/step - acc: 0.0100 - loss: 4.7230 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 414/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:40 4s/step - acc: 0.0100 - loss: 4.7228 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 415/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:36 4s/step - acc: 0.0100 - loss: 4.7226 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 416/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:32 4s/step - acc: 0.0100 - loss: 4.7224 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 417/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:28 4s/step - acc: 0.0100 - loss: 4.7222 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 418/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:24 4s/step - acc: 0.0100 - loss: 4.7220 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 419/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:20 4s/step - acc: 0.0100 - loss: 4.7218 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 420/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:16 4s/step - acc: 0.0100 - loss: 4.7215 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 421/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:12 4s/step - acc: 0.0100 - loss: 4.7213 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 422/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:09 4s/step - acc: 0.0100 - loss: 4.7211 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 423/704 ━━━━━━━━━━━━[37m━━━━━━━━  18:05 4s/step - acc: 0.0100 - loss: 4.7209 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 424/704 ━━━━━━━━━━━━[37m━━━━━━━━  18:01 4s/step - acc: 0.0100 - loss: 4.7207 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 425/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:57 4s/step - acc: 0.0100 - loss: 4.7205 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 426/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:53 4s/step - acc: 0.0100 - loss: 4.7203 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 427/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:49 4s/step - acc: 0.0100 - loss: 4.7201 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 428/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:46 4s/step - acc: 0.0100 - loss: 4.7199 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 429/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:42 4s/step - acc: 0.0100 - loss: 4.7197 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 430/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:38 4s/step - acc: 0.0100 - loss: 4.7195 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 431/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:35 4s/step - acc: 0.0100 - loss: 4.7193 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 432/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:31 4s/step - acc: 0.0100 - loss: 4.7191 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 433/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:27 4s/step - acc: 0.0100 - loss: 4.7189 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 434/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:23 4s/step - acc: 0.0100 - loss: 4.7187 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 435/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:20 4s/step - acc: 0.0100 - loss: 4.7185 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 436/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:16 4s/step - acc: 0.0100 - loss: 4.7183 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 437/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:12 4s/step - acc: 0.0100 - loss: 4.7181 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 438/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:08 4s/step - acc: 0.0100 - loss: 4.7180 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 439/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:05 4s/step - acc: 0.0100 - loss: 4.7178 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 440/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:01 4s/step - acc: 0.0100 - loss: 4.7176 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 441/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:57 4s/step - acc: 0.0100 - loss: 4.7174 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 442/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:54 4s/step - acc: 0.0100 - loss: 4.7172 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 443/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:50 4s/step - acc: 0.0100 - loss: 4.7170 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 444/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:46 4s/step - acc: 0.0100 - loss: 4.7168 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 445/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:43 4s/step - acc: 0.0100 - loss: 4.7166 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 446/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:39 4s/step - acc: 0.0100 - loss: 4.7164 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 447/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:35 4s/step - acc: 0.0100 - loss: 4.7162 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 448/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:31 4s/step - acc: 0.0100 - loss: 4.7161 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 449/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:28 4s/step - acc: 0.0100 - loss: 4.7159 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 450/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:24 4s/step - acc: 0.0100 - loss: 4.7157 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 451/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:20 4s/step - acc: 0.0100 - loss: 4.7155 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 452/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:16 4s/step - acc: 0.0100 - loss: 4.7153 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 453/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:13 4s/step - acc: 0.0100 - loss: 4.7151 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 454/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:09 4s/step - acc: 0.0100 - loss: 4.7150 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 455/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:05 4s/step - acc: 0.0100 - loss: 4.7148 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 456/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:01 4s/step - acc: 0.0100 - loss: 4.7146 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 457/704 ━━━━━━━━━━━━[37m━━━━━━━━  15:57 4s/step - acc: 0.0100 - loss: 4.7144 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 458/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:53 4s/step - acc: 0.0100 - loss: 4.7142 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 459/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:50 4s/step - acc: 0.0100 - loss: 4.7141 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 460/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:46 4s/step - acc: 0.0100 - loss: 4.7139 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 461/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:42 4s/step - acc: 0.0100 - loss: 4.7137 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 462/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:38 4s/step - acc: 0.0100 - loss: 4.7135 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 463/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:34 4s/step - acc: 0.0100 - loss: 4.7134 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 464/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:31 4s/step - acc: 0.0100 - loss: 4.7132 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 465/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:27 4s/step - acc: 0.0100 - loss: 4.7130 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 466/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:23 4s/step - acc: 0.0100 - loss: 4.7128 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 467/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:19 4s/step - acc: 0.0100 - loss: 4.7127 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 468/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:15 4s/step - acc: 0.0100 - loss: 4.7125 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 469/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:12 4s/step - acc: 0.0100 - loss: 4.7123 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 470/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:08 4s/step - acc: 0.0100 - loss: 4.7121 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 471/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:04 4s/step - acc: 0.0100 - loss: 4.7120 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 472/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:00 4s/step - acc: 0.0100 - loss: 4.7118 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 473/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:56 4s/step - acc: 0.0100 - loss: 4.7116 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 474/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:52 4s/step - acc: 0.0100 - loss: 4.7115 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 475/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:49 4s/step - acc: 0.0100 - loss: 4.7113 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 476/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:45 4s/step - acc: 0.0100 - loss: 4.7111 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 477/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:41 4s/step - acc: 0.0100 - loss: 4.7110 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 478/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:37 4s/step - acc: 0.0100 - loss: 4.7108 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 479/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:33 4s/step - acc: 0.0100 - loss: 4.7106 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 480/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:29 4s/step - acc: 0.0100 - loss: 4.7105 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 481/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:26 4s/step - acc: 0.0100 - loss: 4.7103 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 482/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:22 4s/step - acc: 0.0100 - loss: 4.7101 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 483/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:18 4s/step - acc: 0.0100 - loss: 4.7100 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 484/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:14 4s/step - acc: 0.0100 - loss: 4.7098 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 485/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:10 4s/step - acc: 0.0100 - loss: 4.7096 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 486/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:06 4s/step - acc: 0.0100 - loss: 4.7095 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 487/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:03 4s/step - acc: 0.0100 - loss: 4.7093 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 488/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:59 4s/step - acc: 0.0100 - loss: 4.7092 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 489/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:55 4s/step - acc: 0.0100 - loss: 4.7090 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 490/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:51 4s/step - acc: 0.0100 - loss: 4.7088 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 491/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:47 4s/step - acc: 0.0100 - loss: 4.7087 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 492/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:43 4s/step - acc: 0.0100 - loss: 4.7085 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 493/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:40 4s/step - acc: 0.0100 - loss: 4.7084 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 494/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:36 4s/step - acc: 0.0100 - loss: 4.7082 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 495/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:32 4s/step - acc: 0.0100 - loss: 4.7080 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 496/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:28 4s/step - acc: 0.0100 - loss: 4.7079 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 497/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:24 4s/step - acc: 0.0100 - loss: 4.7077 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 498/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:20 4s/step - acc: 0.0100 - loss: 4.7076 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 499/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:17 4s/step - acc: 0.0100 - loss: 4.7074 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 500/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:13 4s/step - acc: 0.0100 - loss: 4.7073 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 501/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:09 4s/step - acc: 0.0100 - loss: 4.7071 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 502/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:05 4s/step - acc: 0.0100 - loss: 4.7070 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 503/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:01 4s/step - acc: 0.0100 - loss: 4.7068 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 504/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:57 4s/step - acc: 0.0100 - loss: 4.7066 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 505/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:54 4s/step - acc: 0.0100 - loss: 4.7065 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 506/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:50 4s/step - acc: 0.0100 - loss: 4.7063 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 507/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:46 4s/step - acc: 0.0100 - loss: 4.7062 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 508/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:42 4s/step - acc: 0.0100 - loss: 4.7060 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 509/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:38 4s/step - acc: 0.0100 - loss: 4.7059 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 510/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:34 4s/step - acc: 0.0100 - loss: 4.7057 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 511/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:30 4s/step - acc: 0.0100 - loss: 4.7056 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 512/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:27 4s/step - acc: 0.0100 - loss: 4.7054 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 513/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:23 4s/step - acc: 0.0100 - loss: 4.7053 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 514/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:19 4s/step - acc: 0.0100 - loss: 4.7051 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 515/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:15 4s/step - acc: 0.0100 - loss: 4.7050 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 516/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:11 4s/step - acc: 0.0100 - loss: 4.7049 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 517/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:07 4s/step - acc: 0.0100 - loss: 4.7047 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 518/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:04 4s/step - acc: 0.0100 - loss: 4.7046 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 519/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:00 4s/step - acc: 0.0100 - loss: 4.7044 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 520/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:56 4s/step - acc: 0.0100 - loss: 4.7043 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 521/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:52 4s/step - acc: 0.0100 - loss: 4.7041 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 522/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:48 4s/step - acc: 0.0100 - loss: 4.7040 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 523/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:44 4s/step - acc: 0.0100 - loss: 4.7038 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 524/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:41 4s/step - acc: 0.0100 - loss: 4.7037 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 525/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:37 4s/step - acc: 0.0100 - loss: 4.7036 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 526/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:33 4s/step - acc: 0.0100 - loss: 4.7034 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 527/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:29 4s/step - acc: 0.0100 - loss: 4.7033 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 528/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:25 4s/step - acc: 0.0100 - loss: 4.7031 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 529/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:21 4s/step - acc: 0.0100 - loss: 4.7030 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 530/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:17 4s/step - acc: 0.0100 - loss: 4.7029 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 531/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:14 4s/step - acc: 0.0100 - loss: 4.7027 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 532/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:10 4s/step - acc: 0.0100 - loss: 4.7026 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 533/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:06 4s/step - acc: 0.0100 - loss: 4.7024 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 534/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:02 4s/step - acc: 0.0100 - loss: 4.7023 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 535/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:58 4s/step - acc: 0.0100 - loss: 4.7022 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 536/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:54 4s/step - acc: 0.0100 - loss: 4.7020 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 537/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:50 4s/step - acc: 0.0100 - loss: 4.7019 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 538/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:47 4s/step - acc: 0.0100 - loss: 4.7017 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 539/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:43 4s/step - acc: 0.0100 - loss: 4.7016 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 540/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:39 4s/step - acc: 0.0100 - loss: 4.7015 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 541/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:35 4s/step - acc: 0.0100 - loss: 4.7013 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 542/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:31 4s/step - acc: 0.0100 - loss: 4.7012 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 543/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:27 4s/step - acc: 0.0100 - loss: 4.7011 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 544/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:23 4s/step - acc: 0.0100 - loss: 4.7009 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 545/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:19 4s/step - acc: 0.0100 - loss: 4.7008 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 546/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:16 4s/step - acc: 0.0100 - loss: 4.7007 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 547/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:12 4s/step - acc: 0.0100 - loss: 4.7005 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 548/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:08 4s/step - acc: 0.0100 - loss: 4.7004 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 549/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:04 4s/step - acc: 0.0100 - loss: 4.7003 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 550/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:00 4s/step - acc: 0.0100 - loss: 4.7001 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 551/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:56 4s/step - acc: 0.0100 - loss: 4.7000 - top5-acc: 0.0524 

<div class="k-default-codeblock">
```

```
</div>
 552/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:52 4s/step - acc: 0.0100 - loss: 4.6999 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 553/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:49 4s/step - acc: 0.0100 - loss: 4.6997 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 554/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:45 4s/step - acc: 0.0100 - loss: 4.6996 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 555/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:41 4s/step - acc: 0.0100 - loss: 4.6995 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 556/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:37 4s/step - acc: 0.0100 - loss: 4.6994 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 557/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:33 4s/step - acc: 0.0100 - loss: 4.6992 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 558/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:29 4s/step - acc: 0.0100 - loss: 4.6991 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 559/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:26 4s/step - acc: 0.0100 - loss: 4.6990 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 560/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:22 4s/step - acc: 0.0100 - loss: 4.6988 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 561/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:18 4s/step - acc: 0.0100 - loss: 4.6987 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 562/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:14 4s/step - acc: 0.0100 - loss: 4.6986 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 563/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:10 4s/step - acc: 0.0100 - loss: 4.6985 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 564/704 ━━━━━━━━━━━━━━━━[37m━━━━  9:07 4s/step - acc: 0.0100 - loss: 4.6983 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 565/704 ━━━━━━━━━━━━━━━━[37m━━━━  9:03 4s/step - acc: 0.0100 - loss: 4.6982 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 566/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:59 4s/step - acc: 0.0100 - loss: 4.6981 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 567/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:55 4s/step - acc: 0.0100 - loss: 4.6980 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 568/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:51 4s/step - acc: 0.0100 - loss: 4.6978 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 569/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:47 4s/step - acc: 0.0100 - loss: 4.6977 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 570/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:43 4s/step - acc: 0.0100 - loss: 4.6976 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 571/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:40 4s/step - acc: 0.0100 - loss: 4.6975 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 572/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:36 4s/step - acc: 0.0100 - loss: 4.6973 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 573/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:32 4s/step - acc: 0.0100 - loss: 4.6972 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 574/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:28 4s/step - acc: 0.0100 - loss: 4.6971 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 575/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:24 4s/step - acc: 0.0100 - loss: 4.6970 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 576/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:20 4s/step - acc: 0.0100 - loss: 4.6969 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 577/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:16 4s/step - acc: 0.0100 - loss: 4.6967 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 578/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:13 4s/step - acc: 0.0100 - loss: 4.6966 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 579/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:09 4s/step - acc: 0.0100 - loss: 4.6965 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 580/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:05 4s/step - acc: 0.0100 - loss: 4.6964 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 581/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:01 4s/step - acc: 0.0100 - loss: 4.6963 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 582/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:57 4s/step - acc: 0.0100 - loss: 4.6961 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 583/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:53 4s/step - acc: 0.0100 - loss: 4.6960 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 584/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:49 4s/step - acc: 0.0100 - loss: 4.6959 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 585/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:46 4s/step - acc: 0.0100 - loss: 4.6958 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 586/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:42 4s/step - acc: 0.0100 - loss: 4.6957 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 587/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:38 4s/step - acc: 0.0100 - loss: 4.6955 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 588/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:34 4s/step - acc: 0.0100 - loss: 4.6954 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 589/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:30 4s/step - acc: 0.0100 - loss: 4.6953 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 590/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:26 4s/step - acc: 0.0100 - loss: 4.6952 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 591/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:22 4s/step - acc: 0.0100 - loss: 4.6951 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 592/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:18 4s/step - acc: 0.0100 - loss: 4.6950 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 593/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:15 4s/step - acc: 0.0100 - loss: 4.6948 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 594/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:11 4s/step - acc: 0.0100 - loss: 4.6947 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 595/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:07 4s/step - acc: 0.0100 - loss: 4.6946 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 596/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:03 4s/step - acc: 0.0100 - loss: 4.6945 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 597/704 ━━━━━━━━━━━━━━━━[37m━━━━  6:59 4s/step - acc: 0.0100 - loss: 4.6944 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 598/704 ━━━━━━━━━━━━━━━━[37m━━━━  6:55 4s/step - acc: 0.0100 - loss: 4.6943 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 599/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:51 4s/step - acc: 0.0100 - loss: 4.6942 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 600/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:47 4s/step - acc: 0.0100 - loss: 4.6940 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 601/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:44 4s/step - acc: 0.0100 - loss: 4.6939 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 602/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:40 4s/step - acc: 0.0100 - loss: 4.6938 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 603/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:36 4s/step - acc: 0.0100 - loss: 4.6937 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 604/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:32 4s/step - acc: 0.0099 - loss: 4.6936 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 605/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:28 4s/step - acc: 0.0099 - loss: 4.6935 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 606/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:24 4s/step - acc: 0.0099 - loss: 4.6934 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 607/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:20 4s/step - acc: 0.0099 - loss: 4.6933 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 608/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:16 4s/step - acc: 0.0099 - loss: 4.6931 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 609/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:12 4s/step - acc: 0.0099 - loss: 4.6930 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 610/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:09 4s/step - acc: 0.0099 - loss: 4.6929 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 611/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:05 4s/step - acc: 0.0099 - loss: 4.6928 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 612/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:01 4s/step - acc: 0.0099 - loss: 4.6927 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 613/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:57 4s/step - acc: 0.0099 - loss: 4.6926 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 614/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:53 4s/step - acc: 0.0099 - loss: 4.6925 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 615/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:49 4s/step - acc: 0.0099 - loss: 4.6924 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 616/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:45 4s/step - acc: 0.0099 - loss: 4.6923 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 617/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:41 4s/step - acc: 0.0099 - loss: 4.6922 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 618/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:37 4s/step - acc: 0.0099 - loss: 4.6920 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 619/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:34 4s/step - acc: 0.0099 - loss: 4.6919 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 620/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:30 4s/step - acc: 0.0099 - loss: 4.6918 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 621/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:26 4s/step - acc: 0.0099 - loss: 4.6917 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 622/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:22 4s/step - acc: 0.0099 - loss: 4.6916 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 623/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:18 4s/step - acc: 0.0099 - loss: 4.6915 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 624/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:14 4s/step - acc: 0.0099 - loss: 4.6914 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 625/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:10 4s/step - acc: 0.0099 - loss: 4.6913 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 626/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:06 4s/step - acc: 0.0099 - loss: 4.6912 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 627/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:02 4s/step - acc: 0.0099 - loss: 4.6911 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 628/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:58 4s/step - acc: 0.0099 - loss: 4.6910 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 629/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:55 4s/step - acc: 0.0099 - loss: 4.6909 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 630/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:51 4s/step - acc: 0.0099 - loss: 4.6908 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 631/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:47 4s/step - acc: 0.0099 - loss: 4.6907 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 632/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:43 4s/step - acc: 0.0099 - loss: 4.6906 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 633/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:39 4s/step - acc: 0.0099 - loss: 4.6905 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 634/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:35 4s/step - acc: 0.0099 - loss: 4.6904 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 635/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:31 4s/step - acc: 0.0099 - loss: 4.6903 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 636/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:27 4s/step - acc: 0.0099 - loss: 4.6902 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 637/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:23 4s/step - acc: 0.0099 - loss: 4.6901 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 638/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:19 4s/step - acc: 0.0099 - loss: 4.6900 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 639/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:15 4s/step - acc: 0.0099 - loss: 4.6898 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 640/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:11 4s/step - acc: 0.0099 - loss: 4.6897 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 641/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:08 4s/step - acc: 0.0099 - loss: 4.6896 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 642/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:04 4s/step - acc: 0.0099 - loss: 4.6895 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 643/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:00 4s/step - acc: 0.0099 - loss: 4.6894 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 644/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:56 4s/step - acc: 0.0099 - loss: 4.6893 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 645/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:52 4s/step - acc: 0.0099 - loss: 4.6892 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 646/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:48 4s/step - acc: 0.0099 - loss: 4.6891 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 647/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:44 4s/step - acc: 0.0099 - loss: 4.6890 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 648/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:40 4s/step - acc: 0.0099 - loss: 4.6889 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 649/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:36 4s/step - acc: 0.0099 - loss: 4.6888 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 650/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:32 4s/step - acc: 0.0099 - loss: 4.6887 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 651/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:28 4s/step - acc: 0.0099 - loss: 4.6886 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 652/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:24 4s/step - acc: 0.0099 - loss: 4.6885 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 653/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:20 4s/step - acc: 0.0099 - loss: 4.6885 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 654/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:17 4s/step - acc: 0.0099 - loss: 4.6884 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 655/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:13 4s/step - acc: 0.0099 - loss: 4.6883 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 656/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:09 4s/step - acc: 0.0099 - loss: 4.6882 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 657/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:05 4s/step - acc: 0.0099 - loss: 4.6881 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 658/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:01 4s/step - acc: 0.0099 - loss: 4.6880 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 659/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:57 4s/step - acc: 0.0099 - loss: 4.6879 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 660/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:53 4s/step - acc: 0.0099 - loss: 4.6878 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 661/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:49 4s/step - acc: 0.0099 - loss: 4.6877 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 662/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:45 4s/step - acc: 0.0099 - loss: 4.6876 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 663/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:41 4s/step - acc: 0.0099 - loss: 4.6875 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 664/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:37 4s/step - acc: 0.0099 - loss: 4.6874 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 665/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:33 4s/step - acc: 0.0099 - loss: 4.6873 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 666/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:29 4s/step - acc: 0.0099 - loss: 4.6872 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 667/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:25 4s/step - acc: 0.0099 - loss: 4.6871 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 668/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:21 4s/step - acc: 0.0099 - loss: 4.6870 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 669/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:17 4s/step - acc: 0.0099 - loss: 4.6869 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 670/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:13 4s/step - acc: 0.0099 - loss: 4.6868 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 671/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:10 4s/step - acc: 0.0099 - loss: 4.6867 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 672/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:06 4s/step - acc: 0.0099 - loss: 4.6866 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 673/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:02 4s/step - acc: 0.0099 - loss: 4.6865 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 674/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:58 4s/step - acc: 0.0099 - loss: 4.6864 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 675/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:54 4s/step - acc: 0.0099 - loss: 4.6864 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 676/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:50 4s/step - acc: 0.0099 - loss: 4.6863 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 677/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:46 4s/step - acc: 0.0099 - loss: 4.6862 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 678/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:42 4s/step - acc: 0.0099 - loss: 4.6861 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 679/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:38 4s/step - acc: 0.0099 - loss: 4.6860 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 680/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:34 4s/step - acc: 0.0099 - loss: 4.6859 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 681/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:30 4s/step - acc: 0.0099 - loss: 4.6858 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 682/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:26 4s/step - acc: 0.0099 - loss: 4.6857 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 683/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:22 4s/step - acc: 0.0099 - loss: 4.6856 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 684/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:18 4s/step - acc: 0.0099 - loss: 4.6855 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 685/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:14 4s/step - acc: 0.0099 - loss: 4.6854 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 686/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:10 4s/step - acc: 0.0099 - loss: 4.6853 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 687/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:06 4s/step - acc: 0.0099 - loss: 4.6853 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 688/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:03 4s/step - acc: 0.0099 - loss: 4.6852 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 689/704 ━━━━━━━━━━━━━━━━━━━[37m━  59s 4s/step - acc: 0.0099 - loss: 4.6851 - top5-acc: 0.0517 

<div class="k-default-codeblock">
```

```
</div>
 690/704 ━━━━━━━━━━━━━━━━━━━[37m━  55s 4s/step - acc: 0.0099 - loss: 4.6850 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 691/704 ━━━━━━━━━━━━━━━━━━━[37m━  51s 4s/step - acc: 0.0099 - loss: 4.6849 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 692/704 ━━━━━━━━━━━━━━━━━━━[37m━  47s 4s/step - acc: 0.0099 - loss: 4.6848 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 693/704 ━━━━━━━━━━━━━━━━━━━[37m━  43s 4s/step - acc: 0.0099 - loss: 4.6847 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 694/704 ━━━━━━━━━━━━━━━━━━━[37m━  39s 4s/step - acc: 0.0099 - loss: 4.6846 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 695/704 ━━━━━━━━━━━━━━━━━━━[37m━  35s 4s/step - acc: 0.0099 - loss: 4.6845 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 696/704 ━━━━━━━━━━━━━━━━━━━[37m━  31s 4s/step - acc: 0.0099 - loss: 4.6845 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 697/704 ━━━━━━━━━━━━━━━━━━━[37m━  27s 4s/step - acc: 0.0099 - loss: 4.6844 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 698/704 ━━━━━━━━━━━━━━━━━━━[37m━  23s 4s/step - acc: 0.0099 - loss: 4.6843 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 699/704 ━━━━━━━━━━━━━━━━━━━[37m━  19s 4s/step - acc: 0.0099 - loss: 4.6842 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 700/704 ━━━━━━━━━━━━━━━━━━━[37m━  15s 4s/step - acc: 0.0099 - loss: 4.6841 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 701/704 ━━━━━━━━━━━━━━━━━━━[37m━  11s 4s/step - acc: 0.0099 - loss: 4.6840 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 702/704 ━━━━━━━━━━━━━━━━━━━[37m━  7s 4s/step - acc: 0.0099 - loss: 4.6839 - top5-acc: 0.0517 

<div class="k-default-codeblock">
```

```
</div>
 703/704 ━━━━━━━━━━━━━━━━━━━[37m━  3s 4s/step - acc: 0.0099 - loss: 4.6838 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - acc: 0.0099 - loss: 4.6838 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 704/704 ━━━━━━━━━━━━━━━━━━━━ 2887s 4s/step - acc: 0.0099 - loss: 4.6837 - top5-acc: 0.0517 - val_acc: 0.0082 - val_loss: 4.6066 - val_top5-acc: 0.0440 - learning_rate: 0.0010


<div class="k-default-codeblock">
```
Epoch 2/2

```
</div>
    
   1/704 [37m━━━━━━━━━━━━━━━━━━━━  45:26 4s/step - acc: 0.0625 - loss: 4.6016 - top5-acc: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   2/704 [37m━━━━━━━━━━━━━━━━━━━━  44:59 4s/step - acc: 0.0547 - loss: 4.6021 - top5-acc: 0.0938

<div class="k-default-codeblock">
```

```
</div>
   3/704 [37m━━━━━━━━━━━━━━━━━━━━  45:05 4s/step - acc: 0.0486 - loss: 4.6027 - top5-acc: 0.0868

<div class="k-default-codeblock">
```

```
</div>
   4/704 [37m━━━━━━━━━━━━━━━━━━━━  45:01 4s/step - acc: 0.0443 - loss: 4.6031 - top5-acc: 0.0817

<div class="k-default-codeblock">
```

```
</div>
   5/704 [37m━━━━━━━━━━━━━━━━━━━━  44:50 4s/step - acc: 0.0410 - loss: 4.6034 - top5-acc: 0.0779

<div class="k-default-codeblock">
```

```
</div>
   6/704 [37m━━━━━━━━━━━━━━━━━━━━  44:45 4s/step - acc: 0.0385 - loss: 4.6036 - top5-acc: 0.0744

<div class="k-default-codeblock">
```

```
</div>
   7/704 [37m━━━━━━━━━━━━━━━━━━━━  44:42 4s/step - acc: 0.0365 - loss: 4.6037 - top5-acc: 0.0715

<div class="k-default-codeblock">
```

```
</div>
   8/704 [37m━━━━━━━━━━━━━━━━━━━━  44:40 4s/step - acc: 0.0347 - loss: 4.6039 - top5-acc: 0.0689

<div class="k-default-codeblock">
```

```
</div>
   9/704 [37m━━━━━━━━━━━━━━━━━━━━  44:39 4s/step - acc: 0.0331 - loss: 4.6040 - top5-acc: 0.0674

<div class="k-default-codeblock">
```

```
</div>
  10/704 [37m━━━━━━━━━━━━━━━━━━━━  44:38 4s/step - acc: 0.0317 - loss: 4.6041 - top5-acc: 0.0661

<div class="k-default-codeblock">
```

```
</div>
  11/704 [37m━━━━━━━━━━━━━━━━━━━━  44:32 4s/step - acc: 0.0305 - loss: 4.6041 - top5-acc: 0.0653

<div class="k-default-codeblock">
```

```
</div>
  12/704 [37m━━━━━━━━━━━━━━━━━━━━  44:27 4s/step - acc: 0.0296 - loss: 4.6042 - top5-acc: 0.0649

<div class="k-default-codeblock">
```

```
</div>
  13/704 [37m━━━━━━━━━━━━━━━━━━━━  44:24 4s/step - acc: 0.0288 - loss: 4.6042 - top5-acc: 0.0646

<div class="k-default-codeblock">
```

```
</div>
  14/704 [37m━━━━━━━━━━━━━━━━━━━━  44:21 4s/step - acc: 0.0280 - loss: 4.6043 - top5-acc: 0.0643

<div class="k-default-codeblock">
```

```
</div>
  15/704 [37m━━━━━━━━━━━━━━━━━━━━  44:16 4s/step - acc: 0.0273 - loss: 4.6043 - top5-acc: 0.0639

<div class="k-default-codeblock">
```

```
</div>
  16/704 [37m━━━━━━━━━━━━━━━━━━━━  44:14 4s/step - acc: 0.0268 - loss: 4.6044 - top5-acc: 0.0637

<div class="k-default-codeblock">
```

```
</div>
  17/704 [37m━━━━━━━━━━━━━━━━━━━━  44:09 4s/step - acc: 0.0263 - loss: 4.6044 - top5-acc: 0.0635

<div class="k-default-codeblock">
```

```
</div>
  18/704 [37m━━━━━━━━━━━━━━━━━━━━  44:06 4s/step - acc: 0.0259 - loss: 4.6044 - top5-acc: 0.0633

<div class="k-default-codeblock">
```

```
</div>
  19/704 [37m━━━━━━━━━━━━━━━━━━━━  44:02 4s/step - acc: 0.0256 - loss: 4.6045 - top5-acc: 0.0632

<div class="k-default-codeblock">
```

```
</div>
  20/704 [37m━━━━━━━━━━━━━━━━━━━━  43:58 4s/step - acc: 0.0252 - loss: 4.6045 - top5-acc: 0.0630

<div class="k-default-codeblock">
```

```
</div>
  21/704 [37m━━━━━━━━━━━━━━━━━━━━  43:54 4s/step - acc: 0.0248 - loss: 4.6045 - top5-acc: 0.0628

<div class="k-default-codeblock">
```

```
</div>
  22/704 [37m━━━━━━━━━━━━━━━━━━━━  43:50 4s/step - acc: 0.0246 - loss: 4.6045 - top5-acc: 0.0627

<div class="k-default-codeblock">
```

```
</div>
  23/704 [37m━━━━━━━━━━━━━━━━━━━━  43:46 4s/step - acc: 0.0244 - loss: 4.6045 - top5-acc: 0.0627

<div class="k-default-codeblock">
```

```
</div>
  24/704 [37m━━━━━━━━━━━━━━━━━━━━  43:43 4s/step - acc: 0.0242 - loss: 4.6045 - top5-acc: 0.0625

<div class="k-default-codeblock">
```

```
</div>
  25/704 [37m━━━━━━━━━━━━━━━━━━━━  43:39 4s/step - acc: 0.0240 - loss: 4.6046 - top5-acc: 0.0624

<div class="k-default-codeblock">
```

```
</div>
  26/704 [37m━━━━━━━━━━━━━━━━━━━━  43:36 4s/step - acc: 0.0238 - loss: 4.6046 - top5-acc: 0.0622

<div class="k-default-codeblock">
```

```
</div>
  27/704 [37m━━━━━━━━━━━━━━━━━━━━  43:33 4s/step - acc: 0.0236 - loss: 4.6046 - top5-acc: 0.0621

<div class="k-default-codeblock">
```

```
</div>
  28/704 [37m━━━━━━━━━━━━━━━━━━━━  43:29 4s/step - acc: 0.0233 - loss: 4.6046 - top5-acc: 0.0620

<div class="k-default-codeblock">
```

```
</div>
  29/704 [37m━━━━━━━━━━━━━━━━━━━━  43:25 4s/step - acc: 0.0231 - loss: 4.6046 - top5-acc: 0.0619

<div class="k-default-codeblock">
```

```
</div>
  30/704 [37m━━━━━━━━━━━━━━━━━━━━  43:22 4s/step - acc: 0.0229 - loss: 4.6046 - top5-acc: 0.0618

<div class="k-default-codeblock">
```

```
</div>
  31/704 [37m━━━━━━━━━━━━━━━━━━━━  43:18 4s/step - acc: 0.0227 - loss: 4.6046 - top5-acc: 0.0617

<div class="k-default-codeblock">
```

```
</div>
  32/704 [37m━━━━━━━━━━━━━━━━━━━━  43:15 4s/step - acc: 0.0225 - loss: 4.6047 - top5-acc: 0.0616

<div class="k-default-codeblock">
```

```
</div>
  33/704 [37m━━━━━━━━━━━━━━━━━━━━  43:10 4s/step - acc: 0.0222 - loss: 4.6047 - top5-acc: 0.0615

<div class="k-default-codeblock">
```

```
</div>
  34/704 [37m━━━━━━━━━━━━━━━━━━━━  43:06 4s/step - acc: 0.0220 - loss: 4.6047 - top5-acc: 0.0614

<div class="k-default-codeblock">
```

```
</div>
  35/704 [37m━━━━━━━━━━━━━━━━━━━━  43:03 4s/step - acc: 0.0218 - loss: 4.6047 - top5-acc: 0.0614

<div class="k-default-codeblock">
```

```
</div>
  36/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:59 4s/step - acc: 0.0216 - loss: 4.6047 - top5-acc: 0.0613

<div class="k-default-codeblock">
```

```
</div>
  37/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:55 4s/step - acc: 0.0214 - loss: 4.6047 - top5-acc: 0.0612

<div class="k-default-codeblock">
```

```
</div>
  38/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:51 4s/step - acc: 0.0212 - loss: 4.6047 - top5-acc: 0.0612

<div class="k-default-codeblock">
```

```
</div>
  39/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:47 4s/step - acc: 0.0210 - loss: 4.6047 - top5-acc: 0.0611

<div class="k-default-codeblock">
```

```
</div>
  40/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:43 4s/step - acc: 0.0208 - loss: 4.6047 - top5-acc: 0.0610

<div class="k-default-codeblock">
```

```
</div>
  41/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:40 4s/step - acc: 0.0206 - loss: 4.6048 - top5-acc: 0.0608

<div class="k-default-codeblock">
```

```
</div>
  42/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:36 4s/step - acc: 0.0204 - loss: 4.6048 - top5-acc: 0.0607

<div class="k-default-codeblock">
```

```
</div>
  43/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:32 4s/step - acc: 0.0203 - loss: 4.6048 - top5-acc: 0.0606

<div class="k-default-codeblock">
```

```
</div>
  44/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:28 4s/step - acc: 0.0201 - loss: 4.6048 - top5-acc: 0.0605

<div class="k-default-codeblock">
```

```
</div>
  45/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:24 4s/step - acc: 0.0199 - loss: 4.6048 - top5-acc: 0.0604

<div class="k-default-codeblock">
```

```
</div>
  46/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:20 4s/step - acc: 0.0198 - loss: 4.6048 - top5-acc: 0.0603

<div class="k-default-codeblock">
```

```
</div>
  47/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:16 4s/step - acc: 0.0196 - loss: 4.6048 - top5-acc: 0.0602

<div class="k-default-codeblock">
```

```
</div>
  48/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:12 4s/step - acc: 0.0194 - loss: 4.6048 - top5-acc: 0.0601

<div class="k-default-codeblock">
```

```
</div>
  49/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:09 4s/step - acc: 0.0193 - loss: 4.6048 - top5-acc: 0.0600

<div class="k-default-codeblock">
```

```
</div>
  50/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:05 4s/step - acc: 0.0191 - loss: 4.6048 - top5-acc: 0.0599

<div class="k-default-codeblock">
```

```
</div>
  51/704 ━[37m━━━━━━━━━━━━━━━━━━━  42:02 4s/step - acc: 0.0190 - loss: 4.6048 - top5-acc: 0.0597

<div class="k-default-codeblock">
```

```
</div>
  52/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:57 4s/step - acc: 0.0188 - loss: 4.6048 - top5-acc: 0.0596

<div class="k-default-codeblock">
```

```
</div>
  53/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:54 4s/step - acc: 0.0187 - loss: 4.6049 - top5-acc: 0.0594

<div class="k-default-codeblock">
```

```
</div>
  54/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:50 4s/step - acc: 0.0186 - loss: 4.6049 - top5-acc: 0.0593

<div class="k-default-codeblock">
```

```
</div>
  55/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:46 4s/step - acc: 0.0184 - loss: 4.6049 - top5-acc: 0.0591

<div class="k-default-codeblock">
```

```
</div>
  56/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:42 4s/step - acc: 0.0183 - loss: 4.6049 - top5-acc: 0.0590

<div class="k-default-codeblock">
```

```
</div>
  57/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:39 4s/step - acc: 0.0182 - loss: 4.6049 - top5-acc: 0.0589

<div class="k-default-codeblock">
```

```
</div>
  58/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:35 4s/step - acc: 0.0181 - loss: 4.6049 - top5-acc: 0.0587

<div class="k-default-codeblock">
```

```
</div>
  59/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:31 4s/step - acc: 0.0180 - loss: 4.6049 - top5-acc: 0.0586

<div class="k-default-codeblock">
```

```
</div>
  60/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:26 4s/step - acc: 0.0179 - loss: 4.6049 - top5-acc: 0.0585

<div class="k-default-codeblock">
```

```
</div>
  61/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:22 4s/step - acc: 0.0178 - loss: 4.6049 - top5-acc: 0.0584

<div class="k-default-codeblock">
```

```
</div>
  62/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:19 4s/step - acc: 0.0177 - loss: 4.6049 - top5-acc: 0.0583

<div class="k-default-codeblock">
```

```
</div>
  63/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:15 4s/step - acc: 0.0176 - loss: 4.6049 - top5-acc: 0.0582

<div class="k-default-codeblock">
```

```
</div>
  64/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:11 4s/step - acc: 0.0175 - loss: 4.6049 - top5-acc: 0.0581

<div class="k-default-codeblock">
```

```
</div>
  65/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:07 4s/step - acc: 0.0175 - loss: 4.6049 - top5-acc: 0.0579

<div class="k-default-codeblock">
```

```
</div>
  66/704 ━[37m━━━━━━━━━━━━━━━━━━━  41:03 4s/step - acc: 0.0174 - loss: 4.6049 - top5-acc: 0.0579

<div class="k-default-codeblock">
```

```
</div>
  67/704 ━[37m━━━━━━━━━━━━━━━━━━━  40:59 4s/step - acc: 0.0173 - loss: 4.6049 - top5-acc: 0.0578

<div class="k-default-codeblock">
```

```
</div>
  68/704 ━[37m━━━━━━━━━━━━━━━━━━━  40:55 4s/step - acc: 0.0172 - loss: 4.6049 - top5-acc: 0.0577

<div class="k-default-codeblock">
```

```
</div>
  69/704 ━[37m━━━━━━━━━━━━━━━━━━━  40:51 4s/step - acc: 0.0171 - loss: 4.6049 - top5-acc: 0.0576

<div class="k-default-codeblock">
```

```
</div>
  70/704 ━[37m━━━━━━━━━━━━━━━━━━━  40:47 4s/step - acc: 0.0171 - loss: 4.6050 - top5-acc: 0.0575

<div class="k-default-codeblock">
```

```
</div>
  71/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:43 4s/step - acc: 0.0170 - loss: 4.6050 - top5-acc: 0.0575

<div class="k-default-codeblock">
```

```
</div>
  72/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:40 4s/step - acc: 0.0169 - loss: 4.6050 - top5-acc: 0.0574

<div class="k-default-codeblock">
```

```
</div>
  73/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:36 4s/step - acc: 0.0169 - loss: 4.6050 - top5-acc: 0.0573

<div class="k-default-codeblock">
```

```
</div>
  74/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:32 4s/step - acc: 0.0168 - loss: 4.6050 - top5-acc: 0.0573

<div class="k-default-codeblock">
```

```
</div>
  75/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:28 4s/step - acc: 0.0167 - loss: 4.6050 - top5-acc: 0.0572

<div class="k-default-codeblock">
```

```
</div>
  76/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:24 4s/step - acc: 0.0167 - loss: 4.6050 - top5-acc: 0.0571

<div class="k-default-codeblock">
```

```
</div>
  77/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:20 4s/step - acc: 0.0166 - loss: 4.6050 - top5-acc: 0.0571

<div class="k-default-codeblock">
```

```
</div>
  78/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:16 4s/step - acc: 0.0165 - loss: 4.6050 - top5-acc: 0.0570

<div class="k-default-codeblock">
```

```
</div>
  79/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:12 4s/step - acc: 0.0165 - loss: 4.6050 - top5-acc: 0.0570

<div class="k-default-codeblock">
```

```
</div>
  80/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:09 4s/step - acc: 0.0164 - loss: 4.6050 - top5-acc: 0.0569

<div class="k-default-codeblock">
```

```
</div>
  81/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:05 4s/step - acc: 0.0163 - loss: 4.6050 - top5-acc: 0.0569

<div class="k-default-codeblock">
```

```
</div>
  82/704 ━━[37m━━━━━━━━━━━━━━━━━━  40:01 4s/step - acc: 0.0163 - loss: 4.6050 - top5-acc: 0.0569

<div class="k-default-codeblock">
```

```
</div>
  83/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:57 4s/step - acc: 0.0162 - loss: 4.6050 - top5-acc: 0.0568

<div class="k-default-codeblock">
```

```
</div>
  84/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:53 4s/step - acc: 0.0162 - loss: 4.6050 - top5-acc: 0.0568

<div class="k-default-codeblock">
```

```
</div>
  85/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:50 4s/step - acc: 0.0161 - loss: 4.6050 - top5-acc: 0.0567

<div class="k-default-codeblock">
```

```
</div>
  86/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:46 4s/step - acc: 0.0160 - loss: 4.6050 - top5-acc: 0.0567

<div class="k-default-codeblock">
```

```
</div>
  87/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:42 4s/step - acc: 0.0160 - loss: 4.6050 - top5-acc: 0.0566

<div class="k-default-codeblock">
```

```
</div>
  88/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:38 4s/step - acc: 0.0159 - loss: 4.6050 - top5-acc: 0.0566

<div class="k-default-codeblock">
```

```
</div>
  89/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:34 4s/step - acc: 0.0159 - loss: 4.6050 - top5-acc: 0.0566

<div class="k-default-codeblock">
```

```
</div>
  90/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:30 4s/step - acc: 0.0158 - loss: 4.6050 - top5-acc: 0.0565

<div class="k-default-codeblock">
```

```
</div>
  91/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:26 4s/step - acc: 0.0158 - loss: 4.6050 - top5-acc: 0.0565

<div class="k-default-codeblock">
```

```
</div>
  92/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:22 4s/step - acc: 0.0157 - loss: 4.6050 - top5-acc: 0.0564

<div class="k-default-codeblock">
```

```
</div>
  93/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:18 4s/step - acc: 0.0157 - loss: 4.6050 - top5-acc: 0.0564

<div class="k-default-codeblock">
```

```
</div>
  94/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:15 4s/step - acc: 0.0156 - loss: 4.6050 - top5-acc: 0.0564

<div class="k-default-codeblock">
```

```
</div>
  95/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:11 4s/step - acc: 0.0156 - loss: 4.6050 - top5-acc: 0.0563

<div class="k-default-codeblock">
```

```
</div>
  96/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:07 4s/step - acc: 0.0155 - loss: 4.6050 - top5-acc: 0.0563

<div class="k-default-codeblock">
```

```
</div>
  97/704 ━━[37m━━━━━━━━━━━━━━━━━━  39:03 4s/step - acc: 0.0155 - loss: 4.6050 - top5-acc: 0.0562

<div class="k-default-codeblock">
```

```
</div>
  98/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:59 4s/step - acc: 0.0154 - loss: 4.6050 - top5-acc: 0.0562

<div class="k-default-codeblock">
```

```
</div>
  99/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:55 4s/step - acc: 0.0154 - loss: 4.6050 - top5-acc: 0.0562

<div class="k-default-codeblock">
```

```
</div>
 100/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:51 4s/step - acc: 0.0153 - loss: 4.6050 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 101/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:47 4s/step - acc: 0.0153 - loss: 4.6050 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 102/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:43 4s/step - acc: 0.0152 - loss: 4.6050 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 103/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:40 4s/step - acc: 0.0152 - loss: 4.6050 - top5-acc: 0.0561

<div class="k-default-codeblock">
```

```
</div>
 104/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:36 4s/step - acc: 0.0152 - loss: 4.6050 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 105/704 ━━[37m━━━━━━━━━━━━━━━━━━  38:32 4s/step - acc: 0.0151 - loss: 4.6050 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 106/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:28 4s/step - acc: 0.0151 - loss: 4.6050 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 107/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:24 4s/step - acc: 0.0151 - loss: 4.6050 - top5-acc: 0.0560

<div class="k-default-codeblock">
```

```
</div>
 108/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:20 4s/step - acc: 0.0150 - loss: 4.6050 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 109/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:17 4s/step - acc: 0.0150 - loss: 4.6050 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 110/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:13 4s/step - acc: 0.0150 - loss: 4.6050 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 111/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:09 4s/step - acc: 0.0149 - loss: 4.6050 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 112/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:05 4s/step - acc: 0.0149 - loss: 4.6050 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 113/704 ━━━[37m━━━━━━━━━━━━━━━━━  38:01 4s/step - acc: 0.0149 - loss: 4.6050 - top5-acc: 0.0559

<div class="k-default-codeblock">
```

```
</div>
 114/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:57 4s/step - acc: 0.0148 - loss: 4.6050 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 115/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:53 4s/step - acc: 0.0148 - loss: 4.6050 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 116/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:49 4s/step - acc: 0.0148 - loss: 4.6050 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 117/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:45 4s/step - acc: 0.0148 - loss: 4.6050 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 118/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:42 4s/step - acc: 0.0147 - loss: 4.6050 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 119/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:38 4s/step - acc: 0.0147 - loss: 4.6050 - top5-acc: 0.0558

<div class="k-default-codeblock">
```

```
</div>
 120/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:34 4s/step - acc: 0.0147 - loss: 4.6050 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 121/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:30 4s/step - acc: 0.0146 - loss: 4.6050 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 122/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:26 4s/step - acc: 0.0146 - loss: 4.6050 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 123/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:22 4s/step - acc: 0.0146 - loss: 4.6050 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 124/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:18 4s/step - acc: 0.0146 - loss: 4.6050 - top5-acc: 0.0557

<div class="k-default-codeblock">
```

```
</div>
 125/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:14 4s/step - acc: 0.0145 - loss: 4.6050 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 126/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:10 4s/step - acc: 0.0145 - loss: 4.6050 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 127/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:06 4s/step - acc: 0.0145 - loss: 4.6050 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 128/704 ━━━[37m━━━━━━━━━━━━━━━━━  37:02 4s/step - acc: 0.0145 - loss: 4.6050 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 129/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:58 4s/step - acc: 0.0144 - loss: 4.6050 - top5-acc: 0.0556

<div class="k-default-codeblock">
```

```
</div>
 130/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:55 4s/step - acc: 0.0144 - loss: 4.6050 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 131/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:51 4s/step - acc: 0.0144 - loss: 4.6050 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 132/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:47 4s/step - acc: 0.0144 - loss: 4.6050 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 133/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:43 4s/step - acc: 0.0143 - loss: 4.6050 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 134/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:39 4s/step - acc: 0.0143 - loss: 4.6050 - top5-acc: 0.0555

<div class="k-default-codeblock">
```

```
</div>
 135/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:35 4s/step - acc: 0.0143 - loss: 4.6050 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 136/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:32 4s/step - acc: 0.0143 - loss: 4.6050 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 137/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:28 4s/step - acc: 0.0143 - loss: 4.6050 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 138/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:24 4s/step - acc: 0.0142 - loss: 4.6050 - top5-acc: 0.0554

<div class="k-default-codeblock">
```

```
</div>
 139/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:20 4s/step - acc: 0.0142 - loss: 4.6050 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 140/704 ━━━[37m━━━━━━━━━━━━━━━━━  36:16 4s/step - acc: 0.0142 - loss: 4.6050 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 141/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:12 4s/step - acc: 0.0142 - loss: 4.6050 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 142/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:08 4s/step - acc: 0.0142 - loss: 4.6050 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 143/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:05 4s/step - acc: 0.0141 - loss: 4.6050 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 144/704 ━━━━[37m━━━━━━━━━━━━━━━━  36:01 4s/step - acc: 0.0141 - loss: 4.6050 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 145/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:57 4s/step - acc: 0.0141 - loss: 4.6050 - top5-acc: 0.0553

<div class="k-default-codeblock">
```

```
</div>
 146/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:53 4s/step - acc: 0.0141 - loss: 4.6050 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 147/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:49 4s/step - acc: 0.0141 - loss: 4.6050 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 148/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:45 4s/step - acc: 0.0140 - loss: 4.6050 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 149/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:41 4s/step - acc: 0.0140 - loss: 4.6050 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 150/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:37 4s/step - acc: 0.0140 - loss: 4.6050 - top5-acc: 0.0552

<div class="k-default-codeblock">
```

```
</div>
 151/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:33 4s/step - acc: 0.0140 - loss: 4.6050 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 152/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:30 4s/step - acc: 0.0140 - loss: 4.6050 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 153/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:26 4s/step - acc: 0.0139 - loss: 4.6050 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 154/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:22 4s/step - acc: 0.0139 - loss: 4.6050 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 155/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:18 4s/step - acc: 0.0139 - loss: 4.6050 - top5-acc: 0.0551

<div class="k-default-codeblock">
```

```
</div>
 156/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:14 4s/step - acc: 0.0139 - loss: 4.6050 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 157/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:10 4s/step - acc: 0.0139 - loss: 4.6050 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 158/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:06 4s/step - acc: 0.0139 - loss: 4.6050 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 159/704 ━━━━[37m━━━━━━━━━━━━━━━━  35:03 4s/step - acc: 0.0138 - loss: 4.6050 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 160/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:59 4s/step - acc: 0.0138 - loss: 4.6050 - top5-acc: 0.0550

<div class="k-default-codeblock">
```

```
</div>
 161/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:55 4s/step - acc: 0.0138 - loss: 4.6050 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 162/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:51 4s/step - acc: 0.0138 - loss: 4.6050 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 163/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:47 4s/step - acc: 0.0138 - loss: 4.6050 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 164/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:43 4s/step - acc: 0.0138 - loss: 4.6050 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 165/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:40 4s/step - acc: 0.0137 - loss: 4.6050 - top5-acc: 0.0549

<div class="k-default-codeblock">
```

```
</div>
 166/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:36 4s/step - acc: 0.0137 - loss: 4.6050 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 167/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:32 4s/step - acc: 0.0137 - loss: 4.6050 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 168/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:28 4s/step - acc: 0.0137 - loss: 4.6050 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 169/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:24 4s/step - acc: 0.0137 - loss: 4.6050 - top5-acc: 0.0548

<div class="k-default-codeblock">
```

```
</div>
 170/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:20 4s/step - acc: 0.0136 - loss: 4.6051 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 171/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:17 4s/step - acc: 0.0136 - loss: 4.6051 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 172/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:13 4s/step - acc: 0.0136 - loss: 4.6051 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 173/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:09 4s/step - acc: 0.0136 - loss: 4.6051 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 174/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:05 4s/step - acc: 0.0136 - loss: 4.6051 - top5-acc: 0.0547

<div class="k-default-codeblock">
```

```
</div>
 175/704 ━━━━[37m━━━━━━━━━━━━━━━━  34:01 4s/step - acc: 0.0136 - loss: 4.6051 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 176/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:57 4s/step - acc: 0.0135 - loss: 4.6051 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 177/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:53 4s/step - acc: 0.0135 - loss: 4.6051 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 178/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:50 4s/step - acc: 0.0135 - loss: 4.6051 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 179/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:46 4s/step - acc: 0.0135 - loss: 4.6051 - top5-acc: 0.0546

<div class="k-default-codeblock">
```

```
</div>
 180/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:42 4s/step - acc: 0.0135 - loss: 4.6051 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 181/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:38 4s/step - acc: 0.0135 - loss: 4.6051 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 182/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:34 4s/step - acc: 0.0134 - loss: 4.6051 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 183/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:30 4s/step - acc: 0.0134 - loss: 4.6051 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 184/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:26 4s/step - acc: 0.0134 - loss: 4.6051 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 185/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:23 4s/step - acc: 0.0134 - loss: 4.6051 - top5-acc: 0.0545

<div class="k-default-codeblock">
```

```
</div>
 186/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:19 4s/step - acc: 0.0134 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 187/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:15 4s/step - acc: 0.0134 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 188/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:11 4s/step - acc: 0.0134 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 189/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:07 4s/step - acc: 0.0133 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 190/704 ━━━━━[37m━━━━━━━━━━━━━━━  33:03 4s/step - acc: 0.0133 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 191/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:59 4s/step - acc: 0.0133 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 192/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:56 4s/step - acc: 0.0133 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 193/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:52 4s/step - acc: 0.0133 - loss: 4.6051 - top5-acc: 0.0544

<div class="k-default-codeblock">
```

```
</div>
 194/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:48 4s/step - acc: 0.0133 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 195/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:44 4s/step - acc: 0.0133 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 196/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:40 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 197/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:37 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 198/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:33 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 199/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:29 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 200/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:25 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 201/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:21 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0543

<div class="k-default-codeblock">
```

```
</div>
 202/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:17 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 203/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:13 4s/step - acc: 0.0132 - loss: 4.6051 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 204/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:10 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 205/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:06 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 206/704 ━━━━━[37m━━━━━━━━━━━━━━━  32:02 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 207/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:58 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 208/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:54 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0542

<div class="k-default-codeblock">
```

```
</div>
 209/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:50 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 210/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:47 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 211/704 ━━━━━[37m━━━━━━━━━━━━━━━  31:43 4s/step - acc: 0.0131 - loss: 4.6051 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 212/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:39 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 213/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:35 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 214/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:31 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 215/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:27 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0541

<div class="k-default-codeblock">
```

```
</div>
 216/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:23 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 217/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:19 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 218/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:16 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 219/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:12 4s/step - acc: 0.0130 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 220/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:08 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 221/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:04 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 222/704 ━━━━━━[37m━━━━━━━━━━━━━━  31:00 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 223/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:56 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0540

<div class="k-default-codeblock">
```

```
</div>
 224/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:52 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 225/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:48 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 226/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:45 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 227/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:41 4s/step - acc: 0.0129 - loss: 4.6051 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 228/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:37 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 229/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:33 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 230/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:29 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0539

<div class="k-default-codeblock">
```

```
</div>
 231/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:25 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 232/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:22 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 233/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:18 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 234/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:14 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 235/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:10 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 236/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:06 4s/step - acc: 0.0128 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 237/704 ━━━━━━[37m━━━━━━━━━━━━━━  30:02 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 238/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:58 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0538

<div class="k-default-codeblock">
```

```
</div>
 239/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:55 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 240/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:51 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 241/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:47 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 242/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:43 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 243/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:39 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 244/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:35 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 245/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:31 4s/step - acc: 0.0127 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 246/704 ━━━━━━[37m━━━━━━━━━━━━━━  29:28 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0537

<div class="k-default-codeblock">
```

```
</div>
 247/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:24 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 248/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:20 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 249/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:16 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 250/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:12 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 251/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:08 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 252/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:04 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 253/704 ━━━━━━━[37m━━━━━━━━━━━━━  29:00 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 254/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:57 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 255/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:53 4s/step - acc: 0.0126 - loss: 4.6051 - top5-acc: 0.0536

<div class="k-default-codeblock">
```

```
</div>
 256/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:49 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 257/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:45 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 258/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:41 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 259/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:37 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 260/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:34 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 261/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:30 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 262/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:26 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 263/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:22 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0535

<div class="k-default-codeblock">
```

```
</div>
 264/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:18 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 265/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:14 4s/step - acc: 0.0125 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 266/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:10 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 267/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:07 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 268/704 ━━━━━━━[37m━━━━━━━━━━━━━  28:03 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 269/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:59 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 270/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:55 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 271/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:51 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 272/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:47 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 273/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:43 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 274/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:40 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 275/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:36 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 276/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:32 4s/step - acc: 0.0124 - loss: 4.6051 - top5-acc: 0.0534

<div class="k-default-codeblock">
```

```
</div>
 277/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:28 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 278/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:24 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 279/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:20 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 280/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:16 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 281/704 ━━━━━━━[37m━━━━━━━━━━━━━  27:13 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 282/704 ━━━━━━━━[37m━━━━━━━━━━━━  27:09 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 283/704 ━━━━━━━━[37m━━━━━━━━━━━━  27:05 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 284/704 ━━━━━━━━[37m━━━━━━━━━━━━  27:01 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 285/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:57 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 286/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:53 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 287/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:49 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 288/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:46 4s/step - acc: 0.0123 - loss: 4.6051 - top5-acc: 0.0533

<div class="k-default-codeblock">
```

```
</div>
 289/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:42 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 290/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:38 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 291/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:34 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 292/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:30 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 293/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:26 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 294/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:22 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 295/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:19 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 296/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:15 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 297/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:11 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 298/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:07 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 299/704 ━━━━━━━━[37m━━━━━━━━━━━━  26:03 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 300/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:59 4s/step - acc: 0.0122 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 301/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:55 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0532

<div class="k-default-codeblock">
```

```
</div>
 302/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:52 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 303/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:48 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 304/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:44 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 305/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:40 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 306/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:36 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 307/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:32 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 308/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:29 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 309/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:25 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 310/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:21 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 311/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:17 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 312/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:13 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 313/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:09 4s/step - acc: 0.0121 - loss: 4.6051 - top5-acc: 0.0531

<div class="k-default-codeblock">
```

```
</div>
 314/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:05 4s/step - acc: 0.0120 - loss: 4.6051 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 315/704 ━━━━━━━━[37m━━━━━━━━━━━━  25:01 4s/step - acc: 0.0120 - loss: 4.6051 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 316/704 ━━━━━━━━[37m━━━━━━━━━━━━  24:58 4s/step - acc: 0.0120 - loss: 4.6051 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 317/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:54 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 318/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:50 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 319/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:46 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 320/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:42 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 321/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:38 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 322/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:34 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 323/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:31 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 324/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:27 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 325/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:23 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 326/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:19 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0530

<div class="k-default-codeblock">
```

```
</div>
 327/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:15 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 328/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:11 4s/step - acc: 0.0120 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 329/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:07 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 330/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:04 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 331/704 ━━━━━━━━━[37m━━━━━━━━━━━  24:00 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 332/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:56 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 333/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:52 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 334/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:48 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 335/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:44 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 336/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:40 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 337/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:36 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 338/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:33 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 339/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:29 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 340/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:25 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0529

<div class="k-default-codeblock">
```

```
</div>
 341/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:21 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 342/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:17 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 343/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:13 4s/step - acc: 0.0119 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 344/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:09 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 345/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:06 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 346/704 ━━━━━━━━━[37m━━━━━━━━━━━  23:02 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 347/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:58 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 348/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:54 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 349/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:50 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 350/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:46 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 351/704 ━━━━━━━━━[37m━━━━━━━━━━━  22:42 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 352/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:39 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 353/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:35 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 354/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:31 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 355/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:27 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0528

<div class="k-default-codeblock">
```

```
</div>
 356/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:23 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 357/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:19 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 358/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:15 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 359/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:11 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 360/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:08 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 361/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:04 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 362/704 ━━━━━━━━━━[37m━━━━━━━━━━  22:00 4s/step - acc: 0.0118 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 363/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:56 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 364/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:52 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 365/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:48 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 366/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:45 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 367/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:41 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 368/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:37 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 369/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:33 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 370/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:29 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0527

<div class="k-default-codeblock">
```

```
</div>
 371/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:25 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 372/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:21 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 373/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:18 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 374/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:14 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 375/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:10 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 376/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:06 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 377/704 ━━━━━━━━━━[37m━━━━━━━━━━  21:02 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 378/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:58 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 379/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:54 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 380/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:50 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 381/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:47 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 382/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:43 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 383/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:39 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 384/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:35 4s/step - acc: 0.0117 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 385/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:31 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0526

<div class="k-default-codeblock">
```

```
</div>
 386/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:27 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 387/704 ━━━━━━━━━━[37m━━━━━━━━━━  20:23 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 388/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:20 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 389/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:16 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 390/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:12 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 391/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:08 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 392/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:04 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 393/704 ━━━━━━━━━━━[37m━━━━━━━━━  20:00 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 394/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:56 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 395/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:53 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 396/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:49 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 397/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:45 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 398/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:41 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 399/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:37 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 400/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:33 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 401/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:29 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 402/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:25 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 403/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:22 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0525

<div class="k-default-codeblock">
```

```
</div>
 404/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:18 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 405/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:14 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 406/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:10 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 407/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:06 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 408/704 ━━━━━━━━━━━[37m━━━━━━━━━  19:02 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 409/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:58 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 410/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:55 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 411/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:51 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 412/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:47 4s/step - acc: 0.0116 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 413/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:43 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 414/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:39 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 415/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:35 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 416/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:31 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 417/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:28 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 418/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:24 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 419/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:20 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 420/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:16 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0524

<div class="k-default-codeblock">
```

```
</div>
 421/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:12 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 422/704 ━━━━━━━━━━━[37m━━━━━━━━━  18:08 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 423/704 ━━━━━━━━━━━━[37m━━━━━━━━  18:04 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 424/704 ━━━━━━━━━━━━[37m━━━━━━━━  18:00 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 425/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:57 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 426/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:53 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 427/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:49 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 428/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:45 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 429/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:41 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 430/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:37 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 431/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:33 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 432/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:30 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 433/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:26 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 434/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:22 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 435/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:18 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 436/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:14 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0523

<div class="k-default-codeblock">
```

```
</div>
 437/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:10 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 438/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:06 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 439/704 ━━━━━━━━━━━━[37m━━━━━━━━  17:03 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 440/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:59 4s/step - acc: 0.0115 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 441/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:55 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 442/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:51 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 443/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:47 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 444/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:43 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 445/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:39 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 446/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:35 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 447/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:32 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 448/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:28 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 449/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:24 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 450/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:20 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 451/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:16 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 452/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:12 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 453/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:08 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 454/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:05 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0522

<div class="k-default-codeblock">
```

```
</div>
 455/704 ━━━━━━━━━━━━[37m━━━━━━━━  16:01 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 456/704 ━━━━━━━━━━━━[37m━━━━━━━━  15:57 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 457/704 ━━━━━━━━━━━━[37m━━━━━━━━  15:53 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 458/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:49 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 459/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:45 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 460/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:41 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 461/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:38 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 462/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:34 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 463/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:30 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 464/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:26 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 465/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:22 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 466/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:18 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 467/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:14 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 468/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:11 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 469/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:07 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 470/704 ━━━━━━━━━━━━━[37m━━━━━━━  15:03 4s/step - acc: 0.0114 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 471/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:59 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 472/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:55 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 473/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:51 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0521

<div class="k-default-codeblock">
```

```
</div>
 474/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:47 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 475/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:44 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 476/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:40 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 477/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:36 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 478/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:32 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 479/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:28 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 480/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:24 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 481/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:20 4s/step - acc: 0.0113 - loss: 4.6052 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 482/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:17 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 483/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:13 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 484/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:09 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 485/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:05 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 486/704 ━━━━━━━━━━━━━[37m━━━━━━━  14:01 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 487/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:57 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 488/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:53 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 489/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:50 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 490/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:46 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 491/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:42 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 492/704 ━━━━━━━━━━━━━[37m━━━━━━━  13:38 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 493/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:34 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0520

<div class="k-default-codeblock">
```

```
</div>
 494/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:30 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 495/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:26 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 496/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:23 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 497/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:19 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 498/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:15 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 499/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:11 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 500/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:07 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 501/704 ━━━━━━━━━━━━━━[37m━━━━━━  13:03 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 502/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:59 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 503/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:56 4s/step - acc: 0.0113 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 504/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:52 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 505/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:48 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 506/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:44 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 507/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:40 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 508/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:36 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 509/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:32 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 510/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:28 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 511/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:25 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 512/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:21 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 513/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:17 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0519

<div class="k-default-codeblock">
```

```
</div>
 514/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:13 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 515/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:09 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 516/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:05 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 517/704 ━━━━━━━━━━━━━━[37m━━━━━━  12:01 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 518/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:58 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 519/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:54 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 520/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:50 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 521/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:46 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 522/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:42 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 523/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:38 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 524/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:34 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 525/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:31 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 526/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:27 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 527/704 ━━━━━━━━━━━━━━[37m━━━━━━  11:23 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 528/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:19 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 529/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:15 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 530/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:11 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 531/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:07 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 532/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:04 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 533/704 ━━━━━━━━━━━━━━━[37m━━━━━  11:00 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0518

<div class="k-default-codeblock">
```

```
</div>
 534/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:56 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 535/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:52 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 536/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:48 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 537/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:44 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 538/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:40 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 539/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:37 4s/step - acc: 0.0112 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 540/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:33 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 541/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:29 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 542/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:25 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 543/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:21 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 544/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:17 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 545/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:13 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 546/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:10 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 547/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:06 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 548/704 ━━━━━━━━━━━━━━━[37m━━━━━  10:02 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 549/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:58 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517 

<div class="k-default-codeblock">
```

```
</div>
 550/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:54 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 551/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:50 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 552/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:46 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 553/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:42 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 554/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:39 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0517

<div class="k-default-codeblock">
```

```
</div>
 555/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:35 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 556/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:31 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 557/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:27 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 558/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:23 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 559/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:19 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 560/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:15 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 561/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:12 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 562/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:08 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 563/704 ━━━━━━━━━━━━━━━[37m━━━━━  9:04 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 564/704 ━━━━━━━━━━━━━━━━[37m━━━━  9:00 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 565/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:56 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 566/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:52 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 567/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:48 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 568/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:45 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 569/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:41 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 570/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:37 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 571/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:33 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 572/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:29 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 573/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:25 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 574/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:21 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 575/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:18 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 576/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:14 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0516

<div class="k-default-codeblock">
```

```
</div>
 577/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:10 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 578/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:06 4s/step - acc: 0.0111 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 579/704 ━━━━━━━━━━━━━━━━[37m━━━━  8:02 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 580/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:58 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 581/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:54 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 582/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:51 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 583/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:47 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 584/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:43 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 585/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:39 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 586/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:35 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 587/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:31 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 588/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:27 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 589/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:24 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 590/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:20 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 591/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:16 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 592/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:12 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 593/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:08 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 594/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:04 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 595/704 ━━━━━━━━━━━━━━━━[37m━━━━  7:00 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 596/704 ━━━━━━━━━━━━━━━━[37m━━━━  6:56 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 597/704 ━━━━━━━━━━━━━━━━[37m━━━━  6:53 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 598/704 ━━━━━━━━━━━━━━━━[37m━━━━  6:49 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0515

<div class="k-default-codeblock">
```

```
</div>
 599/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:45 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 600/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:41 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 601/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:37 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 602/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:33 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 603/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:29 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 604/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:26 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 605/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:22 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 606/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:18 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 607/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:14 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 608/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:10 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 609/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:06 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 610/704 ━━━━━━━━━━━━━━━━━[37m━━━  6:02 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 611/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:59 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 612/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:55 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 613/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:51 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 614/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:47 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 615/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:43 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 616/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:39 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 617/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:35 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 618/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:32 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 619/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:28 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 620/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:24 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 621/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:20 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 622/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:16 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0514

<div class="k-default-codeblock">
```

```
</div>
 623/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:12 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 624/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:08 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 625/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:05 4s/step - acc: 0.0110 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 626/704 ━━━━━━━━━━━━━━━━━[37m━━━  5:01 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 627/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:57 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 628/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:53 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 629/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:49 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 630/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:45 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 631/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:41 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 632/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:38 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 633/704 ━━━━━━━━━━━━━━━━━[37m━━━  4:34 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 634/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:30 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 635/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:26 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 636/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:22 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 637/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:18 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 638/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:14 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 639/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:10 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 640/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:07 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 641/704 ━━━━━━━━━━━━━━━━━━[37m━━  4:03 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 642/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:59 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 643/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:55 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 644/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:51 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 645/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:47 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 646/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:43 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 647/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:40 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 648/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:36 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0513

<div class="k-default-codeblock">
```

```
</div>
 649/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:32 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 650/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:28 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 651/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:24 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 652/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:20 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 653/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:16 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 654/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:13 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 655/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:09 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 656/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:05 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 657/704 ━━━━━━━━━━━━━━━━━━[37m━━  3:01 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 658/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:57 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 659/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:53 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 660/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:49 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 661/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:46 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 662/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:42 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 663/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:38 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 664/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:34 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 665/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:30 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 666/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:26 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 667/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:22 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 668/704 ━━━━━━━━━━━━━━━━━━[37m━━  2:18 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 669/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:15 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 670/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:11 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 671/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:07 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 672/704 ━━━━━━━━━━━━━━━━━━━[37m━  2:03 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 673/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:59 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 674/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:55 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0512

<div class="k-default-codeblock">
```

```
</div>
 675/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:51 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 676/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:48 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 677/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:44 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 678/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:40 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 679/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:36 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 680/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:32 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 681/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:28 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 682/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:24 4s/step - acc: 0.0109 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 683/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:21 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 684/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:17 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 685/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:13 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 686/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:09 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 687/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:05 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 688/704 ━━━━━━━━━━━━━━━━━━━[37m━  1:01 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 689/704 ━━━━━━━━━━━━━━━━━━━[37m━  57s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511 

<div class="k-default-codeblock">
```

```
</div>
 690/704 ━━━━━━━━━━━━━━━━━━━[37m━  54s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 691/704 ━━━━━━━━━━━━━━━━━━━[37m━  50s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 692/704 ━━━━━━━━━━━━━━━━━━━[37m━  46s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 693/704 ━━━━━━━━━━━━━━━━━━━[37m━  42s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 694/704 ━━━━━━━━━━━━━━━━━━━[37m━  38s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 695/704 ━━━━━━━━━━━━━━━━━━━[37m━  34s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 696/704 ━━━━━━━━━━━━━━━━━━━[37m━  30s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 697/704 ━━━━━━━━━━━━━━━━━━━[37m━  27s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 698/704 ━━━━━━━━━━━━━━━━━━━[37m━  23s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 699/704 ━━━━━━━━━━━━━━━━━━━[37m━  19s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 700/704 ━━━━━━━━━━━━━━━━━━━[37m━  15s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0511

<div class="k-default-codeblock">
```

```
</div>
 701/704 ━━━━━━━━━━━━━━━━━━━[37m━  11s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0510

<div class="k-default-codeblock">
```

```
</div>
 702/704 ━━━━━━━━━━━━━━━━━━━[37m━  7s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0510 

<div class="k-default-codeblock">
```

```
</div>
 703/704 ━━━━━━━━━━━━━━━━━━━[37m━  3s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0510

<div class="k-default-codeblock">
```

```
</div>
 704/704 ━━━━━━━━━━━━━━━━━━━━ 0s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0510

<div class="k-default-codeblock">
```

```
</div>
 704/704 ━━━━━━━━━━━━━━━━━━━━ 2827s 4s/step - acc: 0.0108 - loss: 4.6053 - top5-acc: 0.0510 - val_acc: 0.0070 - val_loss: 4.6068 - val_top5-acc: 0.0438 - learning_rate: 0.0010


    
   1/313 [37m━━━━━━━━━━━━━━━━━━━━  3:42 712ms/step - acc: 0.0312 - loss: 4.6011 - top5-acc: 0.0312

<div class="k-default-codeblock">
```

```
</div>
   2/313 [37m━━━━━━━━━━━━━━━━━━━━  3:38 702ms/step - acc: 0.0234 - loss: 4.6018 - top5-acc: 0.0312

<div class="k-default-codeblock">
```

```
</div>
   3/313 [37m━━━━━━━━━━━━━━━━━━━━  3:38 705ms/step - acc: 0.0191 - loss: 4.6018 - top5-acc: 0.0312

<div class="k-default-codeblock">
```

```
</div>
   4/313 [37m━━━━━━━━━━━━━━━━━━━━  3:38 706ms/step - acc: 0.0163 - loss: 4.6021 - top5-acc: 0.0293

<div class="k-default-codeblock">
```

```
</div>
   5/313 [37m━━━━━━━━━━━━━━━━━━━━  3:36 704ms/step - acc: 0.0143 - loss: 4.6024 - top5-acc: 0.0297

<div class="k-default-codeblock">
```

```
</div>
   6/313 [37m━━━━━━━━━━━━━━━━━━━━  3:36 705ms/step - acc: 0.0128 - loss: 4.6028 - top5-acc: 0.0308

<div class="k-default-codeblock">
```

```
</div>
   7/313 [37m━━━━━━━━━━━━━━━━━━━━  3:35 703ms/step - acc: 0.0116 - loss: 4.6031 - top5-acc: 0.0322

<div class="k-default-codeblock">
```

```
</div>
   8/313 [37m━━━━━━━━━━━━━━━━━━━━  3:34 703ms/step - acc: 0.0106 - loss: 4.6033 - top5-acc: 0.0335

<div class="k-default-codeblock">
```

```
</div>
   9/313 [37m━━━━━━━━━━━━━━━━━━━━  3:33 702ms/step - acc: 0.0098 - loss: 4.6034 - top5-acc: 0.0352

<div class="k-default-codeblock">
```

```
</div>
  10/313 [37m━━━━━━━━━━━━━━━━━━━━  3:32 702ms/step - acc: 0.0092 - loss: 4.6035 - top5-acc: 0.0360

<div class="k-default-codeblock">
```

```
</div>
  11/313 [37m━━━━━━━━━━━━━━━━━━━━  3:32 702ms/step - acc: 0.0086 - loss: 4.6036 - top5-acc: 0.0366

<div class="k-default-codeblock">
```

```
</div>
  12/313 [37m━━━━━━━━━━━━━━━━━━━━  3:31 702ms/step - acc: 0.0081 - loss: 4.6037 - top5-acc: 0.0368

<div class="k-default-codeblock">
```

```
</div>
  13/313 [37m━━━━━━━━━━━━━━━━━━━━  3:30 702ms/step - acc: 0.0076 - loss: 4.6038 - top5-acc: 0.0368

<div class="k-default-codeblock">
```

```
</div>
  14/313 [37m━━━━━━━━━━━━━━━━━━━━  3:30 703ms/step - acc: 0.0073 - loss: 4.6039 - top5-acc: 0.0365

<div class="k-default-codeblock">
```

```
</div>
  15/313 [37m━━━━━━━━━━━━━━━━━━━━  3:29 703ms/step - acc: 0.0072 - loss: 4.6039 - top5-acc: 0.0369

<div class="k-default-codeblock">
```

```
</div>
  16/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:28 703ms/step - acc: 0.0071 - loss: 4.6039 - top5-acc: 0.0374

<div class="k-default-codeblock">
```

```
</div>
  17/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:28 703ms/step - acc: 0.0070 - loss: 4.6040 - top5-acc: 0.0378

<div class="k-default-codeblock">
```

```
</div>
  18/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:27 704ms/step - acc: 0.0069 - loss: 4.6040 - top5-acc: 0.0381

<div class="k-default-codeblock">
```

```
</div>
  19/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:26 704ms/step - acc: 0.0069 - loss: 4.6041 - top5-acc: 0.0384

<div class="k-default-codeblock">
```

```
</div>
  20/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:26 704ms/step - acc: 0.0069 - loss: 4.6041 - top5-acc: 0.0387

<div class="k-default-codeblock">
```

```
</div>
  21/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:25 704ms/step - acc: 0.0068 - loss: 4.6041 - top5-acc: 0.0388

<div class="k-default-codeblock">
```

```
</div>
  22/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:24 704ms/step - acc: 0.0070 - loss: 4.6042 - top5-acc: 0.0391

<div class="k-default-codeblock">
```

```
</div>
  23/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:24 704ms/step - acc: 0.0071 - loss: 4.6042 - top5-acc: 0.0392

<div class="k-default-codeblock">
```

```
</div>
  24/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:23 704ms/step - acc: 0.0072 - loss: 4.6042 - top5-acc: 0.0394

<div class="k-default-codeblock">
```

```
</div>
  25/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:22 704ms/step - acc: 0.0073 - loss: 4.6043 - top5-acc: 0.0396

<div class="k-default-codeblock">
```

```
</div>
  26/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:22 704ms/step - acc: 0.0074 - loss: 4.6043 - top5-acc: 0.0398

<div class="k-default-codeblock">
```

```
</div>
  27/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:21 704ms/step - acc: 0.0074 - loss: 4.6043 - top5-acc: 0.0399

<div class="k-default-codeblock">
```

```
</div>
  28/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:20 705ms/step - acc: 0.0075 - loss: 4.6043 - top5-acc: 0.0400

<div class="k-default-codeblock">
```

```
</div>
  29/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:20 704ms/step - acc: 0.0075 - loss: 4.6044 - top5-acc: 0.0401

<div class="k-default-codeblock">
```

```
</div>
  30/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:19 705ms/step - acc: 0.0076 - loss: 4.6044 - top5-acc: 0.0402

<div class="k-default-codeblock">
```

```
</div>
  31/313 ━[37m━━━━━━━━━━━━━━━━━━━  3:18 705ms/step - acc: 0.0076 - loss: 4.6044 - top5-acc: 0.0403

<div class="k-default-codeblock">
```

```
</div>
  32/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:18 705ms/step - acc: 0.0077 - loss: 4.6044 - top5-acc: 0.0404

<div class="k-default-codeblock">
```

```
</div>
  33/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:17 705ms/step - acc: 0.0077 - loss: 4.6045 - top5-acc: 0.0404

<div class="k-default-codeblock">
```

```
</div>
  34/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:16 705ms/step - acc: 0.0077 - loss: 4.6045 - top5-acc: 0.0405

<div class="k-default-codeblock">
```

```
</div>
  35/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:16 705ms/step - acc: 0.0077 - loss: 4.6045 - top5-acc: 0.0406

<div class="k-default-codeblock">
```

```
</div>
  36/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:15 705ms/step - acc: 0.0077 - loss: 4.6045 - top5-acc: 0.0407

<div class="k-default-codeblock">
```

```
</div>
  37/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:14 705ms/step - acc: 0.0077 - loss: 4.6045 - top5-acc: 0.0408

<div class="k-default-codeblock">
```

```
</div>
  38/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:13 705ms/step - acc: 0.0078 - loss: 4.6045 - top5-acc: 0.0410

<div class="k-default-codeblock">
```

```
</div>
  39/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:13 705ms/step - acc: 0.0078 - loss: 4.6046 - top5-acc: 0.0411

<div class="k-default-codeblock">
```

```
</div>
  40/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:12 706ms/step - acc: 0.0078 - loss: 4.6046 - top5-acc: 0.0412

<div class="k-default-codeblock">
```

```
</div>
  41/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:11 706ms/step - acc: 0.0079 - loss: 4.6046 - top5-acc: 0.0414

<div class="k-default-codeblock">
```

```
</div>
  42/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:11 705ms/step - acc: 0.0079 - loss: 4.6046 - top5-acc: 0.0415

<div class="k-default-codeblock">
```

```
</div>
  43/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:10 705ms/step - acc: 0.0080 - loss: 4.6046 - top5-acc: 0.0416

<div class="k-default-codeblock">
```

```
</div>
  44/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:09 705ms/step - acc: 0.0080 - loss: 4.6046 - top5-acc: 0.0417

<div class="k-default-codeblock">
```

```
</div>
  45/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:09 705ms/step - acc: 0.0080 - loss: 4.6046 - top5-acc: 0.0418

<div class="k-default-codeblock">
```

```
</div>
  46/313 ━━[37m━━━━━━━━━━━━━━━━━━  3:08 705ms/step - acc: 0.0080 - loss: 4.6046 - top5-acc: 0.0419

<div class="k-default-codeblock">
```

```
</div>
  47/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:07 705ms/step - acc: 0.0081 - loss: 4.6046 - top5-acc: 0.0420

<div class="k-default-codeblock">
```

```
</div>
  48/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:06 705ms/step - acc: 0.0081 - loss: 4.6047 - top5-acc: 0.0421

<div class="k-default-codeblock">
```

```
</div>
  49/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:06 706ms/step - acc: 0.0081 - loss: 4.6047 - top5-acc: 0.0422

<div class="k-default-codeblock">
```

```
</div>
  50/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:05 706ms/step - acc: 0.0082 - loss: 4.6047 - top5-acc: 0.0422

<div class="k-default-codeblock">
```

```
</div>
  51/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:04 706ms/step - acc: 0.0082 - loss: 4.6047 - top5-acc: 0.0423

<div class="k-default-codeblock">
```

```
</div>
  52/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:04 706ms/step - acc: 0.0082 - loss: 4.6047 - top5-acc: 0.0423

<div class="k-default-codeblock">
```

```
</div>
  53/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:03 705ms/step - acc: 0.0082 - loss: 4.6047 - top5-acc: 0.0424

<div class="k-default-codeblock">
```

```
</div>
  54/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:02 706ms/step - acc: 0.0082 - loss: 4.6047 - top5-acc: 0.0424

<div class="k-default-codeblock">
```

```
</div>
  55/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:02 706ms/step - acc: 0.0082 - loss: 4.6047 - top5-acc: 0.0424

<div class="k-default-codeblock">
```

```
</div>
  56/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:01 706ms/step - acc: 0.0082 - loss: 4.6048 - top5-acc: 0.0425

<div class="k-default-codeblock">
```

```
</div>
  57/313 ━━━[37m━━━━━━━━━━━━━━━━━  3:00 705ms/step - acc: 0.0082 - loss: 4.6048 - top5-acc: 0.0426

<div class="k-default-codeblock">
```

```
</div>
  58/313 ━━━[37m━━━━━━━━━━━━━━━━━  2:59 705ms/step - acc: 0.0082 - loss: 4.6048 - top5-acc: 0.0426

<div class="k-default-codeblock">
```

```
</div>
  59/313 ━━━[37m━━━━━━━━━━━━━━━━━  2:59 705ms/step - acc: 0.0083 - loss: 4.6048 - top5-acc: 0.0427

<div class="k-default-codeblock">
```

```
</div>
  60/313 ━━━[37m━━━━━━━━━━━━━━━━━  2:58 706ms/step - acc: 0.0083 - loss: 4.6048 - top5-acc: 0.0427

<div class="k-default-codeblock">
```

```
</div>
  61/313 ━━━[37m━━━━━━━━━━━━━━━━━  2:57 705ms/step - acc: 0.0083 - loss: 4.6048 - top5-acc: 0.0427

<div class="k-default-codeblock">
```

```
</div>
  62/313 ━━━[37m━━━━━━━━━━━━━━━━━  2:57 705ms/step - acc: 0.0082 - loss: 4.6048 - top5-acc: 0.0428

<div class="k-default-codeblock">
```

```
</div>
  63/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:56 705ms/step - acc: 0.0082 - loss: 4.6048 - top5-acc: 0.0428

<div class="k-default-codeblock">
```

```
</div>
  64/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:55 706ms/step - acc: 0.0082 - loss: 4.6048 - top5-acc: 0.0429

<div class="k-default-codeblock">
```

```
</div>
  65/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:54 706ms/step - acc: 0.0082 - loss: 4.6048 - top5-acc: 0.0429

<div class="k-default-codeblock">
```

```
</div>
  66/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:54 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0430

<div class="k-default-codeblock">
```

```
</div>
  67/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:53 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0431

<div class="k-default-codeblock">
```

```
</div>
  68/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:52 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0431

<div class="k-default-codeblock">
```

```
</div>
  69/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:52 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0432

<div class="k-default-codeblock">
```

```
</div>
  70/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:51 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0432

<div class="k-default-codeblock">
```

```
</div>
  71/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:50 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0433

<div class="k-default-codeblock">
```

```
</div>
  72/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:50 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0434

<div class="k-default-codeblock">
```

```
</div>
  73/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:49 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0435

<div class="k-default-codeblock">
```

```
</div>
  74/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:48 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0435

<div class="k-default-codeblock">
```

```
</div>
  75/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:47 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0436

<div class="k-default-codeblock">
```

```
</div>
  76/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:47 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0437

<div class="k-default-codeblock">
```

```
</div>
  77/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:46 706ms/step - acc: 0.0082 - loss: 4.6049 - top5-acc: 0.0437

<div class="k-default-codeblock">
```

```
</div>
  78/313 ━━━━[37m━━━━━━━━━━━━━━━━  2:45 706ms/step - acc: 0.0083 - loss: 4.6049 - top5-acc: 0.0438

<div class="k-default-codeblock">
```

```
</div>
  79/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:45 706ms/step - acc: 0.0083 - loss: 4.6049 - top5-acc: 0.0439

<div class="k-default-codeblock">
```

```
</div>
  80/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:44 706ms/step - acc: 0.0083 - loss: 4.6049 - top5-acc: 0.0439

<div class="k-default-codeblock">
```

```
</div>
  81/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:43 706ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0440

<div class="k-default-codeblock">
```

```
</div>
  82/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:42 706ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0440

<div class="k-default-codeblock">
```

```
</div>
  83/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:42 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0441

<div class="k-default-codeblock">
```

```
</div>
  84/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:41 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0442

<div class="k-default-codeblock">
```

```
</div>
  85/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:40 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0442

<div class="k-default-codeblock">
```

```
</div>
  86/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:40 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0443

<div class="k-default-codeblock">
```

```
</div>
  87/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:39 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0443

<div class="k-default-codeblock">
```

```
</div>
  88/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:38 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0444

<div class="k-default-codeblock">
```

```
</div>
  89/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:37 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0444

<div class="k-default-codeblock">
```

```
</div>
  90/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:37 705ms/step - acc: 0.0082 - loss: 4.6050 - top5-acc: 0.0445

<div class="k-default-codeblock">
```

```
</div>
  91/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:36 705ms/step - acc: 0.0082 - loss: 4.6050 - top5-acc: 0.0446

<div class="k-default-codeblock">
```

```
</div>
  92/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:35 705ms/step - acc: 0.0082 - loss: 4.6050 - top5-acc: 0.0446

<div class="k-default-codeblock">
```

```
</div>
  93/313 ━━━━━[37m━━━━━━━━━━━━━━━  2:35 705ms/step - acc: 0.0082 - loss: 4.6050 - top5-acc: 0.0447

<div class="k-default-codeblock">
```

```
</div>
  94/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:34 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0448

<div class="k-default-codeblock">
```

```
</div>
  95/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:33 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0448

<div class="k-default-codeblock">
```

```
</div>
  96/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:33 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0449

<div class="k-default-codeblock">
```

```
</div>
  97/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:32 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0450

<div class="k-default-codeblock">
```

```
</div>
  98/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:31 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0450

<div class="k-default-codeblock">
```

```
</div>
  99/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:30 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0451

<div class="k-default-codeblock">
```

```
</div>
 100/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:30 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0452

<div class="k-default-codeblock">
```

```
</div>
 101/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:29 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0452

<div class="k-default-codeblock">
```

```
</div>
 102/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:28 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0453

<div class="k-default-codeblock">
```

```
</div>
 103/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:28 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0453

<div class="k-default-codeblock">
```

```
</div>
 104/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:27 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0454

<div class="k-default-codeblock">
```

```
</div>
 105/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:26 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0454

<div class="k-default-codeblock">
```

```
</div>
 106/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:25 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0455

<div class="k-default-codeblock">
```

```
</div>
 107/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:25 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0455

<div class="k-default-codeblock">
```

```
</div>
 108/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:24 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0456

<div class="k-default-codeblock">
```

```
</div>
 109/313 ━━━━━━[37m━━━━━━━━━━━━━━  2:23 705ms/step - acc: 0.0083 - loss: 4.6050 - top5-acc: 0.0456

<div class="k-default-codeblock">
```

```
</div>
 110/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:23 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0457

<div class="k-default-codeblock">
```

```
</div>
 111/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:22 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0457

<div class="k-default-codeblock">
```

```
</div>
 112/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:21 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0458

<div class="k-default-codeblock">
```

```
</div>
 113/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:20 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0458

<div class="k-default-codeblock">
```

```
</div>
 114/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:20 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0459

<div class="k-default-codeblock">
```

```
</div>
 115/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:19 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0459

<div class="k-default-codeblock">
```

```
</div>
 116/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:18 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0459

<div class="k-default-codeblock">
```

```
</div>
 117/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:18 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0460

<div class="k-default-codeblock">
```

```
</div>
 118/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:17 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0460

<div class="k-default-codeblock">
```

```
</div>
 119/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:16 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0460

<div class="k-default-codeblock">
```

```
</div>
 120/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:16 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0461

<div class="k-default-codeblock">
```

```
</div>
 121/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:15 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0461

<div class="k-default-codeblock">
```

```
</div>
 122/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:14 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0462

<div class="k-default-codeblock">
```

```
</div>
 123/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:13 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0462

<div class="k-default-codeblock">
```

```
</div>
 124/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:13 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0462

<div class="k-default-codeblock">
```

```
</div>
 125/313 ━━━━━━━[37m━━━━━━━━━━━━━  2:12 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0463

<div class="k-default-codeblock">
```

```
</div>
 126/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:11 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0463

<div class="k-default-codeblock">
```

```
</div>
 127/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:11 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0464

<div class="k-default-codeblock">
```

```
</div>
 128/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:10 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0464

<div class="k-default-codeblock">
```

```
</div>
 129/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:09 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0464

<div class="k-default-codeblock">
```

```
</div>
 130/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:08 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0465

<div class="k-default-codeblock">
```

```
</div>
 131/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:08 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0465

<div class="k-default-codeblock">
```

```
</div>
 132/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:07 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0466

<div class="k-default-codeblock">
```

```
</div>
 133/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:06 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0466

<div class="k-default-codeblock">
```

```
</div>
 134/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:06 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0466

<div class="k-default-codeblock">
```

```
</div>
 135/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:05 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0467

<div class="k-default-codeblock">
```

```
</div>
 136/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:04 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0467

<div class="k-default-codeblock">
```

```
</div>
 137/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:04 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0467

<div class="k-default-codeblock">
```

```
</div>
 138/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:03 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0468

<div class="k-default-codeblock">
```

```
</div>
 139/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:02 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0468

<div class="k-default-codeblock">
```

```
</div>
 140/313 ━━━━━━━━[37m━━━━━━━━━━━━  2:01 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0468

<div class="k-default-codeblock">
```

```
</div>
 141/313 ━━━━━━━━━[37m━━━━━━━━━━━  2:01 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0469

<div class="k-default-codeblock">
```

```
</div>
 142/313 ━━━━━━━━━[37m━━━━━━━━━━━  2:00 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0469

<div class="k-default-codeblock">
```

```
</div>
 143/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:59 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0469

<div class="k-default-codeblock">
```

```
</div>
 144/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:59 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0469

<div class="k-default-codeblock">
```

```
</div>
 145/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:58 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0470

<div class="k-default-codeblock">
```

```
</div>
 146/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:57 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0470

<div class="k-default-codeblock">
```

```
</div>
 147/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:57 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0470

<div class="k-default-codeblock">
```

```
</div>
 148/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:56 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0471

<div class="k-default-codeblock">
```

```
</div>
 149/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:55 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0471

<div class="k-default-codeblock">
```

```
</div>
 150/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:54 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0471

<div class="k-default-codeblock">
```

```
</div>
 151/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:54 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0472

<div class="k-default-codeblock">
```

```
</div>
 152/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:53 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0472

<div class="k-default-codeblock">
```

```
</div>
 153/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:52 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0472

<div class="k-default-codeblock">
```

```
</div>
 154/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:52 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0473

<div class="k-default-codeblock">
```

```
</div>
 155/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:51 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0473

<div class="k-default-codeblock">
```

```
</div>
 156/313 ━━━━━━━━━[37m━━━━━━━━━━━  1:50 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0473

<div class="k-default-codeblock">
```

```
</div>
 157/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:49 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0473

<div class="k-default-codeblock">
```

```
</div>
 158/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:49 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0474

<div class="k-default-codeblock">
```

```
</div>
 159/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:48 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0474

<div class="k-default-codeblock">
```

```
</div>
 160/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:47 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0474

<div class="k-default-codeblock">
```

```
</div>
 161/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:47 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0474

<div class="k-default-codeblock">
```

```
</div>
 162/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:46 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0475

<div class="k-default-codeblock">
```

```
</div>
 163/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:45 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0475

<div class="k-default-codeblock">
```

```
</div>
 164/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:45 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0475

<div class="k-default-codeblock">
```

```
</div>
 165/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:44 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0475

<div class="k-default-codeblock">
```

```
</div>
 166/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:43 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0476

<div class="k-default-codeblock">
```

```
</div>
 167/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:42 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0476

<div class="k-default-codeblock">
```

```
</div>
 168/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:42 705ms/step - acc: 0.0083 - loss: 4.6051 - top5-acc: 0.0476

<div class="k-default-codeblock">
```

```
</div>
 169/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:41 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0476

<div class="k-default-codeblock">
```

```
</div>
 170/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:40 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0477

<div class="k-default-codeblock">
```

```
</div>
 171/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:40 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0477

<div class="k-default-codeblock">
```

```
</div>
 172/313 ━━━━━━━━━━[37m━━━━━━━━━━  1:39 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0477

<div class="k-default-codeblock">
```

```
</div>
 173/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:38 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0477

<div class="k-default-codeblock">
```

```
</div>
 174/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:37 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0477

<div class="k-default-codeblock">
```

```
</div>
 175/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:37 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0478

<div class="k-default-codeblock">
```

```
</div>
 176/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:36 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0478

<div class="k-default-codeblock">
```

```
</div>
 177/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:35 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0478

<div class="k-default-codeblock">
```

```
</div>
 178/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:35 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0478

<div class="k-default-codeblock">
```

```
</div>
 179/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:34 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0479

<div class="k-default-codeblock">
```

```
</div>
 180/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:33 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0479

<div class="k-default-codeblock">
```

```
</div>
 181/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:33 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0479

<div class="k-default-codeblock">
```

```
</div>
 182/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:32 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0479

<div class="k-default-codeblock">
```

```
</div>
 183/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:31 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0479

<div class="k-default-codeblock">
```

```
</div>
 184/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:30 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0480

<div class="k-default-codeblock">
```

```
</div>
 185/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:30 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0480

<div class="k-default-codeblock">
```

```
</div>
 186/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:29 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0480

<div class="k-default-codeblock">
```

```
</div>
 187/313 ━━━━━━━━━━━[37m━━━━━━━━━  1:28 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0480

<div class="k-default-codeblock">
```

```
</div>
 188/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:28 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0480

<div class="k-default-codeblock">
```

```
</div>
 189/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:27 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0480

<div class="k-default-codeblock">
```

```
</div>
 190/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:26 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0481

<div class="k-default-codeblock">
```

```
</div>
 191/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:26 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0481

<div class="k-default-codeblock">
```

```
</div>
 192/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:25 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0481

<div class="k-default-codeblock">
```

```
</div>
 193/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:24 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0481

<div class="k-default-codeblock">
```

```
</div>
 194/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:23 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0481

<div class="k-default-codeblock">
```

```
</div>
 195/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:23 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0481

<div class="k-default-codeblock">
```

```
</div>
 196/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:22 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0482

<div class="k-default-codeblock">
```

```
</div>
 197/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:21 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0482

<div class="k-default-codeblock">
```

```
</div>
 198/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:21 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0482

<div class="k-default-codeblock">
```

```
</div>
 199/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:20 705ms/step - acc: 0.0084 - loss: 4.6051 - top5-acc: 0.0482

<div class="k-default-codeblock">
```

```
</div>
 200/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:19 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0482

<div class="k-default-codeblock">
```

```
</div>
 201/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:18 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0482

<div class="k-default-codeblock">
```

```
</div>
 202/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:18 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 203/313 ━━━━━━━━━━━━[37m━━━━━━━━  1:17 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 204/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:16 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 205/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:16 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 206/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:15 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 207/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:14 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 208/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:14 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 209/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:13 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0483

<div class="k-default-codeblock">
```

```
</div>
 210/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:12 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 211/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:11 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 212/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:11 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 213/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:10 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 214/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:09 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 215/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:09 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 216/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:08 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 217/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:07 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0484

<div class="k-default-codeblock">
```

```
</div>
 218/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:06 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0485

<div class="k-default-codeblock">
```

```
</div>
 219/313 ━━━━━━━━━━━━━[37m━━━━━━━  1:06 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0485

<div class="k-default-codeblock">
```

```
</div>
 220/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:05 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0485

<div class="k-default-codeblock">
```

```
</div>
 221/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:04 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0485

<div class="k-default-codeblock">
```

```
</div>
 222/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:04 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0485

<div class="k-default-codeblock">
```

```
</div>
 223/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:03 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0485

<div class="k-default-codeblock">
```

```
</div>
 224/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:02 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0485

<div class="k-default-codeblock">
```

```
</div>
 225/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:02 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0486

<div class="k-default-codeblock">
```

```
</div>
 226/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:01 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0486

<div class="k-default-codeblock">
```

```
</div>
 227/313 ━━━━━━━━━━━━━━[37m━━━━━━  1:00 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0486

<div class="k-default-codeblock">
```

```
</div>
 228/313 ━━━━━━━━━━━━━━[37m━━━━━━  59s 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0486 

<div class="k-default-codeblock">
```

```
</div>
 229/313 ━━━━━━━━━━━━━━[37m━━━━━━  59s 705ms/step - acc: 0.0085 - loss: 4.6051 - top5-acc: 0.0486

<div class="k-default-codeblock">
```

```
</div>
 230/313 ━━━━━━━━━━━━━━[37m━━━━━━  58s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0486

<div class="k-default-codeblock">
```

```
</div>
 231/313 ━━━━━━━━━━━━━━[37m━━━━━━  57s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0486

<div class="k-default-codeblock">
```

```
</div>
 232/313 ━━━━━━━━━━━━━━[37m━━━━━━  57s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0486

<div class="k-default-codeblock">
```

```
</div>
 233/313 ━━━━━━━━━━━━━━[37m━━━━━━  56s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 234/313 ━━━━━━━━━━━━━━[37m━━━━━━  55s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 235/313 ━━━━━━━━━━━━━━━[37m━━━━━  54s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 236/313 ━━━━━━━━━━━━━━━[37m━━━━━  54s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 237/313 ━━━━━━━━━━━━━━━[37m━━━━━  53s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 238/313 ━━━━━━━━━━━━━━━[37m━━━━━  52s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 239/313 ━━━━━━━━━━━━━━━[37m━━━━━  52s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 240/313 ━━━━━━━━━━━━━━━[37m━━━━━  51s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 241/313 ━━━━━━━━━━━━━━━[37m━━━━━  50s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 242/313 ━━━━━━━━━━━━━━━[37m━━━━━  50s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 243/313 ━━━━━━━━━━━━━━━[37m━━━━━  49s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0487

<div class="k-default-codeblock">
```

```
</div>
 244/313 ━━━━━━━━━━━━━━━[37m━━━━━  48s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 245/313 ━━━━━━━━━━━━━━━[37m━━━━━  47s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 246/313 ━━━━━━━━━━━━━━━[37m━━━━━  47s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 247/313 ━━━━━━━━━━━━━━━[37m━━━━━  46s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 248/313 ━━━━━━━━━━━━━━━[37m━━━━━  45s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 249/313 ━━━━━━━━━━━━━━━[37m━━━━━  45s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 250/313 ━━━━━━━━━━━━━━━[37m━━━━━  44s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 251/313 ━━━━━━━━━━━━━━━━[37m━━━━  43s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 252/313 ━━━━━━━━━━━━━━━━[37m━━━━  43s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 253/313 ━━━━━━━━━━━━━━━━[37m━━━━  42s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 254/313 ━━━━━━━━━━━━━━━━[37m━━━━  41s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 255/313 ━━━━━━━━━━━━━━━━[37m━━━━  40s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0488

<div class="k-default-codeblock">
```

```
</div>
 256/313 ━━━━━━━━━━━━━━━━[37m━━━━  40s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 257/313 ━━━━━━━━━━━━━━━━[37m━━━━  39s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 258/313 ━━━━━━━━━━━━━━━━[37m━━━━  38s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 259/313 ━━━━━━━━━━━━━━━━[37m━━━━  38s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 260/313 ━━━━━━━━━━━━━━━━[37m━━━━  37s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 261/313 ━━━━━━━━━━━━━━━━[37m━━━━  36s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 262/313 ━━━━━━━━━━━━━━━━[37m━━━━  35s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 263/313 ━━━━━━━━━━━━━━━━[37m━━━━  35s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 264/313 ━━━━━━━━━━━━━━━━[37m━━━━  34s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 265/313 ━━━━━━━━━━━━━━━━[37m━━━━  33s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 266/313 ━━━━━━━━━━━━━━━━[37m━━━━  33s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0489

<div class="k-default-codeblock">
```

```
</div>
 267/313 ━━━━━━━━━━━━━━━━━[37m━━━  32s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 268/313 ━━━━━━━━━━━━━━━━━[37m━━━  31s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 269/313 ━━━━━━━━━━━━━━━━━[37m━━━  31s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 270/313 ━━━━━━━━━━━━━━━━━[37m━━━  30s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 271/313 ━━━━━━━━━━━━━━━━━[37m━━━  29s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 272/313 ━━━━━━━━━━━━━━━━━[37m━━━  28s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 273/313 ━━━━━━━━━━━━━━━━━[37m━━━  28s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 274/313 ━━━━━━━━━━━━━━━━━[37m━━━  27s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 275/313 ━━━━━━━━━━━━━━━━━[37m━━━  26s 705ms/step - acc: 0.0086 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 276/313 ━━━━━━━━━━━━━━━━━[37m━━━  26s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 277/313 ━━━━━━━━━━━━━━━━━[37m━━━  25s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0490

<div class="k-default-codeblock">
```

```
</div>
 278/313 ━━━━━━━━━━━━━━━━━[37m━━━  24s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 279/313 ━━━━━━━━━━━━━━━━━[37m━━━  23s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 280/313 ━━━━━━━━━━━━━━━━━[37m━━━  23s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 281/313 ━━━━━━━━━━━━━━━━━[37m━━━  22s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 282/313 ━━━━━━━━━━━━━━━━━━[37m━━  21s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 283/313 ━━━━━━━━━━━━━━━━━━[37m━━  21s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 284/313 ━━━━━━━━━━━━━━━━━━[37m━━  20s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 285/313 ━━━━━━━━━━━━━━━━━━[37m━━  19s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 286/313 ━━━━━━━━━━━━━━━━━━[37m━━  19s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 287/313 ━━━━━━━━━━━━━━━━━━[37m━━  18s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 288/313 ━━━━━━━━━━━━━━━━━━[37m━━  17s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0491

<div class="k-default-codeblock">
```

```
</div>
 289/313 ━━━━━━━━━━━━━━━━━━[37m━━  16s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 290/313 ━━━━━━━━━━━━━━━━━━[37m━━  16s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 291/313 ━━━━━━━━━━━━━━━━━━[37m━━  15s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 292/313 ━━━━━━━━━━━━━━━━━━[37m━━  14s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 293/313 ━━━━━━━━━━━━━━━━━━[37m━━  14s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 294/313 ━━━━━━━━━━━━━━━━━━[37m━━  13s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 295/313 ━━━━━━━━━━━━━━━━━━[37m━━  12s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 296/313 ━━━━━━━━━━━━━━━━━━[37m━━  11s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 297/313 ━━━━━━━━━━━━━━━━━━[37m━━  11s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0492

<div class="k-default-codeblock">
```

```
</div>
 298/313 ━━━━━━━━━━━━━━━━━━━[37m━  10s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 299/313 ━━━━━━━━━━━━━━━━━━━[37m━  9s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493 

<div class="k-default-codeblock">
```

```
</div>
 300/313 ━━━━━━━━━━━━━━━━━━━[37m━  9s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 301/313 ━━━━━━━━━━━━━━━━━━━[37m━  8s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 302/313 ━━━━━━━━━━━━━━━━━━━[37m━  7s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 303/313 ━━━━━━━━━━━━━━━━━━━[37m━  7s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 304/313 ━━━━━━━━━━━━━━━━━━━[37m━  6s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 305/313 ━━━━━━━━━━━━━━━━━━━[37m━  5s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 306/313 ━━━━━━━━━━━━━━━━━━━[37m━  4s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 307/313 ━━━━━━━━━━━━━━━━━━━[37m━  4s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 308/313 ━━━━━━━━━━━━━━━━━━━[37m━  3s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 309/313 ━━━━━━━━━━━━━━━━━━━[37m━  2s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0493

<div class="k-default-codeblock">
```

```
</div>
 310/313 ━━━━━━━━━━━━━━━━━━━[37m━  2s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0494

<div class="k-default-codeblock">
```

```
</div>
 311/313 ━━━━━━━━━━━━━━━━━━━[37m━  1s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0494

<div class="k-default-codeblock">
```

```
</div>
 312/313 ━━━━━━━━━━━━━━━━━━━[37m━  0s 705ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0494

<div class="k-default-codeblock">
```

```
</div>
 313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 704ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0494

<div class="k-default-codeblock">
```

```
</div>
 313/313 ━━━━━━━━━━━━━━━━━━━━ 220s 704ms/step - acc: 0.0087 - loss: 4.6051 - top5-acc: 0.0494


<div class="k-default-codeblock">
```
Test accuracy: 0.91%
Test top 5 accuracy: 5.2%

```
</div>
After 40 epochs, the Perceiver model achieves around 53% accuracy and 81% top-5 accuracy on the test data.

As mentioned in the ablations of the [Perceiver](https://arxiv.org/abs/2103.03206) paper,
you can obtain better results by increasing the latent array size,
increasing the (projection) dimensions of the latent array and data array elements,
increasing the number of blocks in the Transformer module, and increasing the number of iterations of applying
the cross-attention and the latent Transformer modules. You may also try to increase the size the input images
and use different patch sizes.

The Perceiver benefits from inceasing the model size. However, larger models needs bigger accelerators
to fit in and train efficiently. This is why in the Perceiver paper they used 32 TPU cores to run the experiments.

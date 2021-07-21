"""
Title: Involutional Neural Networks
Author: [Aritra Roy Gosthipaty](https://twitter.com/ariG23498)
Date created: 2021/07/21
Last modified: 2021/07/21
Description: Deep dive into spatially specific and channel agnostic
    **Involution** kernels.
"""

"""
# Introduction

This example demonstrates a minimal implementation of 
[Involution: Inverting the Inherence of Convolution for Visual 
Recognition](https://arxiv.org/abs/2103.06255) by Li et. al. We will 
build the Involution Layer as a `tf.keras.layers.Layer` and try 
building an image classification model on top of it. The idea behind 
this layer is to invert the inherent properties of Convolution. Where 
convolution is spatial-agnostic and channel-specific, involution is 
spatial-specific and channel-agnostic.
"""

"""
# Setup
"""

# set seed for reproducibility
from tensorflow.random import set_seed
set_seed(42)
from tensorflow import multiply
from tensorflow import identity
from tensorflow import reduce_sum
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.data import Dataset
from tensorflow.random import normal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.image import extract_patches
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

"""
# Convolution

Convolution remains the building mainstay of deep neural networks. To 
understand **Involution** it becomes necessary to talk about the 
**Convolution** operation.

![Imgur](https://i.imgur.com/MSKLsm5.png)

Consider an input tensor $X \in \mathbb{R}^{H \times W \times 
C_{in}}$. We take a collection of $C_{out}$ convolution kernels each 
of shape $(K \times K \times C_{in})$. With the multiply-add 
operation between the input tensor and the kernels we obtain an
output tensor $Y \in \mathbb{R}^{H \times W \times C_{out}}$.

In the diagram above we have replaced $C_{out}$ with $3$. This makes 
the output tensor of shape $H \times W \times 3$. One can notice 
that the convoltuion kernel does not depend on the spatial position 
of the input tensor which makes is **spatially agnostic**, on the
other hand each channel in the output tensor is based on a specific 
convolution filter which makes is **channel specific**.
"""

"""
# Involution

In the paper the authors have tried inverting the properties of 
Convolution and named of operation Involution. The idea was to have 
kernels that are **spatially specific** and **channel agnostic**.

This brings to a problem, with a fixed number of involution kernels 
we will not be able to process variable resolution input tensors. 
To solve this, we would need to condition the generation of the 
kernel based on each pixel of the input tensor.

$$
\phi{(X_{i,j})}=W_{1}\sigma{(W_{0}X_{i,j})}
$$

Where 
- $\phi$ is the mapping between $\mathbb{R}^{C}$ to 
    $\mathbb{R}^{K \times K \times G}$.
- $W_{0} \in \mathbb{R}^{\frac{C}{r} \times C}$ reduces the tensor 
    with a reduction size of $r$.
- $W_{1} \in \mathbb{R}^{K \times K \times G \times \frac{C}{r}}$ 
    expands the tensor.
- $\sigma$ is the combination of batch normalization and relu.

![Imgur](https://i.imgur.com/jtrGGQg.png)
"""


class Involution(Layer):
    def __init__(
        self, channel, group_number, kernel_size, stride,
            reduction_ratio, name):
        super().__init__(name=name)

        # capping the lower bound of reduction_ratio
        assert (
            reduction_ratio <= channel
        ), "reduction ratio must be less than or equal to channel size"

        # capping the higher bound of group_number
        assert (
            group_number < channel
        ), "group number must be smaller than channel size"

        assert (
            channel % group_number == 0
        ), "channel size must be a multiple of group number"

        # initialize the parameters
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        # define layer o
        # this layer is important is stride is greater than 1
        self.o = (
            AveragePooling2D(pool_size=self.stride, 
                strides=self.stride, padding="same")
            if self.stride > 1
            else identity
        )

        # define the kernel generation layer
        self.kernel_gen = Sequential(
            [
                Conv2D(filters=self.channel // self.reduction_ratio,
                    kernel_size=1),
                BatchNormalization(),
                ReLU(),
                Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )

    def call(self, x):
        # get the shape of the input
        (_, H, W, C) = x.shape

        # scale the height and width with respect to the strides
        H = H // self.stride
        W = W // self.stride

        # generate the kernel wrt the input tensor
        # B, H, W, K*K*G
        kernel_inp = self.o(x)
        kernel = self.kernel_gen(kernel_inp)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = Reshape(
            target_shape=(
                H,
                W,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )(kernel)

        # extract input input patches
        # B, H, W, K*K*C
        input_patches = extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # reshape the pathches
        # B, H, W, K*K, C//G, G
        input_patches = Reshape(
            target_shape=(
                H,
                W,
                self.kernel_size * self.kernel_size,
                C // self.group_number,
                self.group_number,
            )
        )(input_patches)

        # multiply-add operation
        # B, H, W, K*K, C//G, G
        out = multiply(kernel, input_patches)
        # B, H, W, C//G, G
        out = reduce_sum(out, axis=3)

        # reshape the output
        # B, H, W, C
        out = Reshape(target_shape=(H, W, C))(out)

        # return output and the kernel
        return out, kernel


"""
## Testing the Involution layer
"""

# define the input tensor
input_tensor = normal((32, 256, 256, 3))

# with stride 1
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=1,
    reduction_ratio=1, name="inv_1"
)(input_tensor)
print(f"with stride 1 ouput shape: {output_tensor.shape}")

# with stride 2
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=2,
    reduction_ratio=1, name="inv_2"
)(input_tensor)
print(f"with stride 2 ouput shape: {output_tensor.shape}")

# with channel 16 and reduction ratio 2
output_tensor, _ = Involution(
    channel=16, group_number=1, kernel_size=5, stride=1,
    reduction_ratio=2, name="inv_3"
)(input_tensor)
print(
    "with channel 16 and reduction ratio 2 ouput shape: {}"
    .format(output_tensor.shape)
)

"""
# Image Classification

In this section, we will build an image classifier model. There will 
be two models one with convolutions and the other with involutions.

The image classification model is heavily inspired by [Convolutional 
Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn)
tutorial from Google.
"""

"""
## Get the CIFAR10 Dataset
"""

# load the CIFAR10 dataset
print("[INFO] loading the CIFAR10 dataset...")
(train_images, train_labels), (test_images, test_labels) = (cifar10
    .load_data())

# normalize pixel values to be between 0 and 1
(train_images, test_images) = (train_images / 255.0, 
    test_images / 255.0)

# batch the dataset
train_ds = (
    Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(256)
        .batch(256)
)
test_ds = (Dataset.from_tensor_slices((test_images, test_labels))
    .batch(256))

"""
## Visualise the data
"""

class_names = [
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

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # the CIFAR labels happen to be arrays, which is why you need
    # the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

"""
## Convolutional Neural Network
"""

# build the conv model
print("building the convolution model...")
conv_model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same"),
        ReLU(name="relu1"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding="same"),
        ReLU(name="relu2"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding="same"),
        ReLU(name="relu3"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10),
    ]
)

# compile the mode with the necessary loss function and optimizer
conv_model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# train the model
print("conv model training...")
conv_hist = conv_model.fit(train_ds, epochs=20,
    validation_data=test_ds)

"""
## Involutional Neural Network
"""

# build the inv model
print("building the involution model...")

inputs = Input((32, 32, 3))
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1,
    reduction_ratio=2, name="inv_1"
)(inputs)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1,
    reduction_ratio=2, name="inv_2"
)(x)
x = ReLU()(x)
x = MaxPooling2D((2, 2))(x)
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1,
    reduction_ratio=2, name="inv_3"
)(x)
x = ReLU()(x)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
x = Dense(10)(x)

inv_model = Model(inputs=[inputs], outputs=[x], name="inv_model")

# compile the mode with the necessary loss function and optimizer
inv_model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# train the model
print("inv model training...")
inv_hist = inv_model.fit(train_ds, epochs=20, validation_data=test_ds)

"""
# Comparisons

In this section, we will be looking at both the models and compare
a few pointers.
"""

"""
## Parameters

One can see that with a similar architecture the parameters in a 
CNN is much more than that of an INN (Involutional Neural Network). 
We are destined to learn a lot less from a similar architecture.
"""

conv_model.summary()

inv_model.summary()

"""
## Loss and Accuracy Plots

Here, the loss and the accuracy plots demonstrate that INNs are slow learners (with lower
parameters).
"""

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title("Convolution Loss")
plt.plot(conv_hist.history["loss"], label="loss")
plt.plot(conv_hist.history["val_loss"], label="val_loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Involution Loss")
plt.plot(inv_hist.history["loss"], label="loss")
plt.plot(inv_hist.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title("Convolution Accuracy")
plt.plot(conv_hist.history["accuracy"], label="accuracy")
plt.plot(conv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.title("Involution Accuracy")
plt.plot(inv_hist.history["accuracy"], label="accuracy")
plt.plot(inv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()

"""
# Visualizing Involution Kernels

In the paper the authors have visualised the kernels. They also say 
that **self-attention** is a complex and specific form of 
**involution**. With the visualisation of the kernels we get a heat 
map of the input tensor which can be losely translated to an 
attention map of the input. The learned heat map indeed looks quite 
promising to be called attention maps.
"""

layer_names = ["inv_1", "inv_2", "inv_3"]
outputs = [inv_model.get_layer(name).output for name in layer_names]
vis_model = Model(inv_model.input, outputs)

fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(10, 30))

for ax, test_image in zip(axes, test_images[:10]):
    (inv1_out, inv2_out, inv3_out) = vis_model.predict(
        test_image[None, ...]
    )

    _, inv1_kernel = inv1_out
    _, inv2_kernel = inv2_out
    _, inv3_kernel = inv3_out

    inv1_kernel = reduce_sum(inv1_kernel, axis=[-1, -2, -3])
    inv2_kernel = reduce_sum(inv2_kernel, axis=[-1, -2, -3])
    inv3_kernel = reduce_sum(inv3_kernel, axis=[-1, -2, -3])

    ax[0].imshow(array_to_img(test_image))
    ax[0].set_title("Input Image")

    ax[1].imshow(array_to_img(inv1_kernel[0, ..., None]))
    ax[1].set_title("Involution Kernel 1")

    ax[2].imshow(array_to_img(inv2_kernel[0, ..., None]))
    ax[2].set_title("Involution Kernel 2")

    ax[3].imshow(array_to_img(inv3_kernel[0, ..., None]))
    ax[3].set_title("Involution Kernel 3")

"""
# Conclusion and Thoughts

In the example the main focus was to build an `Involution` layer 
which can be used off the shelf. The comparisons are based on a 
specific task, feel free to use the layer for different tasks and 
report your comparisons. 

According to me the key take away of the layer is its uncanny 
relationship with that of self-attention. The intuition of spatial 
specific and channel spefic makes sense in a lot of tasks and this 
layer should be taken forward.

Moving forward one can:
- Experiment with the various hyperparameters of the involution layer
- Build different models with the involution layer
- Try building a different $\phi$ which maps from $\mathbb{R}^{C}$ 
    to $\mathbb{R}^{K \times K \times G}$
"""

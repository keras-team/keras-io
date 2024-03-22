"""
Title: Neural Eikonal Solver
Author: [Serafim Grubas](https://github.com/sgrubas)
Date created: 2023/05/22
Last modified: 2023/05/22
Description: Physics-Informed Neural Networks for solving the eikonal equation.

"""

"""
## Introduction

This example implements the [Neural Eikonal Solver](https://github.com/sgrubas/NES) (NES)
introduced by [Grubas et. al. (2023)](https://doi.org/10.1016/j.jcp.2022.111789) for
solving the eikonal equation using physics-informed neural network (PINN) and
demonstrates its usage on a synthetic model.

The idea of PINN is to incorporate PDE (partial differential equation) into the loss
function of a neural network to mimic the PDE solution ([Raissi et. al.
2019](https://doi.org/10.1016/j.jcp.2018.10.045)).

The [eikonal equation](https://en.wikipedia.org/wiki/Eikonal_equation) is a first-order
nonlinear PDE, and NES is a neural-network architecture providing robust approximation to
the eikonal. We consider the following two-point modification of the eikonal equation:

`|| grad_r T(x_s, x_r) || = 1 / v(x_r)`

where `x_s` is spatial coordinate of source, `x_r` is receiver, `|| - ||` is Euclidean
norm, `grad_r` is gradient operator w.r.t. `x_r`, `v(x_r)` is velocity at point `x_r`,
`T(x_s, x_r)` is solution to the equation. The eikonal solution `T(x_s, x_r) >= 0`
describes the fastest travel time from `x_s` to `x_r`, and, as a consequence, it
describes the shortest path trajectory. The travel time at source point `T(x_s, x_s) = 0`
is the boundary condition.

For implementing NES, we will make neural-network function `T(x_s, x_r)` which will
minimize the loss:

`Loss = | || grad_r T(x_s, x_r) || - 1 / v(x_r) | --> 0`

### Application of eikonal

The eikonal is used in many fields such as computer vision, shortest path problems, image
segmentation, and modelling of wave propagation. Specifically, this equation plays an
important role in seismology. It approximates the propagation of seismic waves in the
Earth and can be used to locate earthquake hypocenters.

### Why PINNs?
Finite methods have been developed for decades and achieved high accuracy and convergence
speed. However, they still have some limitations and in complex settings their
application may be complicated. Unlike finite methods, PINNs are mesh free, i.e. the same
neural network can be used for any geometry and distribution of collocation points where
we seek for solution. Another important advantage, it can provide dramatic compression of
the solution in massive computations. The latter will be shown in the end of the example.
"""

"""
## Imports
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

"""
## Implementation of Neural Eikonal Solver

Here, we describe the acrhitecture of the NES, for all the details about each
hyperparameter and other tricks, please see [Grubas et. al.
(2023)](https://doi.org/10.1016/j.jcp.2022.111789).

The following architecture will be implemented:
![img](https://github.com/sgrubas/NES/blob/main/NES/data/NES-TP.png?raw=true)
"""


def build_model(input_scale, velocity_min, velocity_max, num_dim=2):
    # Hyperparameters
    num_layers = 5  # number of hidden layers
    num_units = 50  # number of units on each layer
    kernel_activation = lambda z: tf.math.exp(-(z**2))  # hidden activation
    output_activation = "sigmoid"  # output activation
    kernel_initializer = "he_normal"  # initialization of layers weights

    # Input 1: Source coordinates (start point)
    source_coords_list = [
        layers.Input(shape=(1,), name=f"source_coord_{i}") for i in range(num_dim)
    ]
    source_coords = layers.Concatenate(name="source_coords", axis=-1)(
        source_coords_list
    )

    # Input 2: Receiver coordinates (end point)
    receiver_coords_list = [
        layers.Input(shape=(1,), name=f"receiver_coord_{i}") for i in range(num_dim)
    ]
    receiver_coords = layers.Concatenate(name="receiver_coords", axis=-1)(
        receiver_coords_list
    )

    # Input 3: velocity model at Receiver coordinates
    velocity_at_receiver = layers.Input(shape=(1,), name="velocity_at_receiver")

    # All inputs for solution Traveltime(source, receiver)
    inputs_list = source_coords_list + receiver_coords_list

    # Scaling of inputs, to avoid unit dependence
    source_receiver = layers.Concatenate(name="source_receiver_coords", axis=-1)(
        [source_coords, receiver_coords]
    )
    source_receiver_scaled = layers.Rescaling(
        1 / input_scale, name="source_receiver_scaled"
    )(source_receiver)

    # Trainable body consisting of Dense layers only
    hidden_state = layers.Dense(
        num_units, activation=kernel_activation, kernel_initializer=kernel_initializer
    )(source_receiver_scaled)
    for i in range(1, num_layers):
        hidden_state = layers.Dense(
            num_units,
            activation=kernel_activation,
            kernel_initializer=kernel_initializer,
        )(hidden_state)

    output_state = layers.Dense(
        1,  # traveltime is scalar function -> one output
        activation=output_activation,
        kernel_initializer=kernel_initializer,
    )(hidden_state)

    # Reciprocity principal of traveltimes states:
    # Traveltime(source, receiver) = Traveltime(receiver, source)
    # To make it strictly valid for out network, we apply sum of permutations

    traveltime_initial_model = keras.models.Model(
        inputs=inputs_list, outputs=output_state, name="Traveltime_No_Reciprocity"
    )

    # Permuted arguments 1: Source --> Receiver
    source_receiver_list = source_coords_list + receiver_coords_list
    # Permuted arguments 2: Receiver --> Source
    receiver_source_list = receiver_coords_list + source_coords_list

    traveltime_source_receiver = traveltime_initial_model(source_receiver_list)
    traveltime_receiver_source = traveltime_initial_model(receiver_source_list)

    # Average of input permutations will make T(xs,xr) = T(xr,xs)
    traveltime = layers.Lambda(lambda x: 0.5 * (x[0] + x[1]), name="Reciprocity")(
        [traveltime_source_receiver, traveltime_receiver_source]
    )

    # The velocity constraint for the solution by minimum and maximum velocities
    traveltime = layers.Lambda(
        lambda x: (1 / velocity_min - 1 / velocity_max) * x + 1 / velocity_max,
        name="velocity_scaling",
    )(traveltime)

    # The boundary condition T(source, source) = 0 is introduced by factorization
    source_receiver_difference = layers.Subtract(name="source_receiver_difference")(
        [receiver_coords, source_coords]
    )

    distance = layers.Lambda(
        lambda x: tf.norm(x, axis=-1, keepdims=True), name="distance_factor"
    )(source_receiver_difference)

    traveltime = layers.Multiply(name="Traveltime")([traveltime, distance])

    # Emulation of the eikonal equation

    # Calculation of derivatives
    traveltime_gradient_list = layers.Lambda(
        lambda x: tf.gradients(x[0], x[1], unconnected_gradients="zero"),
        name="gradients_receiver",
    )([traveltime, receiver_coords_list])

    traveltime_gradient = layers.Concatenate(axis=-1)(traveltime_gradient_list)
    traveltime_gradient_norm = layers.Lambda(
        lambda x: tf.norm(x, axis=-1, keepdims=True), name="gradient_norm"
    )(
        traveltime_gradient
    )  # |grad T|

    # The eikonal equation as a loss
    # We use the Hamiltonian form of eikonal: `(v**2 * |grad T|**2 - 1) / 2 = 0`
    eikonal = layers.Lambda(
        lambda x: ((x[0] * x[1]) ** 2 - 1.0) / 2, name="Eikonal_Equation"
    )([traveltime_gradient_norm, velocity_at_receiver])

    # Traveltime model used for the prediction (the solution itself)
    traveltime_model = keras.models.Model(
        inputs=inputs_list, outputs=traveltime, name="Model_Traveltime"
    )

    # Eikonal equation model used for the training (for solving the eikonal)
    eikonal_model = keras.models.Model(
        inputs=inputs_list + [velocity_at_receiver],
        outputs=eikonal,
        name="Model_Eikonal_Equation",
    )

    return traveltime_model, eikonal_model


"""
## Creating synthetic model

For the demonstration we will be using a layered velocity model
"""


def RandomGaussians(velocity_min, velocity_max, num_gaussians, coord_limits):
    np.random.seed(411)

    velocity_diff = velocity_max - velocity_min

    vels = np.random.uniform(-velocity_diff, velocity_diff, size=num_gaussians)
    coord_limits = np.array(coord_limits)
    coord_min, coord_max = coord_limits.min(), coord_limits.max()
    sigma_limit = coord_max - coord_min
    sigmas = np.random.uniform(
        sigma_limit * 0.05, sigma_limit * 0.25, size=num_gaussians
    )
    num_dim = coord_limits.shape[-1]
    centers = np.random.uniform(coord_min, coord_max, size=(num_gaussians, num_dim))

    def _func(input_coords):
        shape = input_coords.shape[:-1]
        velocity = np.zeros(shape)
        dim_reshape = (None,) * (input_coords.ndim - 1)
        for center_i, sigma_i, velocity_i in zip(centers, sigmas, vels):
            gaussian = -((input_coords - center_i[dim_reshape]) ** 2) / 2 / sigma_i**2
            velocity += velocity_i * np.exp(gaussian).prod(axis=-1)
        velocity -= velocity.min()
        velocity = velocity / velocity.max() * velocity_diff + velocity_min
        return velocity

    return _func


x_num, y_num = 151, 151
x_min, x_max = 0, 1
y_min, y_max = 0, 1
receiver_x = np.linspace(x_min, x_max, x_num)
receiver_y = np.linspace(y_min, y_max, y_num)

receiver_x_mesh, receiver_y_mesh = np.meshgrid(receiver_x, receiver_y, indexing="ij")
receiver_xy_mesh = np.stack([receiver_x_mesh, receiver_y_mesh], axis=-1)

input_scale = np.max(np.abs(receiver_xy_mesh))

velocity_min, velocity_max = 1.0, 3.5
velocity_func = RandomGaussians(
    velocity_min, velocity_max, num_gaussians=21, coord_limits=[x_min, x_max]
)

velocity_model = velocity_func(receiver_xy_mesh)

"""
This is how our velocity model looks like. At each spatial point, we have the speed we
can travel with in `km / s`
"""

fig_vmap = plt.figure(figsize=(6, 5))
ax_vmap = fig_vmap.add_subplot(1, 1, 1)
velocity_image = ax_vmap.pcolormesh(receiver_x_mesh, receiver_y_mesh, velocity_model)
fig_vmap.colorbar(velocity_image, ax=ax_vmap, label="Velocity (km/s)")
plt.show()

"""
## Applying Neural Eikonal Solver to the synthetic model
"""

"""
### Training
"""

traveltime_model, eikonal_model = build_model(
    input_scale, velocity_min, velocity_max, num_dim=receiver_xy_mesh.shape[-1]
)

optimizer = tf.optimizers.Adam(learning_rate=2.5e-3)

# We use `MAE` loss for the eikonal to be robust to outliers.
loss = "mae"

eikonal_model.compile(optimizer=optimizer, loss=loss)

# For the training (solving), we generate `N` random `Source -> Receiver` pairs
# inside the given velocity model

num_train = 40000  # Number of random collocation points for training
inputs_train = {
    "source_coord_0": np.random.uniform(x_min, x_max, num_train),
    "source_coord_1": np.random.uniform(y_min, y_max, num_train),
    "receiver_coord_0": np.random.uniform(x_min, x_max, num_train),
    "receiver_coord_1": np.random.uniform(y_min, y_max, num_train),
}

train_receiver_coords = np.stack(
    (inputs_train["receiver_coord_0"], inputs_train["receiver_coord_1"]), axis=-1
)
inputs_train["velocity_at_receiver"] = velocity_func(train_receiver_coords)
eikonal_output_train = np.zeros(num_train)

"""
The training (solving) process
"""

history = eikonal_model.fit(
    x=inputs_train,
    y=eikonal_output_train,
    batch_size=round(num_train / 4),
    epochs=3000,
    verbose=0,
    callbacks=[TqdmCallback(verbose=0, miniters=25, ascii=" >=")],
)

"""
Note, the loss is not the same as solution accuracy because the loss shows only how well
we solve the equation, but it does not really show how accurate the solution is. Below,
we will validate the NES solution with an accurate finite-difference solver
"""

"""
### Prediction
"""

sparsity = 4
source_x = receiver_x[::sparsity]
source_y = receiver_y[::sparsity]

pairs = np.meshgrid(source_x, source_y, receiver_x, receiver_y, indexing="ij")
pairs = np.stack(pairs, axis=-1)

shape = pairs.shape[:-1]

inputs_test = {
    "source_coord_0": pairs[..., 0].ravel(),  # Source - X
    "source_coord_1": pairs[..., 1].ravel(),  # Source - Y
    "receiver_coord_0": pairs[..., 2].ravel(),  # Receiver - X
    "receiver_coord_1": pairs[..., 3].ravel(),
}  # Receiver - Y

traveltime_pred = traveltime_model.predict(inputs_test, batch_size=250000).reshape(
    shape
)

"""
This is how the traveltime field can be shown via contour lines. Each line indicate the
wave front with same traveltime value. The shortest path trajectories can be traced by
drawing a curve from the source point which is always transverse to the fronts (white
contour lines).
"""

source_ind = (5, 5)  # To plot the solution, we choose a sample Source point

fig_vmap = plt.figure(figsize=(6, 5))
ax_vmap = fig_vmap.add_subplot(1, 1, 1)
velocity_image = ax_vmap.pcolormesh(receiver_x_mesh, receiver_y_mesh, velocity_model)
source_point = ax_vmap.scatter(
    source_x[source_ind[0]], source_y[source_ind[1]], s=100, c="blue", marker="*"
)
fig_vmap.colorbar(velocity_image, ax=ax_vmap, label="Velocity (km / s)")
tctr_pred = ax_vmap.contour(
    receiver_x_mesh,
    receiver_y_mesh,
    traveltime_pred[source_ind],
    levels=12,
    colors=["w"],
    linestyles="--",
    linewidths=2,
)
ax_vmap.clabel(tctr_pred, tctr_pred.levels, fmt="{0:.2f} s".format)
plt.show()

"""
## Validation of solutions with a finite-difference method

Here we compare the acquired NES solution with the finite-difference approach called
[factored Fast Marching Method](https://doi.org/10.1016/j.jcp.2016.08.012) of second
order (one of the most accurate for this type of equation). We use the implementation
[`eikonalfm`](https://github.com/kevinganster/eikonalfm).
"""

"""shell
!pip install eikonalfm==0.9.5
"""

from eikonalfm import factored_fast_marching as ffm
from eikonalfm import distance

"""
Obtaining the solution with the finite-difference method
"""

traveltime_ref = np.empty(pairs.shape[:-1])
receiver_dx = receiver_x[1] - receiver_x[0]
receiver_dy = receiver_y[1] - receiver_y[0]
for i, source_ind_x in enumerate(tqdm(range(0, x_num, sparsity), ascii=" >=")):
    for j, source_ind_y in enumerate(range(0, y_num, sparsity)):
        traveltime_ref[i, j] = ffm(
            velocity_model, (source_ind_x, source_ind_y), [receiver_dx, receiver_dy], 2
        )
        traveltime_ref[i, j] *= distance(
            velocity_model.shape,
            [receiver_dx, receiver_dy],
            (source_ind_x, source_ind_y),
            indexing="ij",
        )

"""
Comparison of the NES solutions (white dashed lines) and the finite-difference method
(black solid lines)
"""

source_inds = [(5, 5), (5, 30), (35, 15), (15, 15)]

fig_vmap, axes = plt.subplots(nrows=2, ncols=2, figsize=(6 * 2, 5 * 2))

for si, ax in zip(source_inds, axes.ravel()):
    velocity_image = ax.pcolormesh(receiver_x_mesh, receiver_y_mesh, velocity_model)
    fig_vmap.colorbar(velocity_image, ax=ax, label="Velocity (km / s)")
    source_point = ax.scatter(
        source_x[si[0]], source_y[si[1]], s=100, c="blue", marker="*", label="Source"
    )
    tctr_ref = ax.contour(
        receiver_x_mesh,
        receiver_y_mesh,
        traveltime_ref[si],
        levels=12,
        colors=["k"],
        linestyles="-",
        linewidths=3,
    )

    tctr_pred = ax.contour(
        receiver_x_mesh,
        receiver_y_mesh,
        traveltime_pred[si],
        levels=tctr_ref.levels,
        colors=["w"],
        linestyles="--",
        linewidths=2,
    )

    ax.clabel(tctr_pred, tctr_ref.levels, fmt="{0:.2f} s".format)

axes[0, 0].legend()
plt.show()

"""
From the figure above, one can see that the NES solution almost perfectly match the
finite-difference solution (fronts coincide).
"""

"""
## Retrieving complex solution
"""

"""
Before we considered point solutions, but we can also retrieve the solution for complex
shapes of the source, for example it can be a vertical line. For that we take solution
for each point source on the line, and then calculate minimum (fastest) among them.
"""

source_line = 19
traveltime_line_ref = traveltime_ref[source_line].min(axis=0)
traveltime_line_pred = traveltime_pred[source_line].min(axis=0)

fig_vmap = plt.figure(figsize=(8, 6.6))
ax_vmap = fig_vmap.add_subplot(1, 1, 1)
velocity_image = ax_vmap.pcolormesh(receiver_x_mesh, receiver_y_mesh, velocity_model)
source_points = ax_vmap.scatter(
    np.full_like(source_y, source_x[source_line]), source_y, s=50, c="blue", marker="*"
)
fig_vmap.colorbar(velocity_image, ax=ax_vmap, label="Velocity (km / s)")
tctr_ref = ax_vmap.contour(
    receiver_x_mesh,
    receiver_y_mesh,
    traveltime_line_ref,
    levels=9,
    colors=["k"],
    linestyles="-",
    linewidths=3,
)
tctr_pred = ax_vmap.contour(
    receiver_x_mesh,
    receiver_y_mesh,
    traveltime_line_pred,
    levels=tctr_ref.levels,
    colors=["w"],
    linestyles="--",
    linewidths=2,
)
ax_vmap.clabel(tctr_pred, tctr_ref.levels, fmt="{0:.2f} s".format)

plt.show()

"""
## Accuracy and compression ability
"""

mae = abs(traveltime_ref - traveltime_pred).mean()
rmae = mae / traveltime_ref.mean() * 100
print(f"RMAE = {rmae:.3f} %")

grid_params = traveltime_ref.size
nn_params = traveltime_model.count_params()
print(
    f"Neural network compressed the initial solution with {grid_params} \
params to {nn_params} which is ~{grid_params / nn_params:.0f} times"
)

"""
## Conclusions

We showed how to implement the NES architecture using Tensorflow. We utilized
`tf.gradients` inside the neural-network model to compute derivatives for the eikonal
equation. The obtained NES solution showed very good accuracy compared to the finite
difference method with a reasonable training time, and moreover showed the promising
compression abilities. For more examples with NES and more detailed information, please
see [our github page](https://github.com/sgrubas/NES) and
[paper](https://doi.org/10.1016/j.jcp.2022.111789).
"""

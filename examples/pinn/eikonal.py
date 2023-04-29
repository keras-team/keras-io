"""
Title: Neural Eikonal Solver
Author: [Serafim Grubas](https://github.com/sgrubas)
Date created: 2023/04/29
Last modified: 2023/04/29
Description: Physics-Informed Neural Networks for solving the eikonal equation.

"""

"""
## Introduction

This example implements the [Neural Eikonal Solver](https://github.com/sgrubas/NES) (NES)
introduced by [Grubas et. al. (2023)](https://doi.org/10.1016/j.jcp.2022.111789) for
solving the eikonal equation using neural networks and demonstrates its usage on a
synthetic model.

The eikonal equation is a first-order nonlinear PDE (partial differential equation). This
example is aimed at implementing the specific neural-network architecture NES for
accurate and robust solving the eikonal equation. Similar networks for solving PDEs like
eikonal are called Physics-Informed Neural Networks (PINNs) which were introduced by
[Raissi et. al. (2019)](https://doi.org/10.1016/j.jcp.2018.10.045). The main idea of
PINNs is to mimic PDEs we want to solve so that the trained network approximates the
solution to a PDE.

Here, we will be constructing a neural-network architecture for solving the following
two-point modification of the [eikonal
equation](https://en.wikipedia.org/wiki/Eikonal_equation):

$$\Vert \nabla_r \tau(\textbf{x}_s, \textbf{x}_r) \Vert = \frac{1}{v(\textbf{x}_r)}$$

where $\textbf{x}_s=(x_s, y_s)$ is spatial coordinate of source or start point (for
simplicity we consider 2D space), $\textbf{x}_r=(x_r, y_r)$ is receiver or end point,
$\Vert \cdot \Vert$ is Euclidean norm, $\nabla_r=(\partial_{x_r}, \partial_{y_r})$ is
gradient operator w.r.t. $\textbf{x}_r$, $v(\textbf{x}_r)$ is velocity at point
$\textbf{x}_r$, $\tau(\textbf{x}_s, \textbf{x}_r)$ is solution to the equation (eikonal).
The eikonal $\tau(\textbf{x}_s, \textbf{x}_r)$ describes the fastest travel time from a
point $\textbf{x}_s$ to a point $\textbf{x}_r$, and, as a consequence, it describes the
shortest path trajectory. An important part of the equation is its boundary condition:

$$ \tau(\textbf{x}_s, \textbf{x}_s)=0 $$

which states that the travel time at a start point is zero. Another important constraint
is $\tau \geq 0$.

### Application of eikonal

The eikonal is used in many fields such as computer vision, shortest path problems, image
segmentation, and modelling of wave propagation. Specifically, this equation plays an
important role in seismology. It approximates the propagation of seismic waves in the
Earth and can be used to locate earthquake hypocenters.

### Why PINNs?
Finite methods (finite differences, finite elements, finite volumes) have been developed
for decades and achieved high accuracy and speed. However, they still have some
limitations and in complex settings their application may be complicated. For example, it
is tricky to apply finite-difference methods when the mesh is irregular in space. In this
case, an advantage of PINNs is that they are mesh-free, the same neural network can be
used for any geometry and distribution of collocation points where we seek for solution.
Another important advantage, it can provide dramatic compression of the solution in
massive computations. The latter will be shown in the end of the example.
"""

"""
## Imports
"""

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import Multiply, Subtract, Lambda
from tensorflow.keras.models import Model
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


def build_model(input_scale, vmin, vmax, dim=2):
    # Hyperparameters
    n_layers = 5  # number of hidden layers
    n_units = 50  # number of units on each layer
    kernel_act = lambda z: tf.math.exp(-(z**2))  # hidden activation function
    output_act = "sigmoid"  # output activation
    kernel_initializer = "he_normal"  # initialization of layers weights

    # Input 1: Source coordinates (start point)
    xs_list = [Input(shape=(1,), name=f"xs{i}") for i in range(dim)]
    xs = Concatenate(name="xs", axis=-1)(xs_list)

    # Input 2: Receiver coordinates (end point)
    xr_list = [Input(shape=(1,), name=f"xr{i}") for i in range(dim)]
    xr = Concatenate(name="xr", axis=-1)(xr_list)

    # Input 3: velocity model at Receiver coordinates
    vr = Input(shape=(1,), name="vr")

    # All inputs for solution T(xs, xr)
    inputs_list = xs_list + xr_list

    # Scaling of inputs, to avoid unit dependence
    XsXr = Concatenate(name="x", axis=-1)([xs, xr])
    XsXr_scaled = Rescaling(1 / input_scale, name="X_scaling")(XsXr)

    # Trainable body consisting of Dense layers only
    x = Dense(n_units, activation=kernel_act, kernel_initializer=kernel_initializer)(
        XsXr_scaled
    )
    for i in range(1, n_layers):
        x = Dense(
            n_units, activation=kernel_act, kernel_initializer=kernel_initializer
        )(x)
    T = Dense(
        1,  # traveltime is scalar function, therefore one output
        activation=output_act,
        kernel_initializer=kernel_initializer,
    )(x)

    # Reciprocity principal of traveltimes states T(xs, xr) = T(xr, xs), i.e.
    # traveltime from `xs` to `xr` is equal to traveltime from `xs` to `xr`
    # To make it stricly valid for out network, we apply sum of permutations

    t_model = Model(inputs=inputs_list, outputs=T, name="Model_No_Reciprocity")
    xsr = xs_list + xr_list  # Permuted arguments 1: Source --> Receiver
    xrs = xr_list + xs_list  # Permuted arguments 2: Receiver --> Source
    tsr = t_model(xsr)  # Traveltime (Source --> Receiver)
    trs = t_model(xrs)  # Traveltime (Receiver --> Source)

    # Average of input permutations will make T(xs,xr) = T(xr,xs)
    T = Lambda(lambda x: 0.5 * (x[0] + x[1]), name="Reciprocity")([tsr, trs])

    # Since, the analytical solution is always bounded by the slowest
    # and the fastest possible solutions, we can apply this constraint
    # using max and min values of our velocity model
    T = Lambda(lambda z: (1 / vmin - 1 / vmax) * z + 1 / vmax, name="V_factor")(T)

    # To properly account for the boundary condition T(xs, xs) = 0,
    # and accurately emulate the singularity at the source point `xs`,
    # we factorize the solution by multiplying by distance function (L2 norm)
    xr_xs = Subtract(name="xr_xs_difference")([xr, xs])
    D = Lambda(lambda z: tf.norm(z, axis=-1, keepdims=True), name="R_factor")(xr_xs)
    T = Multiply(name="Traveltime")([T, D])

    # Finally, we emulate the eikonal equation using `tf.gradients`
    # Derivative of `T(xs, xr)` w.r.t. Receiver coordinates `xr`
    dTr_list = Lambda(
        lambda x: tf.gradients(x[0], x[1], unconnected_gradients="zero"),
        name="gradients_xr",
    )([T, xr_list])
    dTr = Concatenate(axis=-1)(dTr_list)
    dTr_norm = Lambda(lambda x: tf.norm(x, axis=-1, keepdims=True), name="T_norm")(
        dTr
    )  # |grad T|

    # Last important ingredient is to use Hamiltonian form of the eikonal
    # Instead of `|grad T| = 1 / v`, we use `(v**2 * |grad T|**2 - 1) / 2 = 0`
    # It is needed to mitigate the caustic singularities
    Eikonal = Lambda(lambda x: ((x[0] * x[1]) ** 2 - 1.0) / 2, name="Eikonal_Equation")(
        [dTr_norm, vr]
    )

    # Traveltime model used for the prediction (the solution itself)
    T_model = Model(inputs=inputs_list, outputs=T, name="Model_Traveltime")

    # Eikonal equation model used for the training (for solving the eikonal)
    Eikonal_model = Model(
        inputs=inputs_list + [vr], outputs=Eikonal, name="Model_Eikonal_Equation"
    )

    return T_model, Eikonal_model


"""
## Creating synthetic model

For the demonstration we will be using a layered velocity model
"""


def RandomGaussians(vmin, vmax, n_gaussians, coord_limits):
    np.random.seed(411)

    dv = vmax - vmin
    vels = np.random.uniform(-dv, dv, size=n_gaussians)
    coord_limits = np.array(coord_limits)
    xmin, xmax = coord_limits.min(), coord_limits.max()
    sigma_limit = xmax - xmin
    sigmas = np.random.uniform(sigma_limit * 0.05, sigma_limit * 0.25, size=n_gaussians)
    dim = coord_limits.shape[-1]
    centers = np.random.uniform(xmin, xmax, size=(n_gaussians, dim))

    def _func(X):
        shape = X.shape[:-1]
        V = np.zeros(shape)
        dim_reshape = (None,) * (X.ndim - 1)
        for ci, si, vi in zip(centers, sigmas, vels):
            V += vi * np.exp(-((X - ci[dim_reshape]) ** 2) / 2 / si**2).prod(axis=-1)
        V -= V.min()
        V = V / V.max() * dv + vmin
        return V

    return _func


Nx, Ny = 151, 151
xmin, xmax = 0, 1
ymin, ymax = 0, 1
xr = np.linspace(xmin, xmax, Nx)
yr = np.linspace(ymin, ymax, Ny)

Xr, Yr = np.meshgrid(xr, yr, indexing="ij")
XYr = np.stack([Xr, Yr], axis=-1)
xscale = np.max(abs(XYr))

vmin, vmax = 1.0, 4.0
velocity = RandomGaussians(vmin, vmax, n_gaussians=20, coord_limits=[xmin, xmax])
V = velocity(XYr)

"""
This is how our velocity model looks like. At each spatial point, we have the speed we
can travel with in `km / s`
"""

fig_vmap = plt.figure(figsize=(6, 5))
ax_vmap = fig_vmap.add_subplot(1, 1, 1)
vmap = ax_vmap.pcolormesh(Xr, Yr, V)
fig_vmap.colorbar(vmap, ax=ax_vmap, label="Velocity")
plt.show()

"""
## Applying Neural Eikonal Solver to the synthetic model
"""

"""
### Training
"""

Traveltime, Eikonal = build_model(xscale, vmin, vmax, dim=XYr.shape[-1])

optimizer = tf.optimizers.Adam(learning_rate=3e-3)

# We use `MAE` loss for the eikonal because it is robust to outliers.
# In the eikonal, caustic singularities are very common issue
# and they produce outliers in the loss
loss = "mae"

Eikonal.compile(optimizer=optimizer, loss=loss)

# For the training (solving), we generate `N` random `Source -> Receiver` pairs
# inside the given velocity model

N_train = 40000  # Number of random collocation points for training (solving)
inputs_train = {
    "xs0": np.random.uniform(xmin, xmax, N_train),  # Source - X
    "xs1": np.random.uniform(ymin, ymax, N_train),  # Source - Y
    "xr0": np.random.uniform(xmin, xmax, N_train),  # Receiver - X
    "xr1": np.random.uniform(ymin, ymax, N_train),
}  # Receiver - Y
Xr_train = np.stack((inputs_train["xr0"], inputs_train["xr1"]), axis=-1)
inputs_train["vr"] = velocity(Xr_train)
Eikonal_train = np.zeros(N_train)

"""
The training (solving) process
"""

h = Eikonal.fit(
    x=inputs_train,
    y=Eikonal_train,
    batch_size=round(N_train / 4),
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
xs = xr[::sparsity]
ys = yr[::sparsity]

Coords = np.meshgrid(xs, ys, xr, yr, indexing="ij")
P = np.stack(Coords, axis=-1)

shape = P.shape[:-1]

inputs_test = {
    "xs0": P[..., 0].ravel(),  # Source - X
    "xs1": P[..., 1].ravel(),  # Source - Y
    "xr0": P[..., 2].ravel(),  # Receiver - X
    "xr1": P[..., 3].ravel(),
}  # Receiver - Y

T_pred = Traveltime.predict(inputs_test, batch_size=250000).reshape(shape)

"""
This is how the traveltime field can be shown via contour lines. Each line indicate the
wave front with same traveltime value. The shortest path trajectories can be traced by
drawing a curve from the source point which is always transverse to the fronts (white
contour lines).
"""

s_ind = (5, 5)  # To plot the solution, we choose a sample Source point

fig_vmap = plt.figure(figsize=(6, 5))
ax_vmap = fig_vmap.add_subplot(1, 1, 1)
vmap = ax_vmap.pcolormesh(Xr, Yr, V)
s_point = ax_vmap.scatter(xs[s_ind[0]], xs[s_ind[1]], s=100, c="blue", marker="*")
fig_vmap.colorbar(vmap, ax=ax_vmap, label="Velocity")
tctr_pred = ax_vmap.contour(
    Xr, Yr, T_pred[s_ind], levels=12, colors=["w"], linestyles="--", linewidths=2
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

T_ref = np.empty(P.shape[:-1])
dx = [xr[1] - xr[0], yr[1] - yr[0]]
for i, ixs in enumerate(tqdm(range(0, Nx, sparsity), ascii=" >=")):
    for j, jzs in enumerate(range(0, Ny, sparsity)):
        T_ref[i, j] = ffm(V, (ixs, jzs), dx, 2)
        T_ref[i, j] *= distance(V.shape, dx, (ixs, jzs), indexing="ij")

"""
Comparison of the NES solutions (white dashed lines) and the finite-difference method
(black solid lines)
"""

s_inds = [(5, 5), (5, 30), (35, 15), (15, 15)]

fig_vmap, axes = plt.subplots(nrows=2, ncols=2, figsize=(6 * 2, 5 * 2))

for si, ax in zip(s_inds, axes.ravel()):
    vmap = ax.pcolormesh(Xr, Yr, V)
    fig_vmap.colorbar(vmap, ax=ax, label="Velocity")
    s_point = ax.scatter(
        xs[si[0]], xs[si[1]], s=100, c="blue", marker="*", label="Source"
    )
    tctr_ref = ax.contour(
        Xr, Yr, T_ref[si], levels=12, colors=["k"], linestyles="-", linewidths=3
    )
    tctr_pred = ax.contour(
        Xr,
        Yr,
        T_pred[si],
        levels=tctr_ref.levels,
        colors=["w"],
        linestyles="--",
        linewidths=2,
    )
    ax.clabel(tctr_pred, tctr_ref.levels, fmt="{0:.2f} s".format)

axes[0, 0].legend()
plt.show()

"""
From the figure above, one can see that the NES solution perfectly match the
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

s_flat = 19
T_flat_ref = T_ref[s_flat].min(axis=0)
T_flat_pred = T_pred[s_flat].min(axis=0)

fig_vmap = plt.figure(figsize=(8, 6.6))
ax_vmap = fig_vmap.add_subplot(1, 1, 1)
vmap = ax_vmap.pcolormesh(Xr, Yr, V)
s_points = ax_vmap.scatter(np.full_like(ys, xs[s_flat]), ys, s=50, c="blue", marker="*")
fig_vmap.colorbar(vmap, ax=ax_vmap, label="Velocity")
tctr_ref = ax_vmap.contour(
    Xr, Yr, T_flat_ref, levels=9, colors=["k"], linestyles="-", linewidths=3
)
tctr_pred = ax_vmap.contour(
    Xr,
    Yr,
    T_flat_pred,
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

mae = abs(T_ref - T_pred).mean()
rmae = mae / T_ref.mean() * 100
print(f"RMAE = {rmae:.3f} %")

grid_params = T_ref.size
nn_params = Traveltime.count_params()
print(
    f"Neural network compressed the initial solution with {grid_params} \
params to {nn_params} which is ~{grid_params / nn_params:.0f} times"
)

"""
## Conclusions

We showed how to implement the NES architecture using Tensorflow. We utilized
`tf.gradients` inside the neural-network model to emulate the eikonal equation. The
obtained NES solution showed very good accuracy compared to the finite difference method
with a reasonable training time, and moreover showed the promising compression abilities.
For more examples with NES and more detailed information, please see [our github
page](https://github.com/sgrubas/NES) and
[paper](https://doi.org/10.1016/j.jcp.2022.111789).
"""

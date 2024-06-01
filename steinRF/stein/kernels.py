# ---------------------------------------------------------------------------------------- #
#                              KERNELS AND GRADIENTS FOR STEIN                             #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxtyping import Float
from typing import Callable, NamedTuple
from dataclasses import field
import chex

__all__ = [
    'matrix_rbf_and_grad',
    'matrix_matern32_and_grad',
    'matrix_matern12_and_grad',
    'mmd_k_and_grad',
    'distrib_matrix_rbf_and_grad'
]


@jax.jit
def pairwise_distance(X1, X2):  # note - not squared distance
    return jax.vmap(lambda x1: jax.vmap(lambda x2: (x1 - x2))(X2))(X1)


@jax.jit
def pairwise_median(X1, X2):   # note: IS squared distance
    ls = jnp.median(jax.vmap(
        lambda x1: jax.vmap(lambda x2: (x1 - x2)**2)(X2)
    )(X1), axis=(0, 1))
    return ls


# ------------------------------------- BASIC KERNELS ------------------------------------ #
@jax.jit
def matrix_rbf_and_grad(particles, Q, ls=None):
    # calculate kernel matrix and gradient
    Q = 0.5*(Q+Q.T)
    dx = pairwise_distance(particles, particles)  # delta x - n * n * d
    Qx = dx @ Q  #  inputs scaled by conditioning matrix Q - n * n * d

    # # calculate median heuristic - median of squared distances
    med_ls = jnp.median(dx * Qx, axis=(0, 1))
    if ls is not None:
        ls *= med_ls
    else:
        ls = med_ls

    # # calculate kernel and gradient
    dX = jnp.sum(Qx * dx / (2 * ls), axis=-1)# n * n
    K = jnp.exp(-dX)  # n * n
    gradK = Qx / ls * K[:,:,None]  # n * n * d
    return K, gradK


@jax.jit
def matrix_matern32_and_grad(particles, Q, ls=None):
    # Compute delta x - n * n * d
    Q = 0.5 * (Q + Q.T)
    dx = pairwise_distance(particles, particles)
    Qx = dx @ Q  # Scale inputs by conditioning matrix Q - n * n * d

    # Calculate median heuristic - median of squared distances
    med_ls = jnp.median(dx * Qx, axis=(0, 1))
    if ls is not None:
        ls *= med_ls
    else:
        ls = med_ls

    r = jnp.sqrt(jnp.sum(Qx * dx / ls, axis=-1))
    # r = jnp.sqrt(jnp.sum(Qx * dx, axis=-1))  # Euclidean distance scaled by Q

    # Calculate Matern 3/2 kernel
    sqrt_3_r_l = jnp.sqrt(3) * r
    K = (1 + sqrt_3_r_l) * jnp.exp(-sqrt_3_r_l)

    # Calculate gradient of Matern 3/2 kernel
    factor = jnp.sqrt(3) * jnp.exp(-sqrt_3_r_l)
    gradK = factor[:, :, None] * Qx / (ls ** 2)

    return K, gradK


@jax.jit
def matrix_matern12_and_grad(particles, Q, ls=None):
    # Compute delta x - n * n * d
    Q = 0.5 * (Q + Q.T)
    dx = pairwise_distance(particles, particles)
    Qx = dx @ Q  # Scale inputs by conditioning matrix Q - n * n * d

    # Calculate median heuristic - median of squared distances
    med_ls = jnp.median(dx * Qx, axis=(0, 1))
    if ls is not None:
        ls *= med_ls
    else:
        ls = med_ls

    r = jnp.sqrt(jnp.sum(Qx * dx / ls, axis=-1))
    # r = jnp.sqrt(jnp.sum(Qx * dx, axis=-1))  # Euclidean distance scaled by Q

    # Calculate Matern 1/2 kernel
    K = jnp.exp(-r)

    # Calculate gradient of Matern 1/2 kernel
    gradK = Qx / ls * K[:,:,None]

    return K, gradK


# --------------------------------- DISTRIBUTION KERNELS --------------------------------- #
@jax.jit
def pairwise_matrix_rbf_and_grad(p1, p2, ls=1.):
    # calculate kernel matrix and gradient
    Q = jnp.eye(p1.shape[-1])
    Q = 0.5*(Q+Q.T)
    dx = pairwise_distance(p1, p2)  # delta x - n * n * d
    Qx = dx @ Q  #  inputs scaled by conditioning matrix Q - n * n * d

    # # calculate kernel and gradient
    dX = jnp.sum(Qx * dx / (2 * ls), axis=-1)# n * n
    K = jnp.exp(-dX)  # n * n
    gradK = Qx / ls * K[:,:,None]  # n * n * d
    return K, gradK


@jax.jit
def distrib_matrix_rbf_and_grad(X, ls=None):
    d = X.shape[-1]
    med_ls = pairwise_median(X.reshape(-1, d), X.reshape(-1, d))
    if ls is None:
        ls = med_ls
    else:
        ls *= med_ls

    K, gradK = jax.vmap(
        lambda x1: jax.vmap(
            lambda x2: pairwise_matrix_rbf_and_grad(x1, x2, ls)
        )(X)
    )(X)
    
    gradK = gradK.sum(axis=-2)
    return K, gradK

    # Q = jnp.eye(X.shape[-1])
    # Q = 0.5*(Q+Q.T)
    # dX = X[:, None, :, None, :] - X[None, :, None, :, :]
    # Qx = jnp.einsum('ijklp, pq -> ijklq', dX, Q)

    # # calculate median heuristic - median of squared distances (over all samples)
    # med_ls = jnp.median(dX * Qx, axis=(0, 1, 2, 3))
    # if ls is not None:
    #     ls *= med_ls
    # else:
    #     ls = med_ls

    # # calculate kernel and gradient
    # dX = jnp.sum(Qx * dX / (2 * ls), axis=-1)
    # K = jnp.exp(-dX)  # m * m * n * n

    # gradK = Qx / ls * K[..., None]
    # gradK = gradK.sum(axis=-2) # summing over j in k(w_i, w_j)

    # return K, gradK


# -------------------------------------- MMD KERNELS ------------------------------------- #
def energy_dist(dw, i, j):
    return dw[i, j] - 0.5 * dw[i, i] - 0.5 * dw[j, j]


def energy_mmd(X):
    dx = X[:, None, :, None, :] - X[None, :, None, :, :]
    dx = jnp.sqrt((dx**2).sum(axis=-1))
    dw = jnp.mean(dx, axis=(-1, -2))
    i, j = dw.shape
    i, j = jnp.arange(i), jnp.arange(j)
    dw = jax.vmap(lambda _i: jax.vmap(lambda _j: energy_dist(dw, _i, _j))(j))(i)

    return dw


@jax.jit
def mmd(X, ls=1.):
    # distance calculation
    dx = X[:, None, :, None, :] - X[None, :, None, :, :]
    dx = dx**2
    exp_dx = jnp.exp(-(dx/(2.*ls)).sum(axis=-1))
    dX = exp_dx.mean(axis=(-1, -2))
    dX_diag = jnp.diag(dX)
    dX = dX_diag[:, None] + dX_diag - 2 * dX

    return dX


@jax.jit
def K_mmd(X, ls=1., ls_mmd=1.):
    dX = mmd(X, ls_mmd)
    K = jnp.exp(-dX/(2.*ls))

    return K


@jax.jit
def mmd_k_and_grad(X, ls=1.):
    dX = X[:, None, :, None, :] - X[None, :, None, :, :]
    ls_mmd = jnp.median(dX**2, axis=(0, 1, 2, 3))
    dX = mmd(X, ls_mmd)
    ls = jnp.maximum(jnp.median(dX), 1e-5)
    K = jnp.exp(-dX/(2.*ls))

    # calculate gradient of kernel matrix
    @jax.grad
    def gradK(_X):
        return -K_mmd(_X, ls, ls_mmd).sum() / 2  # negative as we want \nabla_w_j k(w_i, w_j)
    K_grad = gradK(X)

    return K, K_grad


# ---------------------------------------------------------------------------------------- #
#                                      DATA TRANSFORMS                                     #
# ---------------------------------------------------------------------------------------- #
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

import tinygp
from tinygp.helpers import JAXArray


class Transform(tinygp.kernels.base.Kernel):
    transform: eqx.Module
    kernel: tinygp.kernels.base.Kernel

    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        return self.kernel.evaluate(self.transform(X1), self.transform(X2))
    
    @jax.jit
    def phi(self, X: JAXArray) -> JAXArray:
        X_transformed = jax.vmap(self.transform)(X)
        return self.kernel.phi(X_transformed)


class ARD(eqx.Module):
    scale: Float[Array, "d"]

    def __init__(self, scale):
        self.scale = jnp.log(scale)

    @property
    def _scale(self):
        return jnp.exp(self.scale)
    
    def __call__(self, X: JAXArray) -> JAXArray:
        return X / self._scale


# -------------------------------------- DEEP KERNEL ------------------------------------- #
class NNTransform(eqx.Module):
    layers: list
    kernel: eqx.Module

    def __init__(self, key, d_l, kernel=tinygp.kernels.ExpSquared(), **kwargs):
        self.kernel = kernel

        # make NN
        nn_keys = jax.random.split(key, len(d_l) - 1)
        layers = []
        for i in range(len(d_l) - 2):
            layers.append(eqx.nn.Linear(d_l[i], d_l[i + 1], key=nn_keys[i]))
        layers.append(eqx.nn.Linear(d_l[-2], d_l[-1], key=nn_keys[-1], use_bias=False))
        self.layers = layers

    def single_eval(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        return x

    @eqx.filter_jit
    def evaluate(self, X):
        X = jax.vmap(self.single_eval)(X)
        return X

    @eqx.filter_jit
    def __call__(self, X1, X2):
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = self.evaluate(X1.reshape(-1, d_x1))
        X2 = self.evaluate(X2.reshape(-1, d_x2))

        return self.kernel(X1, X2)

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
class MLP(eqx.Module):
    layers: list
    scale: Float[Array, "d"]

    def __init__(self, key, in_dim, out_dim, d_hidden=32, n_hidden=3, **kwargs):
        key, subkey = jax.random.split(key)

        layers = [eqx.nn.Linear(in_dim, d_hidden, key=subkey)]
        for i in range(n_hidden - 1):
            key, subkey = jax.random.split(key)
            layers.append(eqx.nn.Linear(d_hidden, d_hidden, key=subkey))
        key, subkey = jax.random.split(key)
        layers.append(eqx.nn.Linear(d_hidden, out_dim, key=subkey, use_bias=False))
        self.layers = layers
        self.scale = jnp.log(jnp.ones(out_dim))

    @property
    def _scale(self):
        return jnp.exp(self.scale)

    @eqx.filter_jit
    def __call__(self, x: JAXArray) -> JAXArray:
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        x = self.layers[-1](x)
        x /= self._scale
        return x

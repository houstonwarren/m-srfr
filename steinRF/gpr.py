# ---------------------------------------------------------------------------------------- #
#                              Marginal SRFR GP IMPLEMENTATION                             #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import jit, vmap
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from jax.lax import cond
from tensorflow_probability.substrates.jax import distributions as tfd

from steinRF.utils import stabilize

# ------------------------------------- MARGINAL RFF ------------------------------------- #
class MaRFF(eqx.Module):
    w: Float[Array, "p R d"]
    b: Float[Array, "R"]
    variance: Float

    def __init__(self, key, d, R, p, variance=jnp.float32(1.), base_kernel=None, sampling="qmc"):
        if base_kernel is None:
            base_kernel = RBF()

        wkeys = jax.random.split(key, p)
        ws = []
        for wkey in wkeys:
            w = base_kernel.sample(wkey, (R, d), method=sampling)
            ws.append(w)
        self.w = jnp.array(ws)

        # self.b = jax.random.uniform(key, (R,)) * 2 * jnp.pi
        self.b = (halton_samples(key, R, 1) * 2 * jnp.pi).reshape(-1)
        self.variance = jnp.log(variance)
    
    @property
    def R(self):
        return self.b.shape[0]

    @jit
    def phi(self, _X):
        cos_feats = jax.vmap(lambda _w: 
            jnp.sqrt(2) * jnp.cos(_X @ _w.T + self.b), 0
        )(self.w)
        sin_feats = jax.vmap(lambda _w: 
            jnp.sqrt(2) * jnp.sin(_X @ _w.T + self.b), 0
        )(self.w)

        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)
        return projected / jnp.sqrt(2 * self.R)

    # @jit
    # def evaluate(self, X1, X2):
    #     # altnerative RFF implementation cos(w^T(x1 - x2))
    #     cos_feats =  jnp.cos(self.w @ (X1 - X2))
    #     return cos_feats.mean()
    
    # @jit 
    # def evaluateK(self, X1, X2):
    #     K = vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)
    #     return K

    # @jit
    # def A(self, X):
    #     phiX = self.phi(X)
    #     return jnp.exp(self.variance) * phiX.T @ phiX

    @jit
    def __call__(self, X1, X2):
        return jnp.exp(self.variance) * self.phi(X1) @ self.phi(X2).T


# ------------------------------ NONSTATIONARY MARGINAL RFF ------------------------------ #
class NMarRFF(eqx.Module):
    w: Float[Array, "R d2"]
    b: Float[Array, "R"]
    variance: Float

    def __init__(self, key, d, R, variance=jnp.float32(1.), base_kernel=None, sampling="qmc"):
        if base_kernel is None:
            base_kernel = RBF()
        
        # sample pairs of frequencies
        self.w = base_kernel.sample(key, (R, d * 2), method=sampling)
        
        # self.b = jax.random.uniform(key, (R,)) * 2 * jnp.pi
        self.b = (halton_samples(key, R * 2, 1) * 2 * jnp.pi).reshape(-1)
        self.variance = jnp.log(variance)
    
    @property
    def R(self):
        return self.b.shape[0] // 2

    @property
    def d(self):
        return self.w.shape[1] // 2 

    @jit
    def phi(self, _X):
        d, R = self.d, self.R
        w1 = self.w[:, :d]
        w2 = self.w[:, d:]
        b1 = self.b[:R]
        b2 = self.b[R:]

        # multiply by sqrt 2?
        cos_feats = jnp.cos(_X @ w1.T + b1) + jnp.cos(_X @ w2.T + b2)
        sin_feats = jnp.sin(_X @ w1.T + b1) + jnp.sin(_X @ w2.T + b2) 
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected / jnp.sqrt(4 * R)

    @jit
    def evaluate(self, X1, X2):
        d, R = self.d, self.R
        w1 = self.w[:, :d]
        w2 = self.w[:, d:]

        # altnerative RFF implementation cos(w^T(x1 - x2))
        cos_feats =  jnp.cos(self.w @ (X1 - X2))
        return cos_feats.mean()
    
    @jit 
    def evaluateK(self, X1, X2):
        K = vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)
        return K

    @jit
    def A(self, X):
        phiX = self.phi(X)
        return jnp.exp(self.variance) * phiX.T @ phiX

    @jit
    def __call__(self, X1, X2):
        return jnp.exp(self.variance) * self.phi(X1) @ self.phi(X2).T


# --------------------------------- SPECTRAL MIXTURE SRFR -------------------------------- #
class SpecMixRFF(eqx.Module):
    w: Float[Array, "p R d"]
    b: Float[Array, "R"]
    u: Float[Array, "p q"]
    sigma: Float[Array, "p q d d"]
    omega: Float[Array, "p q"]
    variance: Float

    def __init__(self, key, d, R, p, variance=jnp.float32(1.), base_kernel=None, sampling="qmc"):
        if base_kernel is None:
            base_kernel = RBF()

        wkeys = jax.random.split(key, p)
        ws = []
        for wkey in wkeys:
            w = base_kernel.sample(wkey, (R, d), method=sampling)
            ws.append(w)
        self.w = jnp.array(ws)

        # self.b = jax.random.uniform(key, (R,)) * 2 * jnp.pi
        self.b = (halton_samples(key, R, 1) * 2 * jnp.pi).reshape(-1)
        self.variance = jnp.log(variance)
    
    @property
    def R(self):
        return self.b.shape[0]

    @jit
    def phi(self, _X):
        cos_feats = jax.vmap(lambda _w: 
            jnp.sqrt(2) * jnp.cos(_X @ _w.T + self.b), 0
        )(self.w)
        sin_feats = jax.vmap(lambda _w: 
            jnp.sqrt(2) * jnp.sin(_X @ _w.T + self.b), 0
        )(self.w)

        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)
        return projected / jnp.sqrt(2 * self.R)

    # @jit
    # def evaluate(self, X1, X2):
    #     # altnerative RFF implementation cos(w^T(x1 - x2))
    #     cos_feats =  jnp.cos(self.w @ (X1 - X2))
    #     return cos_feats.mean()
    
    # @jit 
    # def evaluateK(self, X1, X2):
    #     K = vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)
    #     return K

    # @jit
    # def A(self, X):
    #     phiX = self.phi(X)
    #     return jnp.exp(self.variance) * phiX.T @ phiX

    @jit
    def __call__(self, X1, X2):
        return jnp.exp(self.variance) * self.phi(X1) @ self.phi(X2).T


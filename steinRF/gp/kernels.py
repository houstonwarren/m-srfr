# ---------------------------------------------------------------------------------------- #
#                                          KERNELS                                         #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import jit, vmap
from jaxtyping import Array, Float
from typing import Callable
import tinygp
from .sampling import sample_rff
from functools import partial
from tinygp.helpers import JAXArray

from steinRF.stein.samplers import GMMReparam, WeightedGMMReparam, SVGD

# ----------------------------------------- UTILS ---------------------------------------- #
def min_max_dist(X):
    assert jnp.ndim(X) == 2

    # calculate distance max/min
    train_x_sort = jnp.sort(X, axis=-2)
    max_dist = train_x_sort[..., -1, :] - train_x_sort[..., 0, :]
    dists = train_x_sort[..., 1:, :] - train_x_sort[..., :-1, :]
    dists = jnp.where(dists == 0., 1e-10, dists)
    sorted_dists = jnp.sort(dists, axis=-2)
    min_dist = sorted_dists[..., 0, :]

    return min_dist, max_dist


# -------------------------------------- RFF KERNEL -------------------------------------- #
class RFF(tinygp.kernels.base.Kernel):
    w: Float[Array, "R d"]
    b: Float[Array, "R"]
    R: int = eqx.field(static=True)

    def __init__(self, 
            key: jax.random.PRNGKey, 
            R, 
            d,
            dist="normal", 
            sampling="qmc",
            prior=None
        ):

        self.R = R        
        if prior is None:
            self.w, self.b = sample_rff(key, R, d, dist=dist, sampling=sampling)
        else:
            _, self.b = sample_rff(key, R, d, dist=dist, sampling=sampling)
            self.w = prior

    @classmethod
    def initialize_from_data(cls, key, R, X):
        d = X.shape[-1]
        min_dist, _ = min_max_dist(X)
        w = jax.random.uniform(key, (R, d), minval=0, maxval=0.5) / min_dist

        return cls(key, R, d, prior=w)
    
    @jax.jit
    def phi(self, _X):
        cos_feats = jnp.sqrt(2) * jnp.cos(_X @ self.w.T + self.b)
        sin_feats = jnp.sqrt(2) * jnp.sin(_X @ self.w.T + self.b)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected / jnp.sqrt(2 * self.R)
    
    @jax.jit
    def evaluate(self, X1, X2):
        phiX1 = self.phi(X1)
        phiX2 = self.phi(X2)
        return phiX1 @ phiX2.T


# -------------------------------------- RFF KERNEL -------------------------------------- #
class MixRFF(eqx.Module):
    w: Float[Array, "q R d"]
    b: Float[Array, "q R"]
    q: int = eqx.field(static=True)
    R: int = eqx.field(static=True)

    def __init__(self, 
            key: jax.random.PRNGKey,
            q:int, R:int, d:int,
            dist="normal", 
            sampling="qmc",
            prior=None
        ):

        self.q = q
        self.R = R
        if prior is None:
            keys = jax.random.split(key, self.q)
            wb = [sample_rff(k, R, d, dist=dist, sampling=sampling) for k in keys]
            w, b = zip(*wb)
            self.w = jnp.array(w)
            self.b = jnp.array(b)
        else:
            keys = jax.random.split(key, self.q)
            self.w = prior
            self.b = jnp.array([
                sample_rff(k, R, d, dist=dist, sampling=sampling)[1] for k in keys
            ]).reshape(q, R)

    @classmethod
    def initialize_from_data(cls, key, q, R, X):
        d = X.shape[-1]
        min_dist, _ = min_max_dist(X)
        w = jax.random.uniform(key, (q, R, d), minval=0, maxval=0.5) / min_dist

        return cls(key, q, R, d, prior=w)
    
    @jax.jit
    def _phi(self, _X, w, b):
        cos_feats = jnp.sqrt(2) * jnp.cos(_X @ w.T + b)
        sin_feats = jnp.sqrt(2) * jnp.sin(_X @ w.T + b)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected / jnp.sqrt(2 * self.R)
    
    def phi(self, _X):
        phiX = vmap(self._phi, (None, 0, 0))(_X, self.w, self.b)

        return phiX
    
    @jax.jit
    def evaluate(self, X1, X2):
        phiX1 = self.phi(X1)
        phiX2 = self.phi(X2)
        K = vmap(lambda phiX1_p, phiX2_p: phiX1_p @ phiX2_p.T)(phiX1, phiX2)

        return K


# ----------------------------------- NONSTATIONARY RFF ---------------------------------- #
class NonstationaryRFF(tinygp.kernels.base.Kernel):
    w: Float[Array, "R d2"]
    b: Float[Array, "R"]
    R: int = eqx.field(static=True)
    d: int = eqx.field(static=True)

    def __init__(self, key, R, d, dist="normal", sampling="qmc", prior=None):
        self.R = R
        self.d = d
        
        if prior is None:
            self.w, _ = sample_rff(key, R, d * 2, dist=dist, sampling=sampling)
            self.b = sample_rff(key, R * 2, 1, dist=dist, sampling=sampling)[1].reshape(-1)
        else:
            self.w = prior
            self.b = sample_rff(key, R * 2, 1, dist=dist, sampling=sampling)[1].reshape(-1)

    @classmethod
    def initialize_from_data(cls, key, R, X):
        d = X.shape[-1]
        min_dist, _ = min_max_dist(X)
        min_dist = jnp.concatenate([min_dist, min_dist], axis=-1)
        w = jax.random.uniform(key, (R, d * 2), minval=0, maxval=0.5) / min_dist

        return cls(key, R, d, prior=w)

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
        return self.phi(X1) @ self.phi(X2).T


# ------------------------------- NONSTATIONARY MIXTURE RFF ------------------------------ #
class NMixRFF(eqx.Module):
    w: Float[Array, "q R d2"]
    b: Float[Array, "q 2R"]
    q: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    R: int = eqx.field(static=True)

    def __init__(self, 
            key: jax.random.PRNGKey,
            q:int, R:int, d:int,
            dist="normal", 
            sampling="qmc",
            prior=None
        ):

        self.q = q
        self.R = R
        self.d = d

        if prior is None:
            keys = jax.random.split(key, self.q)
            self.w = jnp.array([
                sample_rff(k, R, d * 2, dist=dist, sampling=sampling)[0] for k in keys
            ])
            b = [
                sample_rff(k, R * 2, 1, dist=dist, sampling=sampling)[1].reshape(-1)
                for k in keys
            ]
            self.b = jnp.array(b).reshape(q, 2 * R)
        else:
            keys = jax.random.split(key, self.q)
            self.w = prior
            self.b = jnp.array([
                sample_rff(k, 2 * R, d, dist=dist, sampling=sampling)[1] for k in keys
            ]).reshape(q, 2 * R)

    @classmethod
    def initialize_from_data(cls, key, q, R, X):
        d = X.shape[-1]
        min_dist, _ = min_max_dist(X)
        min_dist = jnp.concatenate([min_dist, min_dist], axis=-1)
        w = jax.random.uniform(key, (q, R, d * 2), minval=0, maxval=0.5) / min_dist

        return cls(key, q, R, d, prior=w)
    
    @jax.jit
    def _phi(self, _X, w, b):
        d, R = self.d, self.R
        w1 = w[:, :d]
        w2 = w[:, d:]
        b1 = b[:R]
        b2 = b[R:]

        # multiply by sqrt 2?
        cos_feats = jnp.cos(_X @ w1.T + b1) + jnp.cos(_X @ w2.T + b2) 
        sin_feats = jnp.sin(_X @ w1.T + b1) + jnp.sin(_X @ w2.T + b2) 
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected / jnp.sqrt(4 * R)

    @jax.jit
    def phi(self, _X):
        phiX = vmap(self._phi, (None, 0, 0))(_X, self.w, self.b)

        return phiX
    
    @jax.jit
    def evaluate(self, X1, X2):
        phiX1 = self.phi(X1)
        phiX2 = self.phi(X2)
        K = vmap(lambda phiX1_p, phiX2_p: phiX1_p @ phiX2_p.T)(phiX1, phiX2)

        return K


# -------------------------------- SPECTRAL MIXTURE KERNEL ------------------------------- #
class SMK(tinygp.kernels.base.Kernel):
    w: Float[Array, "m"]
    u: Float[Array, "m d"]
    l: Float[Array, "m d"]  # note - this is the variance, not the lengthscale

    def __init__(self, m, d, prior=None, randomize=False, key=None):
        if prior is None:
            if randomize:
                key = jax.random.PRNGKey(0) if key is None else key
                # self.w = jnp.log(jax.nn.softmax(jax.random.uniform(key, (m,))))
                self.w = jnp.log(jax.random.uniform(key, (m,)))
                self.u = jnp.log(jax.random.uniform(key, (m, d), minval=0, maxval=3))
                self.l = jnp.log(jax.random.uniform(key, (m, d), minval=0, maxval=3))
            else:
                # self.w = jnp.log(jnp.ones((m)) / m)
                self.w = jnp.log(jnp.ones((m)))
                self.u = jnp.log(jnp.ones((m, d)))
                self.l = jnp.log(jnp.ones((m, d)))

        else:
            w, u, l = prior
            self.w = jnp.log(w)
            self.u = jnp.log(u)
            self.l = jnp.log(l)

    @property
    def _w(self):
        # return jax.nn.softmax(jnp.exp(self.w))
        return jnp.exp(self.w)
    
    @property
    def _l(self):
        return jnp.exp(self.l)

    @property
    def _u(self):
        return jnp.exp(self.u)

    @property
    def _params(self):
        return (self._w, self._u, self._l)
    
    @jit
    def initialize_from_data(self, key, X, y):
        min_dist, max_dist = min_max_dist(X)
        d = X.shape[-1]
        m = self.u.shape[0]

        # draw samples
        w = y.std() / m * jnp.ones(m)
        u = jax.random.uniform(key, (m, d), minval=0, maxval=0.5) / min_dist
        l = jnp.abs(jax.random.normal(key, (m, d)) * max_dist)

        return self.__class__(m, d, prior=(w, u, l))

    @jit
    def p_m(self, dx: JAXArray, w: JAXArray, u: JAXArray, l: JAXArray) -> JAXArray:
        t1 = jnp.exp(-2 * jnp.pi**2 * l * dx**2)
        t2 = jnp.cos(2 * jnp.pi * u * dx)
        return w * jnp.prod(t1 * t2)

    @jit
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dX = X1 - X2
        w, u, l = self._params

        k = jax.vmap(self.p_m, (None, 0, 0, 0))(dX, w, u, l).sum()
        return k


# -------------------------------- SPARSE SPECTRAL MIXTURE ------------------------------- #
class SparseSMK(eqx.Module):
    # GMM params
    w: Float[Array, "p m"]
    u: Float[Array, "p m d"]
    l: Float[Array, "p m d"]
    
    # sampling
    sampler: eqx.Module = eqx.field(static=True)

    # utilities
    m: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    R: int = eqx.field(static=True)

    def __init__(self, p, m, d, R, prior=None, sampler=None, key=None):
        key = jax.random.PRNGKey(0) if key is None else key
        if prior is None:
            # self.w = jnp.log(jax.nn.softmax(jax.random.uniform(key, (m,))))
            self.w = jnp.log(jax.random.uniform(key, (p, m,)))
            self.u = jnp.log(jax.random.uniform(key, (p, m, d), minval=0, maxval=3))
            self.l = jnp.log(jax.random.uniform(key, (p, m, d), minval=0, maxval=3))
        else:
            w, u, l = prior
            self.w = jnp.log(w)
            self.u = jnp.log(u)
            self.l = jnp.log(l)

        if sampler is None:
            sampler = GMMReparam()
        self.sampler = sampler

        self.p = p  # number of particles
        self.m = m  # numer of mixture components
        self.d = d  # number of dimensions
        self.R = R  # number of random features

    @property
    def _w(self):
        return jnp.exp(self.w)
    
    @property
    def _l(self):
        return jnp.exp(self.l)
    
    @property
    def _u(self):
        return jnp.exp(self.u)
    
    @property
    def _params(self):
        return (self._w, self._u, self._l)

    def initialize_from_data(self, key, X, y):
        min_dist, max_dist = min_max_dist(X)
        d = X.shape[-1]
        p, m, R, d = self.p, self.m, self.R, self.d

        # draw samples
        w = y.std() / m * jnp.ones((p, m))
        u = jax.random.uniform(key, (p, m, d), minval=0, maxval=0.5) / min_dist
        l = jnp.abs(jax.random.normal(key, (p, m, d)) * max_dist)

        # sampler
        sampler = self.sampler.__class__(p, m, d, R, params=(w, u, l))

        return self.__class__(p, m, d, R, prior=(w, u, l), sampler=sampler)
    
    def update(self, X: JAXArray) -> eqx.Module:
        # calculate optimal R_q
        sampler = self.sampler.__class__(
            X, self.p, self.m, self.d, self.R, 
            params=self._params
        )

        the_new_me = self.__class__(
            self.p, self.m, self.d, self.R, 
            prior=self._params, sampler=sampler
        )

        return the_new_me
    
    @eqx.filter_jit # calculate phi(x) for a single mixture component
    def phi_q(self, x: JAXArray, w: JAXArray, u: JAXArray) -> JAXArray:  
        # compute mixture features
        x_u = 2 * jnp.pi * x @ u.T
        cos_feats = jnp.cos(x_u)
        sin_feats = jnp.sin(x_u)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)
        projected *= jnp.sqrt(w / self.R)

        return projected

    @eqx.filter_jit
    def _phi(self, _X: JAXArray, w: JAXArray,  u: JAXArray) -> JAXArray:
        u = jnp.moveaxis(u, -1, -2)
        phiX = vmap(
            lambda w_p, u_p: vmap(lambda w_m, u_m: self.phi_q(_X, w_m, u_m))(w_p, u_p)
        )(w, u)

        phiX = jnp.moveaxis(phiX, 1, 2)
        phiX = phiX.reshape(self.p, _X.shape[0], -1)

        return phiX

    @eqx.filter_jit
    def _phi_with_samples(self, X: JAXArray, samples: JAXArray) -> JAXArray:
        R_q, w, omega = samples[..., 0], samples[..., 1], samples[..., 2:]
        x_u = 2 * jnp.pi * X @ omega.T
        cos_feats = jnp.cos(x_u) * jnp.sqrt(w / R_q)
        sin_feats = jnp.sin(x_u) * jnp.sqrt(w / R_q)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected
    
    @eqx.filter_jit
    def phi_with_samples(self, X: JAXArray, samples: JAXArray) -> JAXArray:
        return jax.vmap(self._phi_with_samples, (None, 0))(X, samples)
    
    @eqx.filter_jit
    def phi(self, X: JAXArray, key: jax.random.PRNGKey =jax.random.PRNGKey(0)) -> JAXArray:
        samples = self.sampler(key, self._params, self.R)
        # return samples
        return self.phi_with_samples(X, samples)

    @eqx.filter_jit
    def evaluate(self, X1, X2):
        return self.__call__(X1, X2)

    @partial(jit, static_argnums=(3,))
    def __call__(
        self, X1: JAXArray, 
        X2: JAXArray, 
        samples: JAXArray | None=None,
        key: jax.random.PRNGKey =jax.random.PRNGKey(0)
    ) -> JAXArray:
        
        if samples is None:
            samples = self.sampler(key, self._params, self.R)
        # return samples

        phiX1 = self.phi_with_samples(X1, samples)
        phiX2 = self.phi_with_samples(X2, samples)
        K_stack = vmap(lambda phiX1_p, phiX2_p: phiX1_p @ phiX2_p.T)(phiX1, phiX2)
        return K_stack


# ------------------------------ GENERALIZED SPECTRAL KERNEL ----------------------------- #
class GSK(tinygp.kernels.base.Kernel):
    sigma: Float[Array, "m"]
    w: Float[Array, "m d"]
    l: Float[Array, "m d"]  # note - this is the variance, not the lengthscale
    k: Callable[[JAXArray], JAXArray] = eqx.field(static=True)

    def __init__(self, m, d, prior=None, k="rbf", randomize=False, key=None):
        if prior is None:
            if randomize:
                key = jax.random.PRNGKey(0) if key is None else key
                # self.w = jnp.log(jax.nn.softmax(jax.random.uniform(key, (m,))))
                self.sigma = jnp.log(jax.random.uniform(key, (m,)))
                self.w = jnp.log(jax.random.uniform(key, (m, d), minval=0, maxval=3))
                self.l = jnp.log(jax.random.uniform(key, (m, d), minval=0, maxval=3))
            else:
                # self.w = jnp.log(jnp.ones((m)) / m)
                self.sigma = jnp.log(jnp.ones((m)))
                self.w = jnp.log(jnp.ones((m, d)))
                self.l = jnp.log(jnp.ones((m, d)))

        else:
            sigma, w, l = prior
            self.sigma = jnp.log(sigma)
            self.w = jnp.log(w)
            self.l = jnp.log(l)

        if k == "rbf":
            @jax.jit
            def k(dX):
                return jnp.exp(-2 * jnp.pi**2 * jnp.sum(dX**2, axis=-1))
            self.k = k
        else:
            raise NotImplementedError("Kernel not implemented")

    @property
    def _sigma(self):
        # return jax.nn.softmax(jnp.exp(self.w))
        return jnp.exp(self.sigma)
    
    @property
    def _l(self):
        return jnp.exp(self.l)

    @property
    def _u(self):
        return jnp.exp(self.u)

    @property
    def _params(self):
        return (self._sigma, self._u, self._l)
    
    def to_rff(self, key, R, sampler, **kwargs):
        """Create a random GSK from a GSK using a sampler."""
        if isinstance(sampler, GMMReparam):
            samples = sampler(key, self._params, R)
            sigma, w = samples[..., 0], samples[..., 1:]

        elif isinstance(sampler, SVGD):
            particles = kwargs.pop(  # particle init defaults to GMM sample
                "particles", GMMReparam()(key, self._params, R)
            )
            samples = sampler(key, self._params, R)
            w = sampler(particles=particles, theta={
                'theta': self._params}, **kwargs
            )
            sigma = self._sigma

        m, d = self.w.shape
        prior = (sigma[0], w)
        new_k = RandomGSK(m=m, d=d, R=R, prior=prior)
        return new_k

    @jit
    def initialize_from_data(self, key, X, y):
        min_dist, max_dist = min_max_dist(X)
        d = X.shape[-1]
        m = self.u.shape[0]

        # draw samples
        sigma = y.std() / m * jnp.ones(m)
        w = jax.random.uniform(key, (m, d), minval=0, maxval=0.5) / min_dist
        l = jnp.abs(jax.random.normal(key, (m, d)) * max_dist)

        return self.__class__(m, d, prior=(sigma, w, l))

    @jit
    def p_m(self, dx: JAXArray, sigma: JAXArray, w: JAXArray, l: JAXArray) -> JAXArray:
        t1 = self.k(dx * sigma)
        t2 = jnp.cos(2 * jnp.pi * w * dx)
        return sigma * jnp.prod(t1 * t2)

    @jit
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        dX = X1 - X2
        w, u, l = self._params

        k = jax.vmap(self.p_m, (None, 0, 0, 0))(dX, w, u, l).sum()
        return k


# ------------------------------ RFF IMPLEMENTATION OF A GSK ----------------------------- #
class RandomGSK(tinygp.kernels.base.Kernel):
    sigma: Float  # limitation currently this must be uniform
    w: Float[Array, "R d"]

    # utilities
    m: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    R: int = eqx.field(static=True)

    def __init__(self, m, d, R, prior=None, key=None):
        self.m = m  # numer of mixture components
        self.d = d  # number of dimensions
        self.R = R  # number of random features

        if prior is None:
            key = jax.random.PRNGKey(0) if key is None else key
            self.w, _ = sample_rff(key, R, d)
            self.sigma = jnp.log(jax.random.uniform(key, (1,)))
        else:
            sigma, w = prior
            self.sigma = jnp.log(sigma)
            self.w = w

    @property
    def _sigma(self):
        # return jax.nn.softmax(jnp.exp(self.w))
        return jnp.exp(self.sigma)
    
    @jax.jit
    def phi(self, _X):
        sigma = self._sigma
        x_u = 2 * jnp.pi * _X @ self.w.T
        cos_feats = jnp.cos(x_u)
        sin_feats = jnp.sin(x_u)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)
        projected *= jnp.sqrt(sigma / (self.R // self.m))

        return projected
    
    @jax.jit
    def evaluate(self, X1, X2):
        phiX1 = self.phi(X1)
        phiX2 = self.phi(X2)
        return phiX1 @ phiX2.T

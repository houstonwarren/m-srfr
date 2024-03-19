# ---------------------------------------------------------------------------------------- #
#                                    SAMPLING TECHNIQUES                                   #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd

from jaxtyping import Array, Float, Int
from tinygp.helpers import JAXArray
from typing import Callable

from steinRF.stein.svgd import vanilla_svgd_step, matrix_svgd_step, mixture_svgd_step, \
    annealing


# ----------------------------------------- SVGD ----------------------------------------- #
class SVGD(eqx.Module):
    target: eqx.Module
    step_fn: Callable = eqx.field(static=True)

    def __init__(self, target, method="svgd"):
        if method == "svgd":
            self.step_fn = vanilla_svgd_step
        elif method == "matrix":
            self.step_fn = matrix_svgd_step
        elif method == "mixture":
            self.step_fn = mixture_svgd_step
        self.target = target

    def __call__(self, 
        epochs: int,
        particles: JAXArray | None,
        R: int = 100,
        **kwargs 
    ):
        ls = kwargs.get("ls", None)
        theta = kwargs.get("theta", {})  # target dist parameters, if any

        # annealing schedule
        anneal = kwargs.get("anneal", "cyclical")
        if anneal == "cyclical":
            c = kwargs.get("c", 5)
            s = kwargs.get("s", 0.5)
            gamma = annealing(epochs, c=c, p=s)
        else:
            gamma = lambda t: 1.

        # optimizer
        lr = kwargs.get("lr", 1e-2)
        opt = optax.adamw(lr)

        # initialize particles
        if particles is None:
            key = kwargs.get("key", jax.random.PRNGKey(0))
            d = kwargs.get("d", None)
            if d is None:
                raise ValueError("dimension d must be provided")
            particles = jax.random.uniform(key, (R, d))
        else:
            R, d = particles.shape

        # define an svgd step
        @eqx.filter_jit
        def svgd_step(w, opt_state, gamma_t):
            score, w_grads = self.target.grad(w, **theta)

            # apply step
            velocities = self.step_fn(w, w_grads, ls=ls, gamma=gamma_t)

            # post-process for stability
            velocities = jnp.where(
                jnp.isnan(velocities) | jnp.isinf(velocities), 0.0, velocities
            )

            # apply updates
            updates, opt_state = opt.update(velocities, opt_state, params=w)
            w = optax.apply_updates(w, updates)
            
            return w, opt_state, -score
        
        # initalize optimizer
        opt_state = opt.init(particles)

        # loop over epochs
        verbose = kwargs.get("verbose", False)
        print_iter = kwargs.get("print_iter", 25)
        history = kwargs.get("history", False)
 
        loss_vals = [-self.target.score(particles, **theta)]
        histories = [particles]
        for epoch in range(epochs):
            particles, opt_state, loss = svgd_step(particles, opt_state, gamma(epoch))
            loss_vals.append(loss)
            if history: history.append(particles)
            if verbose and epoch % print_iter == 0:
                print(f"epoch {epoch},loss: {loss}")

        if history:
            return particles, jnp.array(loss_vals), jnp.array(histories)
        else:
            return particles, jnp.array(loss_vals)


# --------------------------------- REPARAMETRIZATION GMM -------------------------------- #
class GMMReparam(eqx.Module):
    def __init__(self, *args, **kwargs):
        # nothing needed here, just allows for cross funcitonal use with other samplers
        pass

    @eqx.filter_jit
    def __call__(self, 
        key: jax.random.PRNGKey, 
        params: tuple[JAXArray, JAXArray, JAXArray], 
        R: Int
    ) -> JAXArray:
        
        # sample mixture components via the reparam trick
        w, u, l = params
        w = jnp.atleast_2d(w)
        u = jnp.atleast_3d(u)
        l = jnp.sqrt(jnp.atleast_3d(l))  #  we parametrize l as the variance in SMK
        p, m, d = u.shape
        m_per_d = R // m  # uniform sampling across mixture components

        # noise for reparametrization trick
        epsilon = jr.normal(key, (m_per_d, d))

        w = jnp.repeat(w[..., None, None], m_per_d, axis=-2)  # p * m -> [p, m, R, d]
        samples = u[..., None, :] + jnp.einsum("pmd,Rd->pmRd", l, epsilon)
        R_q = jnp.ones((p, m)) * m_per_d
        n_q = jnp.repeat(R_q[..., None, None], m_per_d, axis=-2)
        samples = jnp.concatenate([n_q, w, samples], axis=-1).reshape(p, -1, d + 2)

        return samples  # [p, m * R, d + 2]


# --------------------------------- WEIGHTED GMM REPARAM --------------------------------- #
class WeightedGMMReparam(GMMReparam):
    epsilon: Float[Array, "R d"]

    # masking
    mask: Float[Array, "p R"]
    R_q: Float[Array, "p m"]

    # utilities
    m: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    R: int = eqx.field(static=True)

    def __init__(self, key, p, m, d, R, params=None, **kwargs):
        self.key = key

        self.p = p  # number of particles
        self.m = m  # numer of mixture components
        self.d = d  # number of dimensions
        self.R = R  # number of random features

        # noise for reparametrization trick
        self.epsilon = jax.random.normal(key, (R, d))

        # make sample mask based on current mixture params
        if params is None:
            assert R % m == 0
            mask = []
            samples_per_m = R // m
            self.R_q = jnp.ones((p, m)) * samples_per_m
            mask = jnp.concatenate([  # evenly split between mixtures
                jnp.arange(i*self.R, i*self.R + samples_per_m) for i in range(m)
            ])
            self.mask = jnp.repeat(mask[None, :], p, axis=0)
        else:
            self.R_q, self.mask = self.make_mask(params)

    # @eqx.filter_jit
    def make_mask(self, X: JAXArray) -> JAXArray:
        R_q = self.allocate_r_m(X)
        mask = []
        for row in R_q:
            row_mask = jnp.concatenate([jnp.arange(i*self.R, i*self.R+c) for i, c in enumerate(row)])
            mask.append(row_mask)
        return R_q, jnp.array(mask)

    def allocate_r_m(self, X: JAXArray): # -> JAXArray:
        """
        Allocate how many particles each mixture component will have.
        This will happen outside the gradient loop.
        """
        # calculate g(x_i - x_j)
        dX = X[:, None, :] - X[None, :, :]
        w, u, l = self._params
        _k_q = self.k_q(dX, u, l)
        k_q_2 = self.k_q(2 * dX, u, l)
        g_q_X = 1 + k_q_2 - 2 * _k_q**2

        # calculate the summation over i, i<j
        a_q = w * vmap(
            lambda gq_p: jax.vmap(
                lambda gq_p_q: jnp.sum(jnp.tril(gq_p_q, k=-1))
            )(gq_p)
        )(g_q_X)

        a_q = a_q / a_q.sum(axis=-1)[:, None]
        R_q = a_q * self.R
        largest_vals = jnp.argsort(R_q, axis=-1)[:, ::-1]  # sort the indices
        
        # p * m -> number of particles for each mixture component in p mixtures
        R_q = jnp.floor(R_q).astype(int)
        R_q = jnp.where(R_q == 0, 1, R_q)

        # correct any mixtures particle counts
        for i in range(self.p):
            diff = self.R - R_q[i].sum()
            if diff > 0:
                R_q = R_q.at[i, largest_vals[i, :diff]].set(
                    R_q[i, largest_vals[i, :diff]] + 1
                )

            diff = R_q[i].sum() - self.R
            if diff > 0:
                R_q = R_q.at[i, largest_vals[i, -diff:]].set(
                    R_q[i, largest_vals[i, -diff:]] - 1
                )

        return R_q

    @eqx.filter_jit
    def k_q(self, dX: JAXArray, u: JAXArray, l: JAXArray) -> JAXArray:
        """Calculate the GMM component for each mixture"""
        # w = self._w
        u_dX = jnp.einsum('lmk,ijk->lmijk', u, dX)
        l_dX = jnp.einsum('lmk,ijk->lmijk', l, dX**2)
        t1 = jnp.exp(-2 * jnp.pi**2 * l_dX)
        t2 = jnp.cos(2 * jnp.pi * u_dX)
        _k_q = jnp.prod(t1 * t2, axis=-1)

        # w_k_q = jnp.einsum('ij,ijkl->ijkl', w, k_q)
        return _k_q

    @eqx.filter_jit
    def sample(self) -> JAXArray:
        # sample mixture components via the reparam trick
        w = jnp.repeat(self._w[..., None, None], self.R, axis=-2)  # p * m -> [p, m, R, d]
        samples = self._u[..., None, :] + jnp.einsum("pmd,Rd->pmRd", self._l, self.epsilon)
        n_q = jnp.repeat(self.R_q[..., None, None], self.R, axis=-2)
        samples = jnp.concatenate([n_q, w, samples], axis=-1).reshape(self.p, -1, self.d + 2)

        return samples  # [p, m * R, d + 2]
    
    @eqx.filter_jit
    def masked_sample(self) -> JAXArray:
        samples = self.sample()
        dim1_indices = jnp.arange(samples.shape[0])[:, None]
        dim1_indices = jnp.broadcast_to(dim1_indices, self.mask.shape)
        samples = samples[dim1_indices, self.mask, :]

        return samples   # [p, R, 2 + d], where mixtures are stacked  

    def __call__(self, ):
        return self.masked_sample()


# ------------------------------------ NEURAL SAMPLER ------------------------------------ #
class NeuralSampler(eqx.Module):
    layers: list

    def __init__(self, key, d_l, **kwargs):
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
    def __call__(self, noise):
        z = vmap(self.single_eval)(jnp.atleast_2d(noise))
        return z


# ----------------------------------- LANGEVIN SAMPLER ----------------------------------- #
class LangevinSampler(eqx.Module):
    layers: list

    def __init__(self, key, d_l, **kwargs):
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
    def __call__(self, noise):
        z = vmap(self.single_eval)(jnp.atleast_2d(noise))
        return z


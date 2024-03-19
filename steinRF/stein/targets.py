# ---------------------------------------------------------------------------------------- #
#                              DISTRIBUTIONAL TARGETS FOR SVGD                             #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.scipy.stats import multivariate_normal
from functools import partial
from typing import Tuple, Union, Any, NamedTuple
import chex
from jaxtyping import Float, Array


# -------------------------------------- NLL TARGET -------------------------------------- #
class NLLTarget(eqx.Module):
    @jax.jit
    def grad(self, params, static, y):
        @jax.value_and_grad
        def log_prob(_params):
            return -eqx.combine(_params, static).nll(y)
           
        return log_prob(params)

    @jax.jit
    def split_grad(
        self, params: Tuple[chex.ArrayTree, chex.ArrayTree], 
        static: chex.ArrayTree, 
        y: Float[Array, "d"]
    ):
        """Split SVGD and regular GD gradients for prior-augmented loss."""
        @jax.value_and_grad
        def log_prob(split_params):
            _params = eqx.combine(split_params[0], split_params[1])
            model = eqx.combine(_params, static)
            log_like = -model.nll(y)

            return log_like
           
        return log_prob(params)

    @jax.jit
    def score(self, params, static, y):
          return -eqx.combine(params, static).nll(y)

    @jax.jit
    def __call__(self, params, static, y):
        return self.score(params, static, y)


class PriorNLLTarget(eqx.Module):
    prior: eqx.Module

    def __init__(self, prior):
        self.prior = prior

    @jax.jit
    def grad(self, params, static, y):
        @jax.value_and_grad
        def log_prob(_params):
            model = eqx.combine(_params, static)
            log_like = -model.nll(y)
            prior_score = jax.tree_flatten(jax.tree_map(
                lambda x: self.prior.score(x).sum(), _params 
            ))[0][0]

            return log_like + prior_score
           
        return log_prob(params)
    
    @jax.jit
    def split_grad(
        self, params: Tuple[chex.ArrayTree, chex.ArrayTree], 
        static: chex.ArrayTree, 
        y: Float[Array, "d"]
    ):
        """Split SVGD and regular GD gradients for prior-augmented loss."""
        @jax.value_and_grad
        def log_prob(split_params):
            _params = eqx.combine(split_params[0], split_params[1])
            model = eqx.combine(_params, static)
            log_like = -model.nll(y)
            prior_score = jax.tree_flatten(jax.tree_map(
                lambda x: self.prior.score(x).sum(), split_params[0] 
            ))[0][0]

            return log_like + prior_score
           
        return log_prob(params)
        
    def grad_nll(self, params, static, y):
        @jax.value_and_grad
        def loss(_params):
            model = eqx.combine(_params, static)
            return model.nll(y)
           
        return loss(params)
    
    @jax.jit
    def score(self, params, static, y):
        model = eqx.combine(params, static)
        log_like = -model.nll(y)
        prior_score = jax.tree_flatten(jax.tree_map(
            lambda x: self.prior.score(x).sum(), params
        ))[0][0]

        return log_like + prior_score

    @jax.jit
    def __call__(self, params, static, y):
        return self.score(params, static, y)
    

# ------------------------- TENSORFLOW PROBABILITY DEFINED TARGET ------------------------ #
class TFTarget(eqx.Module):
    dist: tfd.Distribution

    def __init__(self, dist):
        self.dist = dist

    @jax.jit
    def grad(self, params, static, y):
        @jax.value_and_grad
        def loss(_params):
            return static.log_prob(y).sum()
           
        return loss(params)
    
    @jax.jit
    def score(self, y):
          return self.dist.log_prob(y).sum()

    @jax.jit
    def __call__(self, y):
        return self.dist.log_prob(y).sum()


# -------------------------------------- GMM TARGET -------------------------------------- #
class GMMTarget(eqx.Module):
    @jax.jit
    def grad(self, samples, theta):
        weighted_probs = self.pm_x(samples, theta)  # [n, m]
        normalized_probs = weighted_probs / jnp.sum(weighted_probs, axis=-1)[..., None]  # [n, m]
        y_diff = samples[:, None, ...] - theta[1]  # [n, m, d]
        covs = vmap(lambda l: jnp.diag(l))(theta[2])  # [m, d, d]
        cov_invs = jnp.linalg.inv(covs)  # [m, d, d]
        sigma_m_y_diff = -jnp.einsum("mik,nmk->nmi", cov_invs, y_diff)
        grads = jnp.einsum("nm,nmd->nmd", normalized_probs, sigma_m_y_diff)
        grads = jnp.einsum("nmd->nd", grads)
        return self.score(samples, theta), grads  # [n, d]

    @jax.jit
    def pm_x(self, samples, theta):
        w, u, l = theta
        mvn = tfd.MultivariateNormalDiag(loc=u, scale_diag=jnp.sqrt(l))
        probs = w * vmap(mvn.prob)(samples)  # [n, ..., m]
        min_nonzero = jnp.min(jnp.where(probs > 0, probs, 100)) * 0.99
        probs = jnp.where(probs == 0., min_nonzero, probs)
        return probs

    @jax.jit
    def score(self, samples, theta):
        scores = jnp.log(self.pm_x(samples, theta))
        scores = jax.scipy.special.logsumexp(scores, axis=-1).sum(axis=0)
        return scores

    def __call__(self, samples, theta):
        return self.score(samples, theta)

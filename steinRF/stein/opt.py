# ---------------------------------------------------------------------------------------- #
#                                   OPTAX SVGD OPTIMIZERS                                  #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Float, Array, ArrayLike
from typing import Callable, NamedTuple
import chex
import equinox as eqx
from functools import partial

from steinRF.stein.kernels import *

# ----------------------- PYTREE UTILITIES FOR PARAMETER SPLITTING ----------------------- #
def label_params(tree, *fns):
    for fn, label in fns:
        tree = eqx.tree_at(fn, tree, replace_fn=lambda _: label)
    return tree


# ---------------------------------- BASIC ANNEALED SVGD --------------------------------- #
class SVGDState(NamedTuple):
    count: chex.Array
    gamma: chex.ArrayTree


def _asvgd(
        epochs: int,
        K_k_grad: Callable = matrix_rbf_and_grad, 
        c: int = 5,
        p: Float = 0.5,  # setting to 0 is equivalent to non-annealed SVGD
        ls: chex.Array | None = None,
    ):
    "Annealed stein variational gradient descent."

    @jax.jit
    def svgd_step(
        particles: Float[Array, "R d"], 
        particle_grads: Float[Array, "R d"], 
        _gamma: Float,
    ) -> Float[Array, "R d"]:
        
        R, d = particles.shape

        # kernel matrix and gradient of kernel matrix
        K, K_grad = K_k_grad(particles, jnp.eye(d), ls=ls)

        pull = _gamma * K @ particle_grads
        repulse = K_grad.sum(axis=1)
        svgd_grads = (pull + repulse) / R
        return -svgd_grads

    def anneal_fn(t: int) -> Float:
        return (jnp.mod(t, epochs/c) / (epochs/c)) ** p
    
    def init_fn(params):
        gamma = jax.tree_map(lambda _: anneal_fn(0), params)
        return SVGDState(count=jnp.zeros([], jnp.int32), gamma=gamma)
    
    def update_fn(
        updates: optax.Updates, 
        state: SVGDState, 
        params: optax.Updates
    ) -> optax.GradientTransformation:
        
        count = state.count
        gamma = jax.tree_map(
            lambda _: anneal_fn(count), state.gamma
        )
        updates = jax.tree_map(
            lambda ps, p_grads, gamma_t: svgd_step(ps, p_grads, gamma_t),
            params, updates, gamma
        )

        return updates, SVGDState(count=count+1, gamma=gamma)
    
    return optax.GradientTransformation(init_fn, update_fn)


def asvgd(
        epochs: int,
        lr: Float,
        K_k_grad: Callable = matrix_rbf_and_grad, 
        c: int = 5,
        p: Float = 0.5,  # setting to 0 is equivalent to non-annealed SVGD
        ls: chex.Array | None = None,
    ):
    "Annealed stein variational gradient descent."

    opt = optax.chain(
        _asvgd(epochs, K_k_grad, c, p, ls),
        optax.scale_by_adam(),
        optax.add_decayed_weights(1e-4, None),
        optax.scale_by_learning_rate(lr)
    )

    return opt

# ---------------------------------- FUNCTIONAL STEIN GD --------------------------------- #
def fsvgd(
        epochs: int,
        alpha: Float = 0.5,
        K_k_grad: Callable = matrix_rbf_and_grad, 
        c: int = 5,
        p: Float = 0.5,
        ls: ArrayLike | None = None,
    ):
    "Functional stein variational gradient descent with annealing."
    
    
    # @partial(jnp.vectorize, signature="(R,d),(R,d),()->(R, d)")
    @jax.jit
    def fsvgd_step(
        particles: Float[Array, "R d"], 
        particle_grads: Float[Array, "R d"], 
        _gamma: Float,
    ) -> Float[Array, "R d"]:
        
        R, d = particles.shape

        # kernel matrix and gradient of kernel matrix
        K, K_grad = K_k_grad(particles, jnp.eye(d), ls=ls)

        pull = _gamma * K @ particle_grads
        repulse = alpha * K_grad.sum(axis=1)
        srfr_grads = (pull + repulse) / K.sum(axis=1, keepdims=True)

        return -srfr_grads
    
    def anneal_fn(t: int) -> Float:
        return (jnp.mod(t, epochs/c) / (epochs/c)) ** p
    
    def init_fn(params):
        gamma = jax.tree_map(lambda _: anneal_fn(0), params)
        return SVGDState(count=jnp.zeros([], jnp.int32), gamma=gamma)
    
    def update_fn(
        updates: optax.Updates, 
        state: SVGDState, 
        params: optax.Updates
    ) -> optax.GradientTransformation:
        
        count = state.count
        gamma = jax.tree_map(
            lambda _: anneal_fn(count), state.gamma
        )
        updates = jax.tree_map(fsvgd_step, params, updates, gamma)

        return updates, SVGDState(count=count+1, gamma=gamma)
    
    return optax.GradientTransformation(init_fn, update_fn)


def fsvgd_gd(
    epochs: int,
    lr_svgd: Float,
    lr_gd: Float,
    alpha: Float = 0.5,
    K_k_grad: Callable = matrix_rbf_and_grad, 
    c: int = 5,
    p: Float = 0.5,
    ls: ArrayLike | None = None,
):
    
    """Joint SVGD and GD optimization for mixture stein variational gradient descent."""
    svgd_opt = optax.chain(
        fsvgd(epochs, alpha=alpha, K_k_grad=K_k_grad, c=c, p=p, ls=ls),
        optax.scale_by_adam(),
        optax.add_decayed_weights(1e-4, None),
        optax.scale_by_learning_rate(lr_svgd)
    )
    gd_opt = optax.chain(
        optax.adamw(lr_gd),
        optax.scale(-1.0)  # as we take gradient of log-likelihood thus want to max
    )

    opt = optax.multi_transform(
        {"svgd": svgd_opt, "gd": gd_opt},
        ("svgd", "gd")
    )

    return opt


# ------------------------------------- MIXTURE SVGD ------------------------------------- #
def msvgd(
    epochs: int,
    alpha: Float = 0.5,
    K_k_grad: Callable = mmd_k_and_grad, 
    c: int = 5,
    p: Float = 0.5,
    ls: ArrayLike | None = None,
):
    "Mixture stein variational gradient descent with annealing."
    
    @jax.jit
    def msvgd_step(
        particles: Float[Array, "q R d"], 
        particle_grads: Float[Array, "q R d"], 
        _gamma: Float
    ) -> Float[Array, "q R d"]:

        # kernel matrix and gradient of kernel matrix
        K, K_grad = K_k_grad(particles, ls=ls)

        # calculate forces
        pull = _gamma * jnp.einsum("ij,jrd->ird", K, particle_grads)
        repulse = alpha * K_grad
        msrfr_grads = (pull + repulse) / K.shape[0]
        # mar_srfr_grads = (pull + repulse) / K.sum(axis=1, keepdims=True) 

        return -msrfr_grads

    @jax.jit
    def anneal_fn(t: int) -> Float:
        return (jnp.mod(t, epochs/c) / (epochs/c)) ** p
    
    def init_fn(params):
        gamma = jax.tree_map(lambda _: anneal_fn(0), params)
        return SVGDState(count=jnp.zeros([], jnp.int32), gamma=gamma)
    
    def update_fn(
            updates: optax.Updates, 
            state: SVGDState, 
            params: optax.Updates
    ) -> optax.GradientTransformation:
    
        count = state.count
        gamma = jax.tree_map(
            lambda _: anneal_fn(count), state.gamma
        )
        updates = jax.tree_map(msvgd_step, params, updates, gamma)
        return updates, SVGDState(count=count + 1, gamma=gamma)
    
    return optax.GradientTransformation(init_fn, update_fn)


def msvgd_gd(
    epochs: int,
    lr_svgd: Float,
    lr_gd: Float,
    alpha: Float = 0.5,
    K_k_grad: Callable = mmd_k_and_grad, 
    c: int = 5,
    p: Float = 0.5,
    ls: ArrayLike | None = None,
):
    """Joint SVGD and GD optimization for mixture stein variational gradient descent."""
    svgd_opt = optax.chain(
        msvgd(epochs, alpha=alpha, K_k_grad=K_k_grad, c=c, p=p, ls=ls),
        optax.scale_by_adam(),
        optax.add_decayed_weights(1e-4, None),
        optax.scale_by_learning_rate(lr_svgd)
    )
    gd_opt = optax.chain(
        optax.adamw(lr_gd),
        optax.scale(-1.0)  # as we take gradient of log-likelihood thus want to max
    )

    opt = optax.multi_transform(
        {"svgd": svgd_opt, "gd": gd_opt},
        ("svgd", "gd")
    )

    return opt


def msvgd2(
    epochs: int,
    alpha: Float = 0.5,
    K_k_grad: Callable = mmd_k_and_grad, 
    c: int = 5,
    p: Float = 0.5,
    ls: ArrayLike | None = None,
):
    "Mixture stein variational gradient descent with annealing."
    
    @jax.jit
    def msvgd_step(
        particles: Float[Array, "q R d"], 
        particle_grads: Float[Array, "q R d"], 
        _gamma: Float
    ) -> Float[Array, "q R d"]:

        # kernel matrix and gradient of kernel matrix
        K, K_grad = K_k_grad(particles, ls=ls)

        # calculate forces
        pull = _gamma * jnp.einsum("ij,jrd->ird", K, particle_grads)
        repulse = alpha * K_grad
        msrfr_grads = (pull + repulse) / K.shape[0]
        # mar_srfr_grads = (pull + repulse) / K.sum(axis=1, keepdims=True) 

        return -msrfr_grads

    @jax.jit
    def anneal_fn(t: int) -> Float:
        return (jnp.mod(t, epochs/c) / (epochs/c)) ** p
    
    def init_fn(params):
        gamma = jax.tree_map(lambda _: anneal_fn(0), params)
        return SVGDState(count=jnp.zeros([], jnp.int32), gamma=gamma)
    
    def update_fn(
            updates: optax.Updates, 
            state: SVGDState, 
            params: optax.Updates
    ) -> optax.GradientTransformation:
    
        count = state.count
        gamma = jax.tree_map(
            lambda _: anneal_fn(count), state.gamma
        )
        updates = jax.tree_map(msvgd_step, params, updates, gamma)
        return updates, SVGDState(count=count + 1, gamma=gamma)
    
    return optax.GradientTransformation(init_fn, update_fn)


def msvgd_gd2(
    epochs: int,
    lr_svgd: Float,
    lr_gd: Float,
    alpha: Float = 0.5,
    K_k_grad: Callable = mmd_k_and_grad, 
    c: int = 5,
    p: Float = 0.5,
    ls: ArrayLike | None = None,
):
    """Joint SVGD and GD optimization for mixture stein variational gradient descent."""
    svgd_opt = optax.chain(
        msvgd2(epochs, alpha=alpha, K_k_grad=K_k_grad, c=c, p=p, ls=ls),
        optax.scale_by_adam(),
        optax.add_decayed_weights(1e-4, None),
        optax.scale_by_learning_rate(lr_svgd)
    )
    gd_opt = optax.chain(
        optax.adamw(lr_gd),
        optax.scale(-1.0)  # as we take gradient of log-likelihood thus want to max
    )

    opt = optax.multi_transform(
        {"svgd": svgd_opt, "gd": gd_opt},
        ("svgd", "gd")
    )

    return opt
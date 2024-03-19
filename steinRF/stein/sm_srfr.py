# ---------------------------------------------------------------------------------------- #
#                     SPECTRAL MIXTURE STEIN RANDOM FEATURE REGRESSION                     #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
import jaxopt
from functools import partial

from tensorflow_probability.substrates.jax import distributions as tfd
from steinRF.stein.svgd import matrix_rbf_and_grad, matrix_matern32_and_grad, \
    matrix_matern12_and_grad, annealing
from steinRF.utils import convergence, _criteria_fn, _loss_criteria_fn, loss_convergence
from steinRF.gp.training import trainable
from steinRF.stein.targets import NLLTarget

from jaxtyping import Array, Float
from typing import Callable


# ------------------------------------- VANILLA SVGD ------------------------------------- #
@partial(jax.jit, static_argnames=("ls", "alpha", "k_k_grad"))
def _sm_srfr_substep(
        particles: Float[Array, "R d"], 
        particle_grads: Float[Array, "R d"], 
        gamma: Float,
        ls: Float | None = None, 
        alpha: Float =1., 
        k_k_grad: Callable =matrix_rbf_and_grad
    ):

    R, d = particles.shape

    # kernel matrix and gradient of kernel matrix
    K, K_grad = k_k_grad(particles, jnp.eye(d), ls=ls)

    pull = gamma * K @ particle_grads
    # repulse = alpha / R * K_grad.sum(axis=0)
    repulse = alpha * K_grad.sum(axis=0)
    srfr_grads = (pull + repulse) / K.sum(axis=1, keepdims=True)
    # srfr_grads = pull + repulse

    return srfr_grads


@partial(jax.jit, static_argnames=("ls", "alpha", "k_k_grad"))
def sm_srfr_substep(
        particles: Float[Array, "p R d"], 
        particle_grads: Float[Array, "p R d"],
        gamma: Float,
        ls: Float | None = None,
        alpha: Float =1., 
        k_k_grad: Callable = matrix_rbf_and_grad
    ):
    
    particles = jnp.atleast_3d(particles)
    particle_grads = jnp.atleast_3d(particle_grads)

    grads = vmap(
        _sm_srfr_substep, (0, 0, None, None, None, None)
    )(particles, particle_grads, gamma, ls, alpha, k_k_grad)

    return grads


# --------------------------------------- FULL SVGD -------------------------------------- #
def sm_srfr(gp, y, epochs, kernel="rbf", **kwargs):
    ls = kwargs.get("ls", None)
    alpha = kwargs.get("alpha", 1.)  # repulsive force hyperparam
    target = kwargs.get("target", NLLTarget())

    # annealing schedule
    c = kwargs.get("c", 5)
    s = kwargs.get("s", 0.5)
    gamma = kwargs.get('gamma', annealing(epochs, c=c, p=s))

    # optimizer
    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,
    #     peak_value=kwargs.get("lr", 1e-2),
    #     warmup_steps=50,
    #     decay_steps=epochs - int(epochs / 10),
    #     end_value=kwargs.pop("lr_min", 1e-4),
    # )
    # opt= optax.adamw(learning_rate=schedule)
    lr = kwargs.get("lr", 1e-2)
    opt = optax.adamw(lr)
    
    if kernel == "rbf":
        k_k_grad = matrix_rbf_and_grad
    elif kernel == "m12":
        k_k_grad = matrix_matern12_and_grad
    elif kernel == "m32":
        k_k_grad = matrix_matern32_and_grad
    else:
        raise ValueError(f"kernel {kernel} not implemented")

    # make parameter pytrees for SVGD and gradient descent params
    svgd_params_fn = kwargs.get(
        "svgd_params", lambda t: [t.kernel.kernel.u, t.kernel.kernel.l]
    )
    gd_params_fn = kwargs.get("gd_params", None)  # could consider doing w and scale
    if gd_params_fn is None:
        trainable_params_fn = lambda t: svgd_params_fn(t)
    else:
        trainable_params_fn = lambda t: (*svgd_params_fn(t), *gd_params_fn(t))
    
    svgd_params, _ = trainable(gp, svgd_params_fn)
    params, static = trainable(gp, trainable_params_fn)

    ###### define an svgd step
    @eqx.filter_jit
    def sm_srfr_step(params, opt_state, y: Float[Array, "N"], gamma_t: Float):

        # gradient of nll w.r.t particles
        loss, particle_grads_tree = target.grad(params, static, y)
        # return particle_grads_tree

        # calculate velocities based on method
        velocities = jax.tree_map(
            lambda p, p_grads: sm_srfr_substep(
                p, p_grads, gamma_t, ls=ls, alpha=alpha, k_k_grad=k_k_grad
            ),
            svgd_params, particle_grads_tree
        )
        velocities = jax.tree_map(  # reshape to original dims
            lambda v, prm: v.reshape(prm.shape),
            velocities, svgd_params    
        )

        # merge SVGD and regular gradient updates
        grads = jax.tree_map(
            lambda gd_update, svgd_update: svgd_update
                if svgd_update is not None else gd_update,
            particle_grads_tree, velocities
        )
        # grad_info = grads.kernel.kernel.w[1]
        # grads = eqx.tree_at(lambda _t: _t.kernel.kernel.w, grads, replace_fn=lambda _w: _w[0])
        
        # apply updates
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss

    # initalize optimizer
    opt_state = opt.init(eqx.filter(gp, eqx.is_array))
    # return sm_srfr_step(params, opt_state, y)

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 25)

    loss_vals = []
    for epoch in range(epochs):
        params, opt_state, loss = sm_srfr_step(params, opt_state, y, gamma(epoch))
        loss_vals.append(loss)
    
        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch}/{epochs},loss: {loss}")

    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)


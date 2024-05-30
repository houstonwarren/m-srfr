# ---------------------------------------------------------------------------------------- #
#                              STEIN RANDOM FEATURE REGRESSION                             #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import optax
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
from time import time
from typing import Tuple, NamedTuple
from jaxtyping import Float, Array

from steinRF.gp.training import trainable
from steinRF.stein.opt import msvgd, msvgd_gd
from steinRF.stein.kernels import *


# --------------------------------- OPTAX IMPLEMENTATION --------------------------------- #
def msrfr(gp, target, y, epochs, kernel="mmd", **kwargs):
    theta = kwargs.get("theta", {})  # target dist parameters, if any

    #### annealing schedule
    anneal = kwargs.get("anneal", True)
    c = kwargs.get("c", 5)
    s = kwargs.get("s", 0.5) if anneal else 0.

    #### kernel
    if kernel == "mmd":
        k_k_grad = mmd_k_and_grad
    else:
        raise ValueError(f"kernel {kernel} not implemented")
    
    #### make parameter pytrees for SVGD and gradient descent params
    svgd_params_fn = kwargs.get("svgd_params", lambda t: [t.kernel.kernel.w])
    gd_params_fn = kwargs.get("gd_params", None)  # gradient descent params
    if gd_params_fn is None:
        gd_params_fn = lambda t: None
        trainable_params_fn = lambda t: svgd_params_fn(t)
    else:
        trainable_params_fn = lambda t: (*svgd_params_fn(t), *gd_params_fn(t))
    params, static = trainable(gp, trainable_params_fn)
    svgd_params, gd_params = trainable(params, svgd_params_fn)
    all_params = (svgd_params, gd_params)

    #### optimizer - combine an svgd optimizer and gradient descent optimizer
    ls = kwargs.get("ls", 1.)
    alpha = kwargs.get("alpha", 0.5)  # inter-dist repulsive force hyperparam
    lr = kwargs.get("lr", 1e-2)
    lr_gd = kwargs.get("lr_gd", lr)
    opt = msvgd_gd(epochs, lr, lr_gd, alpha, k_k_grad, c, s, ls)

    ###### define an svgd step
    @eqx.filter_jit
    def msrfr_step(
        _all_params: Tuple[eqx.Module, eqx.Module], 
        opt_state: optax.OptState,
    ) -> Tuple[
            Tuple[eqx.Module, eqx.Module], 
            optax.OptState, 
            Float
        ]:
        
        score, grads = target.split_grad(_all_params, static, y, **theta)

        # apply svgd updates
        updates, opt_state = opt.update(grads, opt_state, params=_all_params)
        _all_params = optax.apply_updates(_all_params, updates)
        
        return _all_params, opt_state, -score

    #### initalize optimizer
    opt_state = opt.init(eqx.filter(all_params, eqx.is_array))

    #### run optimization loop
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 25)
    loss_vals = [-target.score(params, static, y, **theta)]
    for epoch in range(epochs):
        all_params, opt_state, loss = msrfr_step(all_params, opt_state)
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")
    
    params = eqx.combine(all_params[0], all_params[1])
    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)


# --------------------------------------- FULL SVGD -------------------------------------- #
# @jax.jit
# def mmd_mar_srfr_step(
#         particles, particle_grads, alpha, gamma, ls=1.
#     ):
        
#     # kernel matrix and gradient of kernel matrix
#     K, K_grad = mmd_k_and_grad(particles, ls=ls)

#     # calculate forces
#     pull = gamma * jnp.einsum("ij,jrd->ird", K, particle_grads)
#     # repulse = alpha * K_grad
#     repulse = alpha * K_grad
#     mar_srfr_grads = pull + repulse 
#     # mar_srfr_grads = (pull + repulse) / K.sum(axis=1, keepdims=True) 

#     return -mar_srfr_grads


# def mar_srfr(gp, target, y, epochs, kernel="mmd", **kwargs):
#     ls = kwargs.get("ls", 1.)
#     alpha = kwargs.get("alpha", 0.5)  # inter-dist repulsive force hyperparam
#     theta = kwargs.get("theta", {})  # target dist parameters, if any

#     # annealing schedule
#     anneal = kwargs.get("anneal", "cyclical")
#     if anneal == "cyclical":
#         c = kwargs.get("c", 5)
#         s = kwargs.get("s", 0.5)
#         gamma = annealing(epochs, c=c, p=s)
#     else:
#         gamma = lambda t: 1.

#     # optimizer
#     lr = kwargs.get("lr", 1e-2)
#     opt = optax.adamw(lr)

#     if kernel == "mmd":
#         k_k_grad = mmd_k_and_grad
#     else:
#         raise ValueError(f"kernel {kernel} not implemented")

#     # make parameter pytrees for SVGD and gradient descent params
#     # for individual component GPs
#     svgd_params_fn = kwargs.get("svgd_params", lambda t: [t.kernel.kernel.w])
#     gd_params_fn = kwargs.get("gd_params", None)  # gradient descent params
#     if gd_params_fn is None:
#         trainable_params_fn = lambda t: svgd_params_fn(t)
#     else:
#         trainable_params_fn = lambda t: (*svgd_params_fn(t), *gd_params_fn(t))

#     svgd_params, _ = trainable(gp, svgd_params_fn)
#     params, static = trainable(gp, trainable_params_fn)

#     ###### define an svgd step
#     @jax.jit
#     def mar_srfr_step(params, opt_state, gamma_t):
#         score, grads = target.grad(params, static, y, **theta)

#         # calculate velocities
#         svgd_grads = jax.tree_map(
#             lambda Ws, grad_Ws: mmd_mar_srfr_step(
#                 Ws, grad_Ws, alpha=alpha, gamma=gamma_t, 
#                 ls=ls,
#             ), svgd_params, grads
#         )

#         # apply updates
#         updates, opt_state = opt.update(svgd_grads, opt_state, params=params)
#         params = optax.apply_updates(params, updates)
        
#         return params, opt_state, -score

#     # initalize optimizer
#     opt_state = opt.init(eqx.filter(gp, eqx.is_array))
#     # return mar_srfr_step(params, opt_state, gamma(0))[0]

#     # loop over epochs
#     verbose = kwargs.get("verbose", False)
#     print_iter = kwargs.get("print_iter", 25)

#     loss_vals = [-target.score(params, static, y, **theta)]
#     for epoch in range(epochs):
#         params, opt_state, loss = mar_srfr_step(params, opt_state, gamma(epoch))
#         loss_vals.append(loss)

#         # # print output
#         if verbose and epoch % print_iter == 0:
#             print(f"epoch {epoch},loss: {loss}")
    
#     model = eqx.combine(params, static)
#     return model, jnp.array(loss_vals) #, jnp.array(grad_infos), history

# ---------------------------------------------------------------------------------------- #
#                              STEIN RANDOM FEATURE REGRESSION                             #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
import jaxopt
from functools import partial
from typing import Tuple, NamedTuple
from jaxtyping import Float, Array

from steinRF.stein.svgd import fisher_info, hessian_approx, gaussian_mixture_softmax
# from steinRF.utils import convergence, _criteria_fn, _loss_criteria_fn, loss_convergence
from steinRF.stein.kernels import *
from steinRF.stein.opt import fsvgd, fsvgd_gd
from steinRF.gp.training import trainable
from steinRF.stein.targets import NLLTarget

# --------------------------------------- FULL SVGD -------------------------------------- #
def srfr(gp, y, epochs, kernel="rbf", **kwargs):
    #### target
    target = NLLTarget()
    
    #### annealing schedule
    anneal = kwargs.get("anneal", True)
    c = kwargs.get("c", 5)
    s = kwargs.get("s", 0.5) if anneal else 0.

    #### kernel
    if kernel == "rbf":
        k_k_grad = matrix_rbf_and_grad
    elif kernel == "m12":
        k_k_grad = matrix_matern12_and_grad
    elif kernel == "m32":
        k_k_grad = matrix_matern32_and_grad
    else:
        raise ValueError(f"kernel {kernel} not implemented")

    #### make parameter pytrees for SVGD and gradient descent params
    svgd_params_fn = kwargs.get("svgd_params", lambda t: [t.kernel.kernel.w])
    gd_params_fn = kwargs.get("gd_params", None)  # gradient descent params
    if gd_params_fn is None:
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
    opt = fsvgd_gd(epochs, lr, lr_gd, alpha, k_k_grad, c, s, ls)

    #### define a optimization loop:
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 25)

    ###### define an svgd step
    @eqx.filter_jit
    def srfr_step(
        _all_params: Tuple[eqx.Module, eqx.Module], 
        opt_state: optax.OptState,
    ) -> Tuple[
            Tuple[eqx.Module, eqx.Module], 
            optax.OptState,
            Float[Array, "R"]
        ]:

        score, grads = target.split_grad(_all_params, static, y)

        # apply svgd updates
        updates, opt_state = opt.update(grads, opt_state, params=_all_params)
        _all_params = optax.apply_updates(_all_params, updates)

        return _all_params, opt_state, -score

    #### initalize optimizer
    opt_state = opt.init(eqx.filter(all_params, eqx.is_array))

    #### run optimization loop
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 25)
    loss_vals = [-target.score(params, static, y)]
    for epoch in range(epochs):
        all_params, opt_state, loss = srfr_step(all_params, opt_state)
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")
    
    params = eqx.combine(all_params[0], all_params[1])
    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)


# ------------------------------------- VANILLA SVGD ------------------------------------- #
# @partial(jax.jit, static_argnames=['k_k_grad'])
# def vanilla_srfr_step(
#         particles, particle_grads, gamma, ls=None, alpha=1., k_k_grad=matrix_rbf_and_grad
#     ):
#     R, d = particles.shape

#     # kernel matrix and gradient of kernel matrix
#     K, K_grad = k_k_grad(particles, jnp.eye(d), ls=ls)

#     pull = gamma * K @ particle_grads
#     # repulse = alpha / R * K_grad.sum(axis=0)
#     repulse = alpha * K_grad.sum(axis=0)
#     srfr_grads = (pull + repulse) / K.sum(axis=1, keepdims=True)
#     # srfr_grads = pull + repulse

#     return -srfr_grads


# # -------------------------------------- MATRIX SVGD ------------------------------------- #
# # code below here adapted from https://github.com/dilinwang820/matrix_svgd
# def matrix_srfr_step(
#         particles, particle_grads, ls=None, alpha=1., k_k_grad=matrix_rbf_and_grad
#     ):

#     H = hessian_approx(particle_grads)

#     K, grad_K = k_k_grad(particles, H, ls=ls)
#     R = K.shape[0]

#     # repulse term
#     # repulse = alpha / R * grad_K.sum(axis=0)
#     repulse = alpha * grad_K.sum(axis=0)
#     # attractive term
#     pull_term = K @ particle_grads
#     srfr_update = (pull_term + repulse) @ jnp.linalg.inv(H) / K.sum(axis=1, keepdims=True)

#     return -srfr_update


# # ------------------------------------- MIXTURE SVGD ------------------------------------- #
# def mixture_srfr_substep(
#         particles, particle_grads, Q, w, w_grad, ls=None, alpha=1., 
#         k_k_grad=matrix_rbf_and_grad
#     ):

#     K, grad_K = k_k_grad(particles, Q, ls=ls)
#     particle_grads += w_grad
#     R = K.shape[0]

#     # repulse term
#     K_repulse = alpha / R * jax.vmap(lambda x, y: x * y)(w, grad_K).sum(axis=0)  # n * d
#     # attractive term
#     K_pull = np.sum(w[None,:,None] * K[:,:,None] * particle_grads[None,:,:], axis = 1)
#     svgd_update = (K_pull + K_repulse) @ jnp.linalg.inv(Q) / K.sum(axis=1, keepdims=True)
#     return svgd_update


# def mixture_srfr_step(
#         particles, particle_grads, ls=None, alpha=1., k_k_grad=matrix_rbf_and_grad
#     ):

#     Qs = fisher_info(particle_grads)
#     w, w_grads = gaussian_mixture_softmax(particles, Qs)

#     # map over everything
#     mixture_grads = jax.vmap(
#             mixture_srfr_substep, 
#             (None, None, 0, 0, 0, None, None, None)
#         )(particles, particle_grads, Qs, w, w_grads, ls, alpha, k_k_grad)
#     mixture_grads = jax.vmap(lambda _w, _grad: _w[:, None] * _grad)(
#         w, mixture_grads
#     ).sum(axis=0)
#     return -mixture_grads


# def srfr(gp, y, epochs, method="srfr", kernel="rbf", **kwargs):
#     ls = kwargs.get("ls", None)
#     alpha = kwargs.get("alpha", 1.)  # repulsive force hyperparam

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

#     # convergence criteria
#     # check_convergence = kwargs.get("check_convergence", False)
#     # param_tol = kwargs.get("param_tol", 1e-5)
#     # loss_tol = kwargs.get("loss_tol", 1e-5)
#     # eta = kwargs.get("eta", 0.1)
#     # patience = kwargs.get("patience", 10)
#     # # criteria_fn = _loss_criteria_fn(patience, eta)
#     # criteria_fn = _criteria_fn(patience, eta)
    
#     if kernel == "rbf":
#         k_k_grad = matrix_rbf_and_grad
#     elif kernel == "m12":
#         k_k_grad = matrix_matern12_and_grad
#     elif kernel == "m32":
#         k_k_grad = matrix_matern32_and_grad
#     else:
#         raise ValueError(f"kernel {kernel} not implemented")

#     if method == "srfr":
#         srfr_step_fn = vanilla_srfr_step

#     elif method == "matrix":
#         srfr_step_fn = matrix_srfr_step

#     elif method == "mixture":
#         srfr_step_fn = mixture_srfr_step

#     # make parameter pytrees for SVGD and gradient descent params
#     svgd_params_fn = kwargs.get("svgd_params", lambda t: [t.kernel.kernel.w])
#     gd_params_fn = kwargs.get("gd_params", None)  # gradient descent params
#     if gd_params_fn is None:
#         trainable_params_fn = lambda t: svgd_params_fn(t)
#     else:
#         trainable_params_fn = lambda t: (*svgd_params_fn(t), *gd_params_fn(t))
    
#     svgd_params, _ = trainable(gp, svgd_params_fn)
#     params, static = trainable(gp, trainable_params_fn)

#     ###### define an svgd step
#     @eqx.filter_jit
#     def srfr_step(params, opt_state, gamma_t):
#         # gradient of nll w.r.t particles
#         @jax.value_and_grad
#         def loss_fn(params):
#             model = eqx.combine(params, static)
#             return model.nll(y) 
#         loss, particle_grads_tree = loss_fn(params)
#         # return particle_grads_tree

#         # calculate velocities based on method
#         velocities = jax.tree_map(
#             lambda p, p_grads: srfr_step_fn(
#                 p, p_grads, gamma_t, ls=ls, alpha=alpha, k_k_grad=k_k_grad
#             ),
#             svgd_params, particle_grads_tree
#         )

#         # merge SVGD and regular gradient updates
#         grads = jax.tree_map(
#             lambda gd_update, svgd_update: svgd_update
#                 if svgd_update is not None else gd_update,
#             particle_grads_tree, velocities
#         )
        
#         # apply updates
#         updates, opt_state = opt.update(grads, opt_state, params=params)
#         params = optax.apply_updates(params, updates)
        
#         return params, opt_state, -loss

#     # initalize optimizer
#     opt_state = opt.init(eqx.filter(gp, eqx.is_array))

#     # loop over epochs
#     verbose = kwargs.get("verbose", False)
#     checkpoints = kwargs.get("checkpoints", 50)
#     print_iter = kwargs.get("print_iter", checkpoints)

#     loss_vals = []
#     checks = []
#     for epoch in range(epochs):
#         params, opt_state, loss = srfr_step(params, opt_state, gamma(epoch))
#         loss_vals.append(loss)
#         # grad_history.append(grads)
#         # if epoch > patience: grad_history.pop(0)
    
#         # # print output
#         if verbose and epoch % print_iter == 0:
#             print(f"epoch {epoch}/{epochs},loss: {loss}")

#         # if check_convergence:
#         #     if epoch > patience and epoch % checkpoints == 0:
#         #         converged = convergence(
#         #             svgd_params_fn, criteria_fn, grad_history[-patience-1:], 
#         #             jnp.array(loss_vals[-patience-1:]), param_tol, loss_tol, eta
#         #         )
#         #         # converged = loss_convergence(criteria_fn, jnp.array(loss_vals), patience, loss_tol)
                
#         #         checks.append(bool(converged))

#         #         if len(checks) > patience:
#         #             checks.pop(0)

#         #         if sum(checks) > 2 * patience // 3:
#         #             print(f"converged at {epoch} iterations.")
#         #             model = eqx.combine(params, static)
#         #             return  model, jnp.array(loss_vals)

#     model = eqx.combine(params, static)
#     return model, jnp.array(loss_vals)


# def srfr_bfgs(gp, y, epochs, method="srfr", kernel="rbf", **kwargs):
#     ls = kwargs.get("ls", None)
#     alpha = kwargs.get("alpha", 1.)  # repulsive force hyperparam
#     solver = kwargs.get("solver", "chol")
    
#     if kernel == "rbf":
#         k_k_grad = matrix_rbf_and_grad
#     elif kernel == "m32":
#         k_k_grad = matrix_matern32_and_grad
#     else:
#         raise ValueError(f"kernel {kernel} not implemented")

#     if method == "srfr":
#         srfr_step_fn = vanilla_srfr_step

#     elif method == "matrix":
#         srfr_step_fn = matrix_srfr_step

#     elif method == "mixture":
#         srfr_step_fn = mixture_srfr_step

#     # make parameter pytrees for SVGD and gradient descent params
#     svgd_params_fn = kwargs.get("svgd_params", lambda t: [t.kernel.kernel.w])
#     gd_params_fn = kwargs.get("gd_params", None)  # gradient descent params
#     if gd_params_fn is None:
#         trainable_params_fn = lambda t: svgd_params_fn(t)
#     else:
#         trainable_params_fn = lambda t: (*svgd_params_fn(t), *gd_params_fn(t))
    
#     svgd_params, _ = trainable(gp, svgd_params_fn)
#     params, static = trainable(gp, trainable_params_fn)

#     ###### define an svgd step
#     @eqx.filter_jit
#     def srfr_step(params):

#         # gradient of nll w.r.t particles
#         @jax.value_and_grad
#         def loss_fn(params):
#             model = eqx.combine(params, static)
#             return -model.nll(y, solver=solver)  # back to positive!
#         loss, particle_grads_tree = loss_fn(params)
#         # return particle_grads_tree

#         # calculate velocities based on method
#         velocities = jax.tree_map(
#             lambda p, p_grads: srfr_step_fn(
#                 p, p_grads, ls=ls, alpha=alpha, k_k_grad=k_k_grad
#             ),
#             svgd_params, particle_grads_tree
#         )

#         # merge SVGD and regular gradient updates
#         grads = jax.tree_map(
#             lambda gd_update, svgd_update: svgd_update
#                 if svgd_update is not None else gd_update,
#             particle_grads_tree, velocities
#         )
        
#         return -loss, grads

#     # initalize optimizer
#     solver = jaxopt.LBFGS(srfr_step, value_and_grad=True, maxiter=epochs, jit=True)

#     # run optimization
#     params, state = solver.run(params)

#     model = eqx.combine(params, static)
#     return model, state

# ---------------------------------------------------------------------------------------- #
#                                       SVGD TRAINING                                      #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import equinox as eqx
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxtyping import Float, Array
from typing import Tuple


from steinRF.stein.kernels import *
from steinRF.stein.opt import asvgd

# -------------------------------------- SVGD UTILS -------------------------------------- #
@jax.jit
def pairwise_distance(X):
    X1X1 = jnp.dot(X.T, X)
    x1_norm = jnp.diag(X1X1)
    dists = jnp.sqrt(jnp.maximum(x1_norm - 2 * X1X1 + x1_norm[:, None], 0))
    return dists


@jax.jit
def median_heuristic(X):
    dists = pairwise_distance(X)
    return jnp.median(dists)


@jax.jit
def fisher_info(grads, normalize=True):
    fisher_matrix = vmap(lambda g: jnp.outer(g, g))(grads)
    if normalize:
        fisher_matrix = (fisher_matrix + fisher_matrix.mean(axis=0)) / 2

    return fisher_matrix


def annealing(epochs: int, c:int=5, p:int=2):
    """
    Anealed SVGD schedule.
    C controls for the number of cylces of the annealing schedule.
    p controls for the power of the annealing schedule.
    """
    def gamma(t: int) -> Float:
        return (jnp.mod(t, epochs/c) / (epochs/c)) ** p
    
    return gamma


# ------------------------------------- VANILLA SVGD ------------------------------------- #
def vanilla_svgd_step(particles, particle_grads, gamma, ls=None):
    R, d = particles.shape

    # kernel matrix and gradient of kernel matrix
    K, K_grad = matrix_rbf_and_grad(particles, jnp.eye(d), ls=ls)

    pull = gamma * K @ particle_grads
    repulse = -1 * K_grad.sum(axis=0)
    svgd_grads = (pull + repulse) / R
    return -svgd_grads


# -------------------------------------- MATRIX SVGD ------------------------------------- #
# code below here adapted from https://github.com/dilinwang820/matrix_svgd
def hessian_approx(grads, eps=1e-3):
    avg_hess = jnp.mean(2 * grads[:, :, None] * grads[:, None, :], axis=0)
    psd_avg_hess = avg_hess + jnp.eye(avg_hess.shape[-1]) * eps
    return psd_avg_hess


def matrix_svgd_step(particles, particle_grads, ls=None):
    H = hessian_approx(particle_grads)

    K, grad_K = matrix_rbf_and_grad(particles, H, ls=ls)
    R = K.shape[0]

    # repulse term
    repulse = -1 * grad_K.sum(axis=0)
    # attractive term
    pull_term = K @ particle_grads
    svgd_update = (pull_term + repulse) / R @ jnp.linalg.inv(H)

    return -svgd_update


# ------------------------------------- MIXTURE SVGD ------------------------------------- #
def gaussian_mixture_softmax(particles, Qs, beta=1.):
    # # make n_particles gaussians
    # mixture = tfd.MultivariateNormalFullCovariance(loc=particles, covariance_matrix=Qs)
    # mixture_px = vmap(lambda x: mixture.prob(x))(particles)
    # w_x = mixture_px / jnp.sum(mixture_px, axis=0)

    # diff = particles[:,None,:] - particles[None,:,:]
    # Hdiff = np.sum(diff[:,:,:,None] * Qs[:,None,:,:], axis = 2)
    # wx_grad = np.sum(
    #     (Hdiff[:,None,:,:] - Hdiff[None,:,:,:]) * mixture_px[None,:,:,None], axis = 1
    # )/ jnp.sum(mixture_px, axis=0)[None,:,None]
    # return w_x, wx_grad

    diff = particles[:,None,:] - particles[None,:,:]
    Hdiff = jnp.sum(diff[:,:,:,None] * Qs[:,None,:,:], axis = 2)
    dist2H = jnp.sum(Hdiff * diff, axis = -1)
    dist2H -= jax.vmap(lambda h: jnp.log(jnp.linalg.det(h)), 0)(Qs)
    dist2H -= jnp.min(dist2H, axis = 0)
    ww = jnp.exp(-0.5* beta * dist2H)
    w = ww / jnp.sum(ww, axis = 0)
    # w = np.eye(n)
    Dlogw = beta * jnp.sum((Hdiff[:,None,:,:] - Hdiff[None,:,:,:]) * ww[None,:,:,None], axis = 1)/jnp.sum(ww, axis = 0)[None,:,None]
    return w, Dlogw


def mixture_svgd_substep(particles, particle_grads, Q, w, w_grad, ls=None):
    K, grad_K = matrix_rbf_and_grad(particles, Q, ls=ls)
    particle_grads += w_grad
    R = K.shape[0]

    # repulse term
    K_repulse = -1 * jax.vmap(lambda x, y: x * y)(w, grad_K).sum(axis=0)  # n * d
    # attractive term
    K_pull = np.sum(w[None,:,None] * K[:,:,None] * particle_grads[None,:,:], axis = 1)
    svgd_update = (K_pull + K_repulse) / R @ jnp.linalg.inv(Q)  # observe negative here
    return -svgd_update


def mixture_svgd_step(particles, particle_grads, ls=None):
    Qs = fisher_info(particle_grads)
    w, w_grads = gaussian_mixture_softmax(particles, Qs)

    # map over everything
    mixture_grads = jax.vmap(mixture_svgd_substep, (None, None, 0, 0, 0, None))(
        particles, particle_grads, Qs, w, w_grads, ls
    )
    mixture_grads = jax.vmap(lambda _w, _grad: _w[:, None] * _grad)(
        w, mixture_grads
    ).sum(axis=0)
    return mixture_grads


# ------------------------------------ FULL SVGD LOOP ------------------------------------ #
def svgd(w, target, epochs, kernel="rbf", **kwargs):
    theta = kwargs.get("theta", {})  # target dist parameters, if any
    grad_fn = jax.value_and_grad(lambda particles: target.score(particles, **theta))

    #### annealing schedule
    anneal = kwargs.get("anneal", True)
    c = kwargs.get("c", 5)
    s = kwargs.get("s", 0.5) if anneal else 0.

    #### kernel
    if kernel == "rbf":
        K_k_grad = matrix_rbf_and_grad
    elif kernel == "m32":
        K_k_grad = matrix_matern32_and_grad
    elif kernel == "m12":
        K_k_grad = matrix_matern12_and_grad
    else:
        raise ValueError(f"kernel {kernel} not implemented")

    #### optimizer
    ls = kwargs.get("ls", 1.)
    lr = kwargs.get("lr", 1e-2)
    opt = asvgd(epochs, lr, K_k_grad, c, s, ls)

    ###### define an svgd step
    @eqx.filter_jit
    def svgd_step(
        _w: Float[Array, "R d"],
        opt_state: optax.OptState,
    ) -> Tuple[
            Float[Array, "R d"], 
            optax.OptState, 
            Float
        ]:
        
        score, grads = grad_fn(_w)

        # apply svgd updates
        updates, opt_state = opt.update(grads, opt_state, params=_w)
        _w = optax.apply_updates(_w, updates)
        
        return _w, opt_state, -score

    # initalize optimizer
    opt_state = opt.init(w)
    # return svgd_step(w, opt_state)

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 25)
    loss_vals = []

    loss_vals = []
    history = [w]
    for epoch in range(epochs):
        w, opt_state, loss = svgd_step(w, opt_state)
        loss_vals.append(loss)
        history.append(w)
        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")
    
    return w, jnp.array(loss_vals) #, jnp.array(history)


# def svgd(target, w, opt, epochs, method="svgd", **kwargs):
#     ls = kwargs.get("ls", None)

#     if method == "svgd":
#         svgd_step_fn = vanilla_svgd_step

#     elif method == "matrix":
#         svgd_step_fn = matrix_svgd_step

#     elif method == "mixture":
#         svgd_step_fn = mixture_svgd_step

#     grad_fn = kwargs.get("grad_fn", target.grad)

#     # annealing schedule
#     c = kwargs.get("c", 5)
#     s = kwargs.get("s", 2)
#     gamma = kwargs.get('gamma', annealing(epochs, c=c, p=s))

#     # define an svgd step
#     @eqx.filter_jit
#     def svgd_step(w, opt_state, gamma_t):
#         # @jax.value_and_grad
#         # def loss_fn(particles):
#         #     return target.log_prob(particles).sum()
#         # loss, w_grads = loss_fn(w)
#         score, w_grads = grad_fn(w)

#         # apply step
#         velocities = svgd_step_fn(w, w_grads, ls=ls, gamma=gamma_t)
#         # return velocities

#         # apply updates
#         updates, opt_state = opt.update(velocities, opt_state)
#         w = optax.apply_updates(w, updates)
        
#         return w, opt_state, -score

#     # initalize optimizer
#     opt_state = opt.init(w)
#     # return svgd_step(w, opt_state)

#     # loop over epochs
#     verbose = kwargs.get("verbose", False)
#     print_iter = kwargs.get("print_iter", 25)
#     loss_vals = []

#     loss_vals = []
#     history = [w]
#     for epoch in range(epochs):
#         w, opt_state, loss = svgd_step(w, opt_state, gamma(epoch))
#         # params, opt_state, loss, a_r = svgd_step(params, opt_state, y)
#         loss_vals.append(loss)
#         history.append(w)
#         # # print output
#         if verbose and epoch % print_iter == 0:
#             print(f"epoch {epoch},loss: {loss}")
    
#     return w, jnp.array(loss_vals) #, jnp.array(history)


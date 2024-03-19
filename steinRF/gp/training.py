# ---------------------------------------------------------------------------------------- #
#                                   GP TRAINING UTILITIES                                  #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import jit, vmap
import jax.tree_util as jtu
import optax
import jaxopt
from tensorflow_probability.substrates.jax import distributions as tfd
from copy import deepcopy

# -------------------------------------- PARAMETERS -------------------------------------- #
def freeze(model, frozen_fn):
    filter_spec = jtu.tree_map(lambda t: eqx.is_array(t), model)
    filter_spec = eqx.tree_at(frozen_fn, filter_spec, replace_fn=lambda _: False)
    return eqx.partition(model, filter_spec)


def trainable(model, trainable_prms):
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(trainable_prms, filter_spec, replace_fn=lambda _: True)
    return eqx.partition(model, filter_spec)


# -------------------------------------- CONVERGENCE ------------------------------------- #
def _loss_criteria_fn(patience, eta=1.):
    @jax.jit
    def criteria_fn(loss, loss_threshold):
        weights = jnp.exp(jnp.linspace(-eta, 0, patience))
        
        def loss_converged(losses, loss_threshold):
            st = (losses[:-1].T * weights).T.sum(axis=0)
            stp = (losses[1:].T * weights).T.sum(axis=0)
            rel_delta = (st - stp) / jnp.abs(st)
            converged = jnp.all(rel_delta < loss_threshold)
            return converged 

        loss_outcome = loss_converged(loss, loss_threshold)
        return loss_outcome
    
    return criteria_fn


def loss_convergence(criteria_fn, losses, loss_tol=1e-5):
    return criteria_fn(losses, loss_tol)


# ---------------------------------- OPTIMIZATION LOOPS ---------------------------------- #
def train_with_restarts(key, model_fn, restarts):
    best_gp = None
    best_loss = [jnp.inf]
    best_model_ind = 0

    for restart in range(restarts):
        key, subkey = jax.random.split(key)
        gp, loss = model_fn(subkey)

        if restart == 0:
            best_loss = deepcopy(loss)
            best_gp = deepcopy(gp)

        if loss[-1] < best_loss[-1]:
            best_loss = deepcopy(loss)
            best_gp = deepcopy(gp)
            best_model_ind = restart

    # print(f"best model found at restart {best_model_ind} with loss {best_loss[-1]}")
    return best_gp, best_loss


def run_optax(y, gp, param_fn, epochs, lr, **kwargs):
    solver = kwargs.get("solver", "chol")
    update_sampler = kwargs.get("update_sampler", False)
    
    # convergence criteria
    check_convergence = kwargs.get("check_convergence", False)
    patience = kwargs.get("patience", 50)
    eta = kwargs.get("eta", 1.)
    loss_tol = kwargs.get("loss_tol", 1e-5)
    criteria_fn = _loss_criteria_fn(patience, eta)

    # define an opt step
    # opt = optax.adam(lr)
    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,
    #     peak_value=lr,
    #     warmup_steps=50,
    #     decay_steps=epochs - int(epochs / 10),
    #     end_value=kwargs.pop("lr_min", 1e-4),
    # )

    opt = optax.adamw(lr)
    params, static = param_fn(gp)

    @eqx.filter_jit
    def opt_step(params, _static, opt_state):
        @jax.value_and_grad
        def loss_fn(params):
            model = eqx.combine(params, _static)
            return model.nll(y, solver=solver)

        loss, grads = loss_fn(params)
        # updates, opt_state = opt.update(grads, opt_state)
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # initalize optimizer
    opt_state = opt.init(params)

    # loop over epochs     
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 50)
    loss_vals = []
    checks = []

    for epoch in range(epochs):
        params, opt_state, loss = opt_step(params, static, opt_state)
        loss_vals.append(loss)

        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")

        # some models need updates each step
        if hasattr(gp.kernel.kernel, "update") and update_sampler:
            model = eqx.combine(params, static)
            updated_k = model.kernel.kernel.update(model.X)
            model = eqx.tree_at(lambda t: t.kernel.kernel, model, updated_k)
            _, static = param_fn(model)
            # return model, jnp.array(loss_vals)

        if check_convergence:
            if epoch > patience:
                converged = loss_convergence(criteria_fn, jnp.array(loss_vals[-patience-1:]), loss_tol)
                checks.append(bool(converged))

            if len(checks) > patience:
                checks.pop(0)

            if sum(checks) >= int(jnp.round(patience * 0.8)):
                print(f"converged at {epoch} iterations.")
                model = eqx.combine(params, static)
                return  model, jnp.array(loss_vals)


    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)


def run_jaxopt(y, gp, param_fn, epochs, **kwargs):
    solver = kwargs.pop("solver", "chol")

    # define an opt step
    params, static = param_fn(gp)

    @jit
    def loss_fn(_params):
        model = eqx.combine(_params, static)
        return model.nll(y, solver=solver)

    # initalize optimizer
    if epochs is None:
        opt = jaxopt.LBFGS(loss_fn, **kwargs)
    else:
        opt = jaxopt.LBFGS(loss_fn, maxiter=epochs, **kwargs)

    # run optimizer
    params, opt_state = opt.run(params)
    model = eqx.combine(params, static)

    return model, opt_state


# ------------------------------------- FIT FUNCTION ------------------------------------- #
def fitgp(gp, y, epochs, to_train=None, opt="adam", **kwargs):
    # model partition function
    if to_train is not None:
        param_fn = lambda t: trainable(t, to_train)
    else:
        param_fn = lambda t: freeze(t, lambda _t: _t.X)

    # define the optimization loop
    if opt == "adam":
        lr = kwargs.pop("lr", 1e-3)

        def opt_fn(_y, _gp, _param_fn, _epochs, **_kwargs):
            return run_optax(_y, _gp, _param_fn, _epochs, lr, **_kwargs)

    elif opt == "lbfgs":
        lr = kwargs.pop("lr", 1e-3)
        opt_fn = run_jaxopt
    else:
        raise ValueError("Unknown optimizer")

    # run optimizer
    model, loss = opt_fn(y, gp, param_fn, epochs, **kwargs)

    return model, loss

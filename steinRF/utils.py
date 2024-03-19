from sklearn.model_selection import KFold
import jax
from jax import jit, vmap
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import itertools
import pandas as pd
from jax.tree_util import tree_flatten
from steinRF.gp.training import trainable
from steinRF import MixGP


# --------------------------------------- TRAINING --------------------------------------- #
@jax.jit
def stabilize(A, eps=1.):
    eigvals = jnp.linalg.eigh(A)[0]
    jitters = jnp.abs(jnp.where(eigvals < 0, eigvals, 0).sum(axis=1))
    A_stable = jnp.eye(A.shape[-1]) * jitters[:, None, None]
    return A + eps * A_stable


# modified from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def stack_history(history, param_fn):
    # if param_fn is not None:
    #     history = [trainable(t, param_fn)[0] for t in history]
    history = [trainable(t, param_fn)[0] for t in history]

    stacked_history = tree_stack(history)
    return stacked_history


# assumes that grads and loss have already been subset to patience window
def _criteria_fn(patience, eta=1.):
    @jax.jit
    def criteria_fn(grads, loss, grad_threshold, loss_threshold):
        weights = jnp.exp(jnp.linspace(-eta, 0, patience))

        def param_converged(grad_history, grad_threshold):
            # gradient norm of each particle
            grad_norms = jnp.linalg.norm(grad_history, axis=-1).mean(axis=-1)  # t * ... * n_particles
            st = (grad_norms[:-1].T * weights).T.sum(axis=0)
            stp = (grad_norms[1:].T * weights).T.sum(axis=0)
            rel_delta = (st - stp) / jnp.abs(st)
            converged = jnp.all(rel_delta < grad_threshold)

            return converged

        def params_converged(param_tree, grad_threshold):
            converged_tree = jax.tree_map(
                lambda x: param_converged(x, grad_threshold), param_tree
            )
            return converged_tree
        
        def loss_converged(losses, loss_threshold):
            st = (losses[:-1].T * weights).T.sum(axis=0)
            stp = (losses[1:].T * weights).T.sum(axis=0)
            rel_delta = (st - stp) / jnp.abs(st)
            converged = jnp.all(rel_delta < loss_threshold)
            return converged 

        params_outcome = tree_flatten(params_converged(grads, grad_threshold))[0]
        loss_outcome = loss_converged(loss, loss_threshold)

        return jnp.all(jnp.array([*params_outcome, loss_outcome]))
    
    return criteria_fn


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



def convergence(param_fn, criteria_fn, grads, losses, param_tol=1e-5, loss_tol=1e-5, eta=1.):
    # criteria_fn = _criteria_fn(patience, eta=eta)
    # grads = grads[-patience-1:]
    grad_tree = stack_history(grads, param_fn)
    # losses = losses[-patience-1:]

    return criteria_fn(grad_tree, losses, param_tol, loss_tol)
    
    # # loss_val = jnp.max(jnp.abs(losses))
    # loss_val = jnp.abs(jnp.min(losses))
    # grad_threshold = param_tol * loss_val
    # loss_threshold = loss_tol * loss_val
    # print(loss_val, grad_threshold, loss_threshold)

    # converged = criteria_fn(grad_tree, losses, grad_threshold, loss_threshold)
    # return converged


def loss_convergence(criteria_fn, losses, patience, loss_tol=1e-5):
    losses = losses[-patience-1:]
    return criteria_fn(losses, loss_tol)


# -------------------------------------- EVALUATION -------------------------------------- #
def rescale(scaler, y):
    return jnp.array(
        scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
    )


def mse(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_true = rescale(scaler, y_true)
        y_pred = rescale(scaler, y_pred)
    
    return jnp.mean((y_true - y_pred)**2)


def mae(y_true, y_pred, scaler=None):
    if scaler is not None:
        y_true = rescale(scaler, y_true)
        y_pred = rescale(scaler, y_pred)

    return jnp.mean(jnp.abs(y_true - y_pred))


def calibration(means, sd, y, scaler=None):
    ub = means + 1.95 * sd
    lb = means - 1.95 * sd
    if scaler is not None:
        y = rescale(scaler, y)
        lb = rescale(scaler, lb)
        ub = rescale(scaler, ub)

    return np.mean((y >= lb) & (y <= ub))


def mean_zscore(means, sd, y, scaler=None):
    if scaler is not None:
        y = rescale(scaler, y)
        means = rescale(scaler, means)
        sd = rescale(scaler, sd)

    return np.mean((y - means) / sd)


def metric_model(y, y_pred, y_sd, scaler=None):
    mod_mse = mse(y, y_pred, scaler)
    mod_mae = mae(y, y_pred, scaler)
    mod_cal = calibration(y_pred, y_sd, y, scaler)
    mod_z = mean_zscore(y_pred, y_sd, y, scaler)
    return jnp.array([mod_mse, mod_mae, mod_cal, mod_z])


def gp_cross_val(model_fn, key, X, y, params, n_folds=5, metric=mse, shuffle=True):
    seed = int(key[0])
    
    if shuffle:
        kf = KFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
    else:
        kf = KFold(n_splits=n_folds, shuffle=shuffle)
    
    accuracies = []

    for train_index, test_index in kf.split(X):
        key, subkey = jax.random.split(key)
        X_train, X_test = jnp.array(X[train_index]), jnp.array(X[test_index])
        y_train, y_test = jnp.array(y[train_index]), jnp.array(y[test_index])

        gp, _ = model_fn(subkey, X_train, y_train, **params)
        if metric == "nll":
            gp = eqx.tree_at(lambda t: t.X, gp, replace_fn=lambda _: X_test)
            accuracies.append(gp.nll(y_test))
        elif metric == "mae":
            gp_preds = gp.condition(y_train, X_test)[0]
            accuracies.append(mae(y_test, gp_preds))
        elif isinstance(gp, MixGP):
            gp_preds = gp.condition(subkey, y_train, X_test)[0]
            accuracies.append(metric(y_test, gp_preds))
        else:
            gp_preds = gp.condition(y_train, X_test)[0]
            accuracies.append(metric(y_test, gp_preds))

    return np.mean(accuracies)


def run_hyperopt(cross_val_func, key, X, y, n_trials, study=None, **model_params):
    trial_fn = lambda trial: cross_val_func(trial, key, X, y, **model_params)
    if study is None:
        study = optuna.create_study(direction="minimize")
    study.optimize(trial_fn, n_trials=n_trials, show_progress_bar=True)

    return study


def generate_param_combos(hyperparams):
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    param_combinations = []
    
    for combination in itertools.product(*values):
        param_combinations.append(dict(zip(keys, combination)))
    
    return param_combinations


def grid_search(model_fn, key, X, y, h_params, static={}, metric=mse):
    hparam_combos = generate_param_combos(h_params)

    accs = []
    for i, combo in enumerate(hparam_combos):
        combo = {**combo, **static}
        acc = gp_cross_val(model_fn, key, X, y, combo, metric=metric)
        combo["metric"] = acc
        print(f"trial {i+1}/{len(hparam_combos)}: {combo}")
        accs.append(combo)

    res = pd.DataFrame(accs).sort_values("metric", ascending=True).reset_index(drop=True)
    best = res.drop("metric", axis=1).iloc[0].to_dict()

    return best, res


# --------------------------------------- PLOTTING --------------------------------------- #
def animate_particles(particle_history, target=None, n_steps=None):
    fig, ax = plt.subplots()
    if target is not None:
        x = jnp.linspace(jnp.min(particle_history[:, :, 0]), jnp.max(particle_history[:, :, 0]), 100)
        y = jnp.linspace(jnp.min(particle_history[:, :, 1]), jnp.max(particle_history[:, :, 1]), 100)
        X, Y = jnp.meshgrid(x, y)
        Z = jnp.exp(target.log_prob(jnp.stack([X, Y], axis=-1)).reshape(X.shape))
        contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')

    # Create initial scatter plot
    scatter = ax.scatter(particle_history[0, :, 0], particle_history[0, :, 1])

    # Function to update the scatter plot for each frame
    def update(frame_number):
        scatter.set_offsets(particle_history[frame_number])

    if n_steps is None:
        frames=range(particle_history.shape[0])
    else:
        frames=range(0, particle_history.shape[0], particle_history.shape[0] // n_steps)

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.show()
    return ani


def animate_particle_sets(particle_history, n=3, target=None, n_steps=None):
    fig, ax = plt.subplots()
    if target is not None:
        x = jnp.linspace(jnp.min(particle_history[:, :, 0]), jnp.max(particle_history[:, :, 0]), 100)
        y = jnp.linspace(jnp.min(particle_history[:, :, 1]), jnp.max(particle_history[:, :, 1]), 100)
        X, Y = jnp.meshgrid(x, y)
        Z = jnp.exp(target.log_prob(jnp.stack([X, Y], axis=-1)).reshape(X.shape))
        contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis')

    # Create initial scatter plot
    scatters = [
        ax.scatter(particle_history[0, i, :, 0], particle_history[0, i, :, 1], label="i")
        for i in range(n)
    ]

    # Function to update the scatter plot for each frame
    def update(frame_number):
        for i, scatter in enumerate(scatters):
            scatter.set_offsets(particle_history[frame_number, i, :, :])

    if n_steps is None:
        frames=range(particle_history.shape[0])
    else:
        frames=range(0, particle_history.shape[0], particle_history.shape[0] // n_steps)

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.show()
    return ani

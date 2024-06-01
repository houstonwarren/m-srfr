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
import jax.random as jr
from copy import deepcopy
from tensorflow_probability.substrates.jax import distributions as tfd

from steinRF.gp.training import trainable
from steinRF import MixGP
from steinRF.gp.models import build_train_rff, build_train_mix_rff


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


def train_test_split(key, X, y, test_size=0.2):
    n = X.shape[0]
    if isinstance(test_size, float):
        n_test = int(n * test_size)
    else:
        n_test = test_size
    
    key, subkey = jax.random.split(key)
    indices = jax.random.permutation(subkey, jnp.arange(n))
    # return indices
    X_train, y_train = X[indices[n_test:]], y[indices[n_test:]]
    X_test, y_test = X[indices[:n_test]], y[indices[:n_test]]

    return X_train, X_test, y_train, y_test


def k_fold(key, X, y, n_folds=5):
    inds = jr.permutation(key, jnp.arange(X.shape[0]))
    fold_size = X.shape[0] // n_folds
    for i in range(n_folds):
        test_inds = inds[i*fold_size:(i+1)*fold_size]
        train_inds = jnp.concatenate([inds[:i*fold_size], inds[(i+1)*fold_size:]])
        yield X[train_inds], X[test_inds], y[train_inds], y[test_inds]


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


def gp_cross_val(model_fn, key, X, y, params, n_folds=5, metric=mse):
    accuracies = []
    kf = k_fold(key, X, y, n_folds=n_folds)

    for X_train, X_test, y_train, y_test in kf:
        key, subkey = jax.random.split(key)

        gp, _ = model_fn(subkey, X_train, y_train, **params)
        if metric == "nll":
            gp = eqx.tree_at(lambda t: t.X, gp, replace_fn=lambda _: X_test)
            accuracies.append(gp.nll(y_test))
        elif metric == "mae":
            gp_preds = gp.condition(y_train, X_test)[0]
            accuracies.append(mae(y_test, gp_preds))
        else:
            gp_preds = gp.condition(y_train, X_test)[0]
            accuracies.append(metric(y_test, gp_preds))

    return jnp.mean(jnp.asarray(accuracies))


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


# --------------------------- VISUALIZING MIXTURE DISTRIBUTIONS -------------------------- #
def mvn_to_normal(mvn, inds=None):
    if inds is not None:
        return tfd.Normal(mvn.mean()[inds], mvn.stddev()[inds])
    return tfd.Normal(mvn.mean(), mvn.stddev())


def mvn_gmm_to_normal(mvn_gmm, inds=None):
    if inds is not None:
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mvn_gmm.mixture_distribution.probs),
            components_distribution=tfd.Normal(
                loc=mvn_gmm.components_distribution.mean()[..., inds],
                scale=mvn_gmm.components_distribution.stddev()[..., inds]
            )
        )
    else:
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=mvn_gmm.mixture_distribution.probs),
            components_distribution=tfd.Normal(
                loc=mvn_gmm.components_distribution.mean(),
                scale=mvn_gmm.components_distribution.stddev()
            )
        )


def mixture_pred_dists(key, X, y, params, R=100, test_size=0.2):
    X_tr, X_test, y_tr, y_test = train_test_split(key, X, y, test_size=test_size)

    # train models    
    params = deepcopy(params)
    rff_gp, _ = build_train_rff(key, X_tr, y_tr, R=R, restarts=1, **params["rff"])
    mix_gp, _ = build_train_mix_rff(key, X_tr, y_tr, R=R, restarts=1, **params["mix_rff"])

    # get predictive distributions
    rff_mu, rff_sigma = rff_gp.condition(y_tr, X_test)
    single_dist = tfd.MultivariateNormalDiag(rff_mu, rff_sigma)
    mixture_dist = mix_gp.mixture_dist(y_tr, X_test)
    return rff_gp, single_dist, mix_gp, mixture_dist, y_test


def plot_kernel_mixture(w_mix, plots_per_row, bw_adjust=1., fontsize=20, title=None):
    m, R, d = w_mix.shape

    if d % plots_per_row == 0:
        n_row = d // plots_per_row
    else:
        n_row = d // plots_per_row + 1

    width, height = 5*plots_per_row, 5*n_row
    fig, axs = plt.subplots(n_row, plots_per_row, figsize=(width, height))
    hue_mat = jnp.repeat(jnp.arange(m), R)

    for i, ax in enumerate(axs.flatten()):
        if i < d:
            # plot mixture components
            w_d = w_mix[..., i]
            sns.kdeplot(
                x=w_d.reshape(-1), ax=ax, palette="viridis", hue=hue_mat, bw_adjust=bw_adjust
            )
            ax.legend_.remove()
            ax.set_title(f"$d = {i}$", fontsize=fontsize)
            ax.set_xlabel("")
            ax.set_ylabel("")

        else:
            ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=fontsize * 1.25)

    fig.text(0.5, 0.04, 'RFF Frequency $\omega$', ha='center', va='center', fontsize=fontsize)
    fig.text(0.1, 0.5, '$p(\omega)$', ha='center', va='center', rotation='vertical', fontsize=fontsize)

    plt.show()
    return fig


def plot_mixture_preds(
        p_rff, p_mix, y, n=None, inds=None, 
        plots_per_row=5, key=jax.random.PRNGKey(0), 
        standardize=True, bounds=None, title=None,
        fontsize=20
    ):
    
    # get plotting bounds
    p_mix_m = p_mix.components_distribution
    if bounds is None:
        p_rff_bounds = jnp.array([p_rff.mean() - 3 * p_rff.stddev(), p_rff.mean() + 3 * p_rff.stddev()]).T
        p_mix_bounds = jnp.array([
            p_mix_m.mean() - 3 * p_mix_m.stddev(),
            p_mix_m.mean() + 3 * p_mix_m.stddev()
        ]).T
        y_bounds = jnp.array([y, y]).T
        all_bounds = jnp.concatenate([
            y_bounds[:, None, :], p_rff_bounds[:, None, :], p_mix_bounds], axis=1
        )
        bounds = jnp.stack([all_bounds.min(axis=(1, 2)), all_bounds.max(axis=(1, 2))], axis=-1)
    else:
        bounds = jnp.tile(bounds, (y.shape[0], 1))

    # get indices
    if n is None and inds is None:
        n = 5
    if inds is None:
        inds = jax.random.choice(key, jnp.arange(y.shape[0]), shape=(n,), replace=False)
    
    # subset distributions and data
    y = y[inds]
    bounds = bounds[inds]
    x_vals = jnp.linspace(bounds[:, 0], bounds[:, 1], 1000)
    p_x_rff = tfd.Normal(p_rff.mean()[inds], p_rff.stddev()[inds]).prob(x_vals).T
    p_x_mix_m = jax.vmap(
        tfd.Normal(p_mix_m.mean()[:, inds], p_mix_m.stddev()[:, inds]).prob
    )(x_vals).T
    p_x_mix = tfd.Normal(p_mix.mean()[inds], p_mix.stddev()[inds]).prob(x_vals).T
    # p_x_mix = p_x_mix_m.mean(axis=1)
    
    m = p_x_mix_m.shape[1]

    # make subplots
    if n % plots_per_row == 0:
        n_row = n // plots_per_row
    else:
        n_row = n // plots_per_row + 1

    width, height = 5*plots_per_row, 5*n_row
    fig, axs = plt.subplots(n_row, plots_per_row, figsize=(width, height))

    for i, ax in enumerate(axs.flatten()):
        y_val = y[i]
        x_val = x_vals[:, i]
        Px_rff = p_x_rff[i]
        Px_mix_m = p_x_mix_m[i] / m
        Px_mix = p_x_mix[i]
        if standardize:
            Px_rff /= Px_rff.max()
            Px_mix_m /= Px_mix_m.max(axis=1, keepdims=True)
            Px_mix /= Px_mix.max()

        sns.lineplot(x=x_val, y=Px_rff, label="SSGP $p(y)$", color="blue", ax=ax)
        for px_m in Px_mix_m:
            sns.lineplot(x=x_val, y=px_m, color="red", alpha=0.4, ax=ax)
        sns.lineplot(x=x_val, y=Px_mix, label="M-SRFR $p(y)$", color="red", linestyle="--", ax=ax)
        vline_height = jnp.max(jnp.concatenate([Px_rff, Px_mix]))
        ax.vlines(y_val, 0, vline_height, linestyle="--", label="True Value", color="black")
        if i == 0:
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05), fontsize=fontsize)
        else:
            ax.legend().remove()

    if title is not None:
        fig.suptitle(title, fontsize=fontsize * 1.25)

    fig.text(0.5, 0.08, '$y$', ha='center', va='center', fontsize=fontsize)
    fig.text(
        0.1, 0.5, 'Model Predictive $p(y | \\mathbf{X})$', 
        ha='center', va='center', rotation='vertical', fontsize=fontsize
    )

    plt.show()
    return fig
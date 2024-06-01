# ---------------------------------------------------------------------------------------- #
#                                MODEL CONSTRUCTOR UTILTIES                                #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tensorflow_probability.substrates.jax import distributions as tfd

from steinRF import LowRankGP, MixGP
from steinRF.gp.kernels import RFF, MixRFF
from steinRF.gp.transforms import ARD, MLP, Transform
from steinRF.gp.training import fitgp, train_with_restarts

from steinRF.stein.targets import NLLTarget, PriorNLLTarget, TFTarget
from steinRF.stein.srfr import srfr
from steinRF.stein.msrfr import msrfr
from steinRF.stein.kernels import pairwise_median


__all__ = [
    "build_train_rff",
    'build_rff',
    'build_deep_rff',
    'build_train_deep_rff',
    'build_rff_rbf',
    "build_train_rff_rbf", 
    'build_srf',
    "build_train_srf",
    'build_train_mix_rff',
    'build_mix_rff',
    'build_deep_mix_rff',
    'build_train_deep_mix_rff'
]

# ---------------------------------------- RFF RBF --------------------------------------- #
def build_rff_rbf(key, X_tr, R, diag, mean=None, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        ls_init = pairwise_median(X_tr, X_tr)
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = Transform(ARD(ls_init), RFF(key, d=d, R=R))
    gp_pre = LowRankGP(k, X_tr, diag=diag, mean=mean)
    
    return gp_pre


def build_train_rff_rbf(key, X_tr, y_tr, R, diag, epochs, lr, **kwargs):
    mean = kwargs.pop("mean", None)
    init_ls = kwargs.pop("init_ls", True)
    
    # trainable parameters
    to_train = kwargs.pop("to_train", lambda t: [t.kernel.transform.scale])

    def _train(_key):
        # Initialize model with current hyperparameters
        gp_pre = build_rff_rbf(_key, X_tr, R, diag, mean, init_ls)

        # Train the model
        gp, gp_losses = fitgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses
    
    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# --------------------------------------- BASIC RFF -------------------------------------- #
def build_rff(key, X_tr, R, diag, mean=None, w_init=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        ls_init = pairwise_median(X_tr, X_tr)
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = RFF(key, d=d, R=R)
    if from_data:
        k = k.initialize_from_data(key, R, X_tr)
    k = Transform(ARD(ls_init), k)
    
    # kernel and gp initialization
    if w_init is not None: 
        k = eqx.tree_at(lambda t: t.kernel.kernel.w, k, w_init)
    gp_pre = LowRankGP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_rff(key, X_tr, y_tr, R, diag, epochs, lr, **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: [t.kernel.kernel.w, t.kernel.transform.scale]
    )
    w_init = kwargs.pop("w_init", None)
    mean = kwargs.pop("mean", None)
    from_data = kwargs.pop("from_data", True)
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_rff(
            subkey, X_tr, R, diag, mean=mean, w_init=w_init, from_data=from_data, init_ls=init_ls
        )

        # Train the model
        gp, gp_losses = fitgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# --------------------------------------- DEEP RFF --------------------------------------- #
def build_deep_rff(key, X_tr, out_dim, R, diag, mean=None):
    # Initialize model with current hyperparameters
    d = X_tr.shape[-1]
    k = RFF(key, d=out_dim, R=R)
    mlp = MLP(key, in_dim=d, out_dim=out_dim, d_hidden=32, n_hidden=3)
    k = Transform(mlp, k)

    # init gp
    gp_pre = LowRankGP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_deep_rff(key, X_tr, y_tr, out_dim, R, diag, epochs, lr, **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: [
            t.kernel.kernel.w, t.kernel.transform.layers, t.kernel.transform.scale
        ]
    )
    mean = kwargs.pop("mean", None)

    def _train(subkey):
        gp_pre = build_deep_rff(
            subkey, X_tr, out_dim, R, diag, mean=mean,
        )

        # Train the model
        gp, gp_losses = fitgp(
            gp_pre, y_tr, epochs, 
            to_train=to_train, lr=lr, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# --------------------------------- STEIN RANDOM FEATURES -------------------------------- #
def build_srf(key, X_tr, R, diag, mean=None, w_init=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        ls_init = pairwise_median(X_tr, X_tr)
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = RFF(key, d=d, R=R)
    if from_data:
        k = k.initialize_from_data(key, R, X_tr)
    k = Transform(ARD(ls_init), k)
    if w_init is not None: 
        k = eqx.tree_at(lambda t: t.kernel.kernel.w, k, w_init)
    gp_pre = LowRankGP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_srf(key, X_tr, y_tr, R, diag, epochs, lr, alpha, **kwargs):
    # extract kwargs
    # lr_min = kwargs.pop("lr_min", 1e-4)
    w_init = kwargs.pop("w_init", None)
    mean = kwargs.pop("mean", None)
    from_data = kwargs.pop("from_data", True)
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_srf(
            subkey, X_tr, R, diag, mean=mean, w_init=w_init, from_data=from_data, init_ls=init_ls
        )

        # opt = optax.adam(lr)
        gp, gp_losses = srfr(
            gp_pre, y_tr, epochs=epochs, method="srfr",
            alpha=alpha, lr=lr, **kwargs
        )
    
        return gp, gp_losses
    
    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# -------------------------------------- MIXTURE GP -------------------------------------- #
def build_mix_rff(subkey, X_tr, diag, q, R, mean=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        ls_init = pairwise_median(X_tr, X_tr)
    else:
        ls_init = jnp.ones(d)

    mixrff = MixRFF(subkey, q=q, R=R, d=d)
    if from_data:
        mixrff = mixrff.initialize_from_data(subkey, q=q, R=R, X=X_tr)
    mixrff = Transform(ARD(ls_init), mixrff)

    gp_pre = MixGP(mixrff, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_mix_rff(key, X_tr, y_tr, diag, q, R, alpha, epochs, lr, **kwargs):
    mean = kwargs.pop("mean", None)
    prior = kwargs.pop("prior", None)
    from_data = kwargs.pop("from_data", True)
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_mix_rff(
            subkey, X_tr, diag, q, R, mean, from_data, init_ls=init_ls
        )

        # prior
        if prior is None:
            _prior = TFTarget(tfd.Normal(
                gp_pre.kernel.kernel.w.mean(axis=(0,1)), 
                gp_pre.kernel.kernel.w.std(axis=(0,1))**2
            ))
            target = PriorNLLTarget(prior=_prior)
        elif prior is False:
            target = NLLTarget()
        else:
            target = PriorNLLTarget(prior=prior)

        # Train the model
        gp, gp_losses = msrfr(
            gp_pre, target, y_tr, lr=lr, epochs=epochs, alpha=alpha, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# ------------------------------------ DEEP MIXTURE GP ----------------------------------- #
def build_deep_mix_rff(subkey, X_tr, diag, q, out_dim, R, mean=None):
    d = X_tr.shape[-1]
    mixrff = MixRFF(subkey, q=q, R=R, d=out_dim)
    mlp = MLP(subkey, in_dim=d, out_dim=out_dim, d_hidden=32, n_hidden=3)
    mixrff = Transform(mlp, mixrff)

    gp_pre = MixGP(mixrff, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_deep_mix_rff(key, X_tr, y_tr, diag, q, out_dim, R, alpha, epochs, lr, **kwargs):
    mean = kwargs.pop("mean", None)
    prior = kwargs.pop("prior", None)
    gd_params = lambda t: [
        t.kernel.transform.layers, t.kernel.transform.scale
    ]

    def _train(subkey):
        gp_pre = build_deep_mix_rff(
            subkey, X_tr, diag, q, out_dim, R, mean,
        )

        # prior
        if prior is None:
            _prior = TFTarget(tfd.Normal(
                gp_pre.kernel.kernel.w.mean(axis=(0,1)), 
                gp_pre.kernel.kernel.w.std(axis=(0,1))**2
            ))
            target = PriorNLLTarget(prior=_prior)
        elif prior is False:
            target = NLLTarget()
        else:
            target = PriorNLLTarget(prior=prior)

        # Train the model
        gp, gp_losses = msrfr(
            gp_pre, target, y_tr, lr=lr, epochs=epochs, alpha=alpha,
            gd_params=gd_params, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss

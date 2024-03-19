# ---------------------------------------------------------------------------------------- #
#                                MODEL CONSTRUCTOR UTILTIES                                #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from tinygp import GaussianProcess
from tensorflow_probability.substrates.jax import distributions as tfd

from steinRF import GP, LowRankGP, MixGP
from steinRF.gp.kernels import RFF, NonstationaryRFF, SMK, SparseSMK, MixRFF, NMixRFF
from steinRF.gp.transforms import ARD, Transform
from steinRF.gp.training import fitgp, train_with_restarts

from steinRF.stein.targets import NLLTarget, PriorNLLTarget, TFTarget
from steinRF.stein.srfr import srfr
from steinRF.stein.sm_srfr import sm_srfr
from steinRF.stein.mar_srfr import mar_srfr


__all__ = [
    "build_train_rff",
    'build_rff',
    'build_train_nrff',
    'build_rff_rbf',
    "build_train_rff_rbf", 
    'build_srf',
    "build_train_srf",
    'build_train_nsrf',
    "build_train_smk",
    "build_train_ssmk",
    "build_train_srf_smk",
    "build_srf_smk",
    'build_train_mix_rff',
    'build_mix_rff',
    'build_train_nmix_rff',
    'build_nmix_rff'
]

# ---------------------------------------- RFF RBF --------------------------------------- #
def build_rff_rbf(key, X_tr, R, diag, mean=None, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
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
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
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


# ----------------------------------- NONSTATIONARY RFF ---------------------------------- #
def build_nrff(key, X_tr, R, diag, mean=None, w_init=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = NonstationaryRFF(key, d=d, R=R)
    if from_data:
        k = k.initialize_from_data(key, R, X_tr)
    k = Transform(ARD(ls_init), k)
    
    # kernel and gp initialization
    if w_init is not None: 
        k = eqx.tree_at(lambda t: t.kernel.kernel.w, k, w_init)
    gp_pre = LowRankGP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_nrff(key, X_tr, y_tr, R, diag, epochs, lr, **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: [t.kernel.kernel.w, t.kernel.transform.scale]
    )
    w_init = kwargs.pop("w_init", None)
    mean = kwargs.pop("mean", None)
    from_data = kwargs.pop("from_data", True)
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_nrff(
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


# --------------------------------- STEIN RANDOM FEATURES -------------------------------- #
def build_srf(key, X_tr, R, diag, mean=None, w_init=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
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


# -------------------------- NONSTATIONARY STEIN RANDOM FEATURES ------------------------- #
def build_nsrf(key, X_tr, R, diag, mean=None, w_init=None, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = NonstationaryRFF(key, d=d, R=R)
    k = k.initialize_from_data(key, R, X_tr)
    k = Transform(ARD(ls_init), k)
    if w_init is not None: 
        k = eqx.tree_at(lambda t: t.kernel.kernel.w, k, w_init)
    gp_pre = LowRankGP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_nsrf(key, X_tr, y_tr, R, diag, epochs, lr, alpha, **kwargs):
    # extract kwargs
    # lr_min = kwargs.pop("lr_min", 1e-4)
    w_init = kwargs.pop("w_init", None)
    mean = kwargs.pop("mean", None)
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_nsrf(
            subkey, X_tr, R, diag, mean=mean, w_init=w_init, init_ls=init_ls
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


# ------------------------------------------ SMK ----------------------------------------- #
def build_smk(key, X_tr, y_tr, diag, q, mean=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    dX = X_tr[:, None, :] - X_tr[None, :, :]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = SMK(m=q, d=d)
    if from_data:
        k = k.initialize_from_data(key, X_tr, y_tr)
    k = Transform(ARD(ls_init), k)
    
    # kernel and gp initialization
    gp_pre = GP(k, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_smk(key, X_tr, y_tr, diag, m, epochs, lr, **kwargs):
    # extract kwargs
    to_train = kwargs.pop(
        "to_train", lambda t: [
            t.kernel.kernel.w, t.kernel.kernel.u, t.kernel.kernel.l, 
            t.kernel.transform.scale
        ]
    )
    mean = kwargs.pop("mean", None)
    from_data = kwargs.pop("from_data", True)
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_smk(
            subkey, X_tr, y_tr, diag, m, mean=mean, 
            from_data=from_data, init_ls=init_ls
        )

        # Train the model
        gp, gp_losses = fitgp(
            gp_pre, y_tr, epochs=epochs, lr=lr, to_train=to_train, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# -------------------------------------- SPARSE SMK -------------------------------------- #
def build_train_ssmk(key, X_tr, y_tr, diag, m, p, R, epochs, lr, **kwargs):
    # extract kwargs
    d = X_tr.shape[-1]
    to_train = kwargs.pop(
        "to_train", lambda t: [
            t.kernel.kernel.w, t.kernel.kernel.u, t.kernel.kernel.l, 
            t.kernel.transform.scale
        ]
    )
    mean = kwargs.pop("mean", None)

    def _train(subkey):
        # Initialize model with current hyperparameters
        k = SparseSMK(p=p, m=m, d=d, R=R, key=subkey)
        k = k.initialize_from_data(subkey, X=X_tr, y=y_tr)
        k = Transform(ARD(jnp.ones(d)), k)
        
        # kernel and gp initialization
        gp_pre = MixGP(k, X_tr, diag=diag, mean=mean)

        # Train the model
        gp, gp_losses = fitgp(
            gp_pre, y_tr, epochs=epochs, lr=lr, to_train=to_train, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# ----------------------------- DIVERSIFIED SPECTRAL MIXTURE ----------------------------- #
def build_srf_smk(subkey, X_tr, y_tr, diag, m, p, R, init_ls=True):
    # extract kwargs
    d = X_tr.shape[-1]
    dX = X_tr[:, None, :] - X_tr[None, :, :]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    # Initialize model with current hyperparameters
    k = SparseSMK(p=p, m=m, d=d, R=R, key=subkey)
    k = k.initialize_from_data(subkey, X=X_tr, y=y_tr)
    k = Transform(ARD(jnp.ones(ls_init)), k)
    
    # kernel and gp initialization
    gp_pre = MixGP(k, X_tr, diag=diag)
    return gp_pre


def build_train_srf_smk(key, X_tr, y_tr, diag, m, p, R, alpha, epochs, lr, **kwargs):
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_srf_smk(subkey, X_tr, y_tr, diag, m, p, R, init_ls=init_ls,)
        
        # Train the model
        gp, gp_losses = sm_srfr(
            gp_pre, y_tr, epochs=epochs, lr=lr, alpha=alpha, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# -------------------------------------- MIXTURE GP -------------------------------------- #
def build_mix_rff(subkey, X_tr, diag, q, R, mean=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
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
        gp, gp_losses = mar_srfr(
            gp_pre, target, y_tr, lr=lr, epochs=epochs, alpha=alpha, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss


# --------------------------------- NONSTATIONARY MIXTURE -------------------------------- #
def build_nmix_rff(subkey, X_tr, diag, q, R, mean=None, from_data=False, init_ls=True):
    d = X_tr.shape[-1]
    if init_ls:
        dX = X_tr[:, None, :] - X_tr[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
    else:
        ls_init = jnp.ones(d)

    mixrff = NMixRFF(subkey, q=q, R=R, d=d)
    if from_data:
        mixrff = mixrff.initialize_from_data(subkey, q=q, R=R, X=X_tr)
    mixrff = Transform(ARD(ls_init), mixrff)

    gp_pre = MixGP(mixrff, X_tr, diag=diag, mean=mean)
    return gp_pre


def build_train_nmix_rff(key, X_tr, y_tr, diag, q, R, alpha, epochs, lr, **kwargs):
    mean = kwargs.pop("mean", None)
    prior = kwargs.pop("prior", None)
    from_data = kwargs.pop("from_data", True)
    init_ls = kwargs.pop("init_ls", True)

    def _train(subkey):
        gp_pre = build_nmix_rff(
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
        gp, gp_losses = mar_srfr(
            gp_pre, target, y_tr, lr=lr, epochs=epochs, alpha=alpha, **kwargs
        )

        return gp, gp_losses

    restarts = kwargs.pop("restarts", 1)
    best_gp, best_loss = train_with_restarts(key, _train, restarts)

    return best_gp, best_loss

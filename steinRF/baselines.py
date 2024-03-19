# ---------------------------------------------------------------------------------------- #
#                                         BASELINES                                        #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from copy import deepcopy
# import gpflow
from scipy.stats import chi
import math
from steinRF.utils import mse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as RBF_skl
from tensorflow_probability.substrates.jax import distributions as tfd
import gpjax as gpx
from gpjax.kernels import RBF as RBF_gpj
import optax as ox
from sklearn.model_selection import KFold


# -------------------------------------- SKLEARN GP -------------------------------------- #
def skl_gp(X_train, y_train, **kwargs):
    gp = GaussianProcessRegressor(kernel=RBF_skl(), n_restarts_optimizer=10, **kwargs)
    gp.fit(X_train, y_train)
    return gp


# ---------------------------------------- NYSTROM --------------------------------------- #
def nystrom_rbf(key, X_train, X_test, R=100):
    seed = int(key[0])
    nys = Nystroem(kernel='rbf', gamma=0.5, random_state=seed, n_components=R)
    nys.fit(X_train)
    phi_X = nys.transform(X_test)

    K_approx = phi_X @ phi_X.T
    # rbf_skl = RBF(0.5)
    # K = rbf_skl(X_test)
    K = rbf_kernel(X_test, gamma=0.5)
    return K, K_approx


# ------------------------------ ORTHOGONAL RANDOM FEATURES ------------------------------ #
# CODE FROM https://github.com/teddykoker/performer/blob/main/performer.py#L64-L82
# generate IID Gaussian random features
def iid_gaussian(m, d):
    return np.random.normal(size=(m, d))


# generate orthogonal Gaussian random features
def orthogonal_gaussian(m, d):
    def orthogonal_square():
        # create orthogonal square matrix using Gram-Schmidt
        q, _ = jnp.linalg.qr(iid_gaussian(d, d))
        S = jnp.diag(chi.rvs(df=d, size=d))
        return S @ q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = jnp.vstack(blocks)
    # matrix /= jnp.sqrt(num_squares + remainder / d)
    # matrix = np.diag(np.sqrt(d) * np.ones(m)) @ matrix

    return matrix


# adapted from https://neonnnnn.github.io/pyrfm
def orf_feats(key, R, d):
    n_feats = math.ceil(R / d)
    G = jax.random.normal(key, (n_feats, d, d))
    keys = jax.random.split(key, R)

    ws = []
    for r in range(n_feats):
        g = G[r, :, :]
        S = jnp.diag(
            chi.rvs(
                df=d, size=d, 
                random_state=np.random.RandomState(keys[r][0])
            )
        )
        Q = jnp.linalg.qr(g)[0]
        ws.append(Q @ S)
    
    return jnp.vstack(ws)


# --------------------------------- SPARSE VARIATIONAL GP -------------------------------- #
def build_svgp(key, X, y, R, diag, mean=None, from_data=True):
    if from_data:
        dX = X[:, None, :] - X[None, :, :]
        ls_init = jnp.median(dX**2, axis=(0, 1))
        k = RBF_gpj(lengthscale=ls_init)
    else:
        k = RBF_gpj()

    dataset = gpx.Dataset(X, jnp.array(y).reshape(-1, 1))
    if mean is None:
        mean = gpx.mean_functions.Zero()
    z = jax.random.uniform(key, (R, X.shape[1]), minval=X.min(axis=0), maxval=X.max(axis=0))
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=X.shape[0])
    prior = gpx.gps.Prior(mean_function=mean, kernel=k)
    p = prior * likelihood
    q = gpx.variational_families.VariationalGaussian(posterior=p, inducing_inputs=z)
    return dataset, q


def build_train_svgp(key, X, y, R, diag, epochs, lr, mean=None, **kwargs):
    mean = kwargs.pop("mean", None)
    from_data = kwargs.pop("from_data", True)

    def _train(subkey):
        data, model = build_svgp(subkey, X, y, R, diag, mean, from_data)
        negative_elbo = gpx.objectives.ELBO(negative=True)

        opt_posterior, history = gpx.fit(
            model=model,
            objective=negative_elbo,
            train_data=data,
            optim=ox.adam(learning_rate=lr),
            num_iters=epochs,
            key=subkey,
            verbose=kwargs.pop("verbose", False),
            **kwargs  
        )

        return opt_posterior, history
    
    return _train(key)


def svgp_predict(model, X_test):
    latent_dist = model(X_test)
    predictive_dist = model.posterior.likelihood(latent_dist)

    meanf = predictive_dist.mean()
    sigma = predictive_dist.stddev()
    return meanf, sigma


def svgp_cross_val(key, X, y, params, n_folds=5, metric=mse):
    key = jax.random.split(key, 1)[0]
    seed = int(key[0])
    
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=True)
    accuracies = []

    keys = jax.random.split(key, n_folds)
    key_ind = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index] 

        gp, _ = build_train_svgp(keys[key_ind], X_train, y_train, **params)
        gp_preds, _ = svgp_predict(gp, X_test)
        accuracies.append(metric(y_test, gp_preds))
        key_ind += 1

    return np.mean(accuracies)


# ---------------------------------------------------------------------------------------- #
#                                EQUINOX GP IMPLEMENTATIONS                                #
# ---------------------------------------------------------------------------------------- #

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Bool
from typing import Callable
from .kernels import RFF
from jax import jit, vmap
from functools import partial
from jax.lax import cond
import jax.tree_util as jtu
from tensorflow_probability.substrates.jax import distributions as tfd
from tinygp.helpers import JAXArray

# from steinRF.utils import stabilize



# --------------------------------------- BASIC GP --------------------------------------- #
class GP(eqx.Module):
    kernel: eqx.Module
    X: Float[Array, "N d"]
    diag: Float
    
    def __init__(self, kernel, X, diag=None):
        
        self.X = X
        self.kernel = kernel

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = jnp.log(diag)

    @property
    def _diag(self):
        return jnp.exp(self.diag)

    @partial(jax.jit, static_argnums=(2,))
    def nll(self, y, solver="chol"):
        return cond(solver == "chol", self.chol_nll, self.full_nll, y)

    @jit
    def full_nll(self, y):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self._diag
        n = y.shape[0]
        term1 = 0.5 * jnp.dot(y.T, jnp.linalg.solve(K, y))
        term2 = 0.5 * jnp.linalg.slogdet(K)[1]
        term3 = 0.5 * n * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3

    @jit
    def chol_nll(self, y):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self._diag
        n = y.shape[0]
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        term1 = 0.5 * jnp.dot(y.T, alpha)
        term2 = jnp.sum(jnp.log(jnp.diag(L)))
        term3 = 0.5 * n * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3

    @partial(jax.jit, static_argnums=(3,))
    def condition(self, y, X_test, solver="chol"):
        return cond(solver == "chol", self.chol_condition, self.full_condition, y, X_test)

    @jit
    def full_condition(self, y, X_test):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self._diag
        K_star = self.kernel(self.X, X_test)
        K_star_star = self.kernel(X_test, X_test) + jnp.eye(X_test.shape[0]) * self.diag
        
        K_inv = jnp.linalg.inv(K)
        
        mu_pred = jnp.dot(K_star.T, jnp.linalg.solve(K, y))
        sigma_pred = jnp.sqrt(jnp.diag(K_star_star - jnp.dot(K_star.T, jnp.dot(K_inv, K_star))))
        
        return jnp.array([mu_pred, sigma_pred])

    @jit
    def chol_condition(self, y, X_test):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self._diag
        K_star = self.kernel(self.X, X_test)
        K_star_star = self.kernel(X_test, X_test) + jnp.eye(X_test.shape[0]) * self._diag
        
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        
        mu_pred = jnp.dot(K_star.T, alpha)
        v = jnp.linalg.solve(L, K_star)
        sigma_pred = jnp.sqrt(jnp.diag(K_star_star - jnp.dot(v.T, v)))
        
        return jnp.array([mu_pred, sigma_pred])

    @partial(jax.jit, static_argnums=(3,))
    def __call__(self, y, X_test, solver="full"):
        mu, sigma = cond(solver == "chol", self.full_condition, self.chol_condition, y, X_test)

        # mu, sigma = jnp.where(
        #     solver == "chol", self.chol_condition(y, X_test), self.condition(y, X_test)
        # )
        return mu, sigma


# -------------------------------------- LOW RANK GP ------------------------------------- #
class LowRankGP(eqx.Module):
    mean: eqx.Module
    kernel: eqx.Module
    X: Float[Array, "N d"]
    diag: Float

    def __init__(self, kernel, X, diag=None, mean=None):
        self.X = X
        self.kernel = kernel

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = jnp.log(diag)

        if mean is None:
            mean = ZeroMean()
        self.mean = mean

    @property
    def _diag(self):
        return jnp.exp(self.diag)

    @partial(jit, static_argnums=(2,))
    def nll(self, y, solver="chol"):
        # return cond(solver == "chol", self.chol_nll, self.chol_nll, y)
        return self.chol_nll(y)

    @eqx.filter_jit
    def chol_nll(self, y: JAXArray) -> JAXArray:
        diag = self._diag
        y_diff = y - self.mean(self.X)
        phiX = self.kernel.phi(self.X)
        n, m = phiX.shape
        A = phiX.T @ phiX + jnp.eye(m) * diag
        R = jnp.linalg.cholesky(A)
        y_loss = jnp.linalg.solve(R, phiX.T @ y_diff)

        lml_1 = -((y_diff.T @ y_diff) - (y_loss.T @ y_loss)) / (2 * diag)
        lml_2 = -0.5 * jnp.sum(jnp.log(jnp.diag(R)**2))
        lml_3 = m * jnp.log(m * diag)
        lml_4 = -0.5 * n * jnp.log(2 * jnp.pi * diag)
        lml = lml_1 + lml_2 + lml_3 + lml_4
        return -lml

    @eqx.filter_jit
    def condition(self, y_train, X_test):
        y_diff = y_train - self.mean(self.X)
        phiXt = self.kernel.phi(X_test)
        phiX = self.kernel.phi(self.X)
        n, m = phiX.shape
        A = phiX.T @ phiX + jnp.eye(m) * self._diag
        R = jnp.linalg.cholesky(A)

        # mean calculation
        R_phiy = jnp.linalg.solve(R, phiX.T @ y_diff)
        y_pred = jnp.linalg.solve(R.T, R_phiy)
        mu = phiXt @ y_pred + self.mean(X_test)

        # variance calculation
        R_phixt = jnp.linalg.solve(R, phiXt.T)
        V = R_phixt.T @ R_phixt * self._diag + self._diag
        # V = R_phixt.T @ R_phixt
        sigma = jnp.sqrt(jnp.diag(V))

        return mu, sigma

    # @partial(eqx.filter_jit, static_argnums=(4,))
    def __call__(self, y_train, X_test, diag=jnp.float32(0.)):
        diag = cond(diag == 0., lambda: self._diag, lambda: diag)
        return self.condition(y_train, X_test, diag)


# -------------------------------------- MARGINAL GP ------------------------------------- #
class MixGP(eqx.Module):
    mean: eqx.Module
    kernel: eqx.Module
    X: Float[Array, "N d"]
    diag: Float

    def __init__(self, kernel, X, diag=None, mean=None):
        self.X = X
        self.kernel = kernel

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = jnp.log(diag)

        if mean is None:
            mean = ZeroMean()
        self.mean = mean

    @property
    def _diag(self):
        return jnp.exp(self.diag)

    @partial(jit, static_argnums=(2,))
    def nll(self, y: JAXArray, solver="chol") -> JAXArray:
        return self.multi_nll(y).mean()

    @eqx.filter_jit
    def multi_nll(self, y: JAXArray) -> JAXArray:
        PhiX = self.kernel.phi(self.X)
        nlls = vmap(self.single_nll, (0, None))(PhiX, y)
        return nlls

    @eqx.filter_jit
    def single_nll(self, phiX: JAXArray, y: JAXArray) -> JAXArray:
        y_diff = y - self.mean(self.X)
        diag = self._diag
        n, m = phiX.shape
        A = phiX.T @ phiX + jnp.eye(m) * diag
        # A = stabilize(A[None, :, :])[0]
        R = jnp.linalg.cholesky(A)
        y_loss = jnp.linalg.solve(R, phiX.T @ y_diff)

        lml_1 = -((y_diff.T @ y_diff) - (y_loss.T @ y_loss)) / (2 * diag)
        lml_2 = -0.5 * jnp.sum(jnp.log(jnp.diag(R)**2))
        lml_3 = m * jnp.log(m * diag)
        lml_4 = -0.5 * n * jnp.log(2 * jnp.pi * diag)
        lml = lml_1 + lml_2 + lml_3 + lml_4
        return -lml

    @eqx.filter_jit
    def single_condition(
        self, y_train: JAXArray,
        Xtest: JAXArray,
        phiX: JAXArray, 
        phiXt: JAXArray, 
        diag: float
    ) -> JAXArray:
        
        n, m = phiX.shape
        y_diff = y_train - self.mean(self.X)
        A = phiX.T @ phiX + jnp.eye(m) * diag
        # A = stabilize(A[None, :, :])[0]
        R = jnp.linalg.cholesky(A)

        # mean calculation
        R_phiy = jnp.linalg.solve(R, phiX.T @ y_diff)
        y_pred = jnp.linalg.solve(R.T, R_phiy)
        mu = phiXt @ y_pred + self.mean(Xtest)

        # variance calculation
        R_phixt = jnp.linalg.solve(R, phiXt.T)
        V = R_phixt.T @ R_phixt * self._diag + self._diag
        sigma = jnp.sqrt(jnp.diag(V))

        return mu, sigma

    @eqx.filter_jit
    def multi_condition(self, y_train: JAXArray, X_test: JAXArray, diag: float) -> JAXArray:
        PhiX = self.kernel.phi(self.X)
        PhiXt = self.kernel.phi(X_test)
        # return self.single_condition(y_train, PhiX[2, :, :], PhiXt[2, :, :])
        mus, sds = vmap(self.single_condition, (None, None, 0, 0, None))(
            y_train, X_test, PhiX, PhiXt, diag
        )
        return mus, sds

    @partial(jax.jit, static_argnames=('diag',))
    def condition(self, key, y_train, X_test, n_samples=100, diag=None):
        if diag is None:
            diag = self._diag
        # return self.multi_condition(y_train, X_test, diag)
        mus, sds = self.multi_condition(y_train, X_test, diag)
        dists = tfd.MultivariateNormalDiag(mus, sds)
        samples = dists.sample(n_samples, seed=key).reshape(-1, X_test.shape[0])
        return jnp.nanmean(samples, axis=0), jnp.nanstd(samples, axis=0)

    # @partial(eqx.filter_jit, static_argnums=(4,))
    def __call__(self, y_train, X_test):
        # diag = cond(diag == 0., lambda: self.diag, lambda: diag)
        return self.condition(y_train, X_test)



# ------------------------------------ MEAN FUNCTIONS ------------------------------------ #
class ZeroMean(eqx.Module):
    @jax.jit
    def __call__(self, X: Float[Array, "N d"]) -> Float[Array, "N"]:
        return jnp.zeros(X.shape[0])


class ConstantMean(eqx.Module):
    mean: Float

    def __init__(self, mean):
        self.mean = mean
    
    @jax.jit
    def __call__(self, X: Float[Array, "N d"]) -> Float[Array, "N"]:
        return jnp.broadcast_to(self.mean, X.shape[0])
    

class LinearMean(eqx.Module):
    weights: Float[Array, "d"]

    def __init__(self, weights):
        self.weights = weights

    @jax.jit
    def __call__(self, X: Float[Array, "N d"]) -> Float[Array, "N"]:
        return jnp.dot(X, self.weights)

"""Copula estimation, high-dimensional simulation, tail-dependence analysis, and Monte Carlo VaR / Expected Shortfall computation."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from src.marginals import MarginalFit, inverse_pit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopulaFit:
    """Fitted Copula model parameters and dependence properties."""
    family: str  # 'gaussian', 't', or 'clayton'
    corr_matrix: np.ndarray
    df: Optional[float] = None
    theta: Optional[float] = None  # Parameter for Archimedean copulas
    tail_dep_lower: float = 0.0
    tail_dep_upper: float = 0.0


def kendall_tau_matrix(u_matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise sample Kendall's rank correlation matrix tau."""
    n_samples, n_dim = u_matrix.shape
    tau_mat = np.eye(n_dim)
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            tau, _ = stats.kendalltau(u_matrix[:, i], u_matrix[:, j])
            tau = float(np.nan_to_num(tau, nan=0.0))
            tau_mat[i, j] = tau
            tau_mat[j, i] = tau
    return tau_mat


def invert_tau_to_linear_corr(tau_mat: np.ndarray) -> np.ndarray:
    """Invert Kendall's tau to Pearson linear correlation for elliptical copulas: R = sin(pi * tau / 2)."""
    r = np.sin(np.pi * tau_mat / 2.0)
    np.fill_diagonal(r, 1.0)
    return r


def repair_to_nearest_psd(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project symmetric matrix to nearest Positive Semi-Definite (PSD) correlation matrix via spectral decomposition."""
    # Ensure exact symmetry
    sym = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)

    # Floor eigenvalues to positive threshold
    eigvals = np.maximum(eigvals, eps)
    reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Rescale to enforce unit diagonal
    inv_sqrt_diag = 1.0 / np.sqrt(np.diag(reconstructed))
    repaired = reconstructed * np.outer(inv_sqrt_diag, inv_sqrt_diag)
    # Exact 1 on diagonal and numerical symmetry
    np.fill_diagonal(repaired, 1.0)
    repaired = (repaired + repaired.T) / 2.0
    return repaired


def calculate_t_copula_tail_dependence(nu: float, rho: float) -> float:
    """Analytic bivariate lower and upper tail dependence coefficient for Student-t copula:

    lambda_L = lambda_U = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))).
    """
    if nu <= 0 or rho >= 1.0:
        return 1.0
    if rho <= -1.0:
        return 0.0

    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t.cdf(arg, df=nu + 1.0))


def t_copula_log_likelihood(nu: float, u_matrix: np.ndarray, r_matrix: np.ndarray) -> float:
    """Compute exact profile log-likelihood of Student-t copula for given degrees of freedom nu and correlation matrix R."""
    n_samples, n_dim = u_matrix.shape
    if nu <= 2.01 or np.isnan(nu):
        return -1e10

    # Invert uniform marginals via standard t quantile function
    zeta = stats.t.ppf(u_matrix, df=nu)  # shape (n_samples, n_dim)
    # Clip any infinities
    zeta = np.clip(zeta, -30.0, 30.0)

    # Invert R matrix
    try:
        r_inv = np.linalg.inv(r_matrix)
        sign, logdet_r = np.linalg.slogdet(r_matrix)
        if sign <= 0:
            return -1e10
    except np.linalg.LinAlgError:
        return -1e10

    # Vectorized quadratic form: sum_j sum_k zeta_ij * Rinv_jk * zeta_ik
    quad_form = np.einsum("ij,jk,ik->i", zeta, r_inv, zeta)

    # Copula log-likelihood terms
    term1 = special.gammaln((nu + n_dim) / 2.0) + (n_dim - 1) * special.gammaln(nu / 2.0)
    term2 = n_dim * special.gammaln((nu + 1.0) / 2.0) + 0.5 * logdet_r
    constant_term = term1 - term2

    sum_log_numerator = -((nu + n_dim) / 2.0) * np.log(1.0 + quad_form / nu)
    sum_log_denominator = -((nu + 1.0) / 2.0) * np.sum(np.log(1.0 + (zeta ** 2) / nu), axis=1)

    log_lik = n_samples * constant_term + np.sum(sum_log_numerator - sum_log_denominator)
    return float(log_lik) if np.isfinite(log_lik) else -1e10


def estimate_t_copula_df(
    u_matrix: np.ndarray,
    r_matrix: np.ndarray,
    nu_bounds: Tuple[float, float] = (2.1, 50.0),
) -> float:
    """Estimate degrees of freedom nu via profile maximum likelihood."""
    # Coarse grid search for global initialization
    grid = np.array([2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0])
    best_nu = 4.0
    best_ll = -np.inf

    for nu_val in grid:
        ll = t_copula_log_likelihood(nu_val, u_matrix, r_matrix)
        if ll > best_ll:
            best_ll = ll
            best_nu = float(nu_val)

    # Refine with scalar bounded optimizer
    res = optimize.minimize_scalar(
        lambda nu: -t_copula_log_likelihood(nu, u_matrix, r_matrix),
        bounds=nu_bounds,
        method="bounded",
    )
    if res.success and -res.fun > best_ll:
        best_nu = float(res.x)

    return float(np.clip(best_nu, nu_bounds[0], nu_bounds[1]))


def fit_copula(
    u_matrix: np.ndarray,
    family: str = "t",
    custom_df: Optional[float] = None,
) -> CopulaFit:
    """Fit Gaussian, Student-t, or Clayton Copula from uniform margins U in (0, 1)."""
    n_samples, n_dim = u_matrix.shape
    eps = 1e-12
    u_clipped = np.clip(u_matrix, eps, 1.0 - eps)

    tau_mat = kendall_tau_matrix(u_clipped)
    r_raw = invert_tau_to_linear_corr(tau_mat)
    r_psd = repair_to_nearest_psd(r_raw)

    avg_rho = float((np.sum(r_psd) - n_dim) / (n_dim * (n_dim - 1))) if n_dim > 1 else 0.0

    if family.lower() == "gaussian":
        return CopulaFit(
            family="gaussian",
            corr_matrix=r_psd,
            df=None,
            tail_dep_lower=0.0,
            tail_dep_upper=0.0,
        )

    elif family.lower() == "clayton":
        # Average pairwise Kendall's tau -> theta = 2*tau / (1 - tau)
        avg_tau = float((np.sum(tau_mat) - n_dim) / (n_dim * (n_dim - 1))) if n_dim > 1 else 0.5
        avg_tau = np.clip(avg_tau, 0.01, 0.95)
        theta = float(2.0 * avg_tau / (1.0 - avg_tau))
        lambda_l = float(2.0 ** (-1.0 / theta)) if theta > 0 else 0.0
        return CopulaFit(
            family="clayton",
            corr_matrix=r_psd,
            theta=theta,
            tail_dep_lower=lambda_l,
            tail_dep_upper=0.0,
        )

    elif family.lower() in ("t", "student_t"):
        if custom_df is not None:
            est_df = float(custom_df)
        else:
            est_df = estimate_t_copula_df(u_clipped, r_psd)

        tail_dep = calculate_t_copula_tail_dependence(est_df, avg_rho)
        return CopulaFit(
            family="t",
            corr_matrix=r_psd,
            df=est_df,
            tail_dep_lower=tail_dep,
            tail_dep_upper=tail_dep,
        )
    else:
        raise ValueError(f"Unknown copula family: {family}. Supported: 'gaussian', 't', 'clayton'.")


def sample_copula(
    copula: CopulaFit,
    n_samples: int = 10000,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Sample multivariate uniform margins U ~ Copula(R, nu / theta)."""
    rng = np.random.default_rng(seed)
    n_dim = copula.corr_matrix.shape[0]

    if copula.family == "clayton":
        # Clayton simulation via Marshall-Olkin representation:
        # V ~ Gamma(1/theta, 1), X_i ~ Exp(1) i.i.d., U_i = (1 + X_i / V)^(-1/theta)
        theta = copula.theta or 2.0
        v = rng.gamma(shape=1.0 / theta, scale=1.0, size=(n_samples, 1))
        x = rng.exponential(scale=1.0, size=(n_samples, n_dim))
        u = (1.0 + x / v) ** (-1.0 / theta)
        eps = 1e-12
        return np.clip(u, eps, 1.0 - eps)

    # Cholesky decomposition of repaired correlation matrix
    try:
        l_mat = np.linalg.cholesky(copula.corr_matrix)
    except np.linalg.LinAlgError:
        r_psd = repair_to_nearest_psd(copula.corr_matrix, eps=1e-5)
        l_mat = np.linalg.cholesky(r_psd)

    # Standard normal vector Z ~ N(0, R)
    z = rng.standard_normal((n_samples, n_dim)) @ l_mat.T

    if copula.family == "gaussian":
        u = stats.norm.cdf(z)
    elif copula.family == "t":
        nu = copula.df or 4.0
        w = rng.chisquare(df=nu, size=(n_samples, 1)) / nu
        t_samples = z / np.sqrt(w)
        u = stats.t.cdf(t_samples, df=nu)
    else:
        raise ValueError(f"Unsupported copula family for sampling: {copula.family}")

    eps = 1e-12
    return np.clip(u, eps, 1.0 - eps)


def calculate_portfolio_risk(
    weights: np.ndarray,
    copula: CopulaFit,
    marginal_fits: List[MarginalFit],
    alpha: float = 0.99,
    n_samples: int = 10000,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """Calculate Monte Carlo Value-at-Risk (VaR_alpha) and Expected Shortfall (ES_alpha).

    Parameters
    ----------
    weights : np.ndarray
        Portfolio asset weights (summing to 1).
    copula : CopulaFit
        Fitted copula dependence structure.
    marginal_fits : List[MarginalFit]
        Univariate marginal fits for each asset.
    alpha : float
        Confidence level (e.g. 0.99).
    n_samples : int
        Number of Monte Carlo scenarios.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    Tuple[float, float]
        (VaR_alpha, ES_alpha) where both are positive numbers representing potential loss fractions.
    """
    weights = np.asarray(weights)
    n_assets = len(weights)
    if len(marginal_fits) != n_assets:
        raise ValueError(f"Mismatch: {n_assets} weights vs {len(marginal_fits)} marginal fits.")

    # 1. Sample uniform copula margins
    u_sim = sample_copula(copula, n_samples=n_samples, seed=seed)

    # 2. Transform through inverse PIT to simulated returns
    sim_returns = np.zeros((n_samples, n_assets))
    for i in range(n_assets):
        sim_returns[:, i] = inverse_pit(u_sim[:, i], marginal_fits[i])

    # 3. Compute portfolio losses: L = - R_portfolio
    portfolio_returns = sim_returns @ weights
    losses = -portfolio_returns

    # 4. Quantile for VaR and Tail Expectation for Expected Shortfall
    var_alpha = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= var_alpha]
    es_alpha = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_alpha

    return var_alpha, es_alpha

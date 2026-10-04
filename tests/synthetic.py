"""Synthetic data generator with known statistical parameters for testing and calibration."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def generate_synthetic_returns(
    n_days: int = 300,
    n_assets: int = 3,
    df_marginals: Optional[List[float]] = None,
    locs: Optional[List[float]] = None,
    scales: Optional[List[float]] = None,
    corr_matrix: Optional[np.ndarray] = None,
    copula_type: str = "t",
    copula_df: float = 4.0,
    seed: int = 42,
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Generate multivariate return series with known marginal t-distributions

    and Gaussian or Student-t copula dependence structure.
    """
    rng = np.random.default_rng(seed)

    if df_marginals is None:
        df_marginals = [5.0 + i * 2.0 for i in range(n_assets)]
    if locs is None:
        locs = [0.0005 for _ in range(n_assets)]
    if scales is None:
        scales = [0.015 for _ in range(n_assets)]

    if corr_matrix is None:
        # Generate a valid equicorrelation or toeplitz correlation matrix
        rho = 0.5
        corr_matrix = np.full((n_assets, n_assets), rho)
        np.fill_diagonal(corr_matrix, 1.0)

    # 1. Sample from Copula to get uniform marginals U in (0, 1)
    # Cholesky decomposition of correlation matrix
    L = np.linalg.cholesky(corr_matrix)
    Z = rng.standard_normal((n_days, n_assets)) @ L.T

    if copula_type.lower() == "t":
        # W ~ chi2(nu) / nu
        w = rng.chisquare(df=copula_df, size=(n_days, 1)) / copula_df
        t_samples = Z / np.sqrt(w)
        # Uniform marginals via t CDF
        U = stats.t.cdf(t_samples, df=copula_df)
    elif copula_type.lower() == "gaussian":
        U = stats.norm.cdf(Z)
    else:
        raise ValueError(f"Unsupported copula type: {copula_type}")

    # Clip to avoid exact 0 or 1
    eps = 1e-12
    U = np.clip(U, eps, 1.0 - eps)

    # 2. Transform Uniform marginals to Student-t margins via PPF
    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        returns[:, i] = stats.t.ppf(
            U[:, i],
            df=df_marginals[i],
            loc=locs[i],
            scale=scales[i],
        )

    # 3. Format into a pandas DataFrame with business daily dates
    date_range = pd.bdate_range(start=start_date, periods=n_days)
    col_names = [f"ASSET_{i+1}" for i in range(n_assets)]
    return pd.DataFrame(returns, index=date_range, columns=col_names)

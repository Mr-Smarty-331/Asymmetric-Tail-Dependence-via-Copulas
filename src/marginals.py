"""Univariate marginal distribution fitting (Student-t) and Probability Integral Transform (PIT)."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarginalFit:
    """Fitted parameters and goodness-of-fit diagnostic for univariate marginal distribution."""
    df: float
    loc: float
    scale: float
    ks_stat: float
    ks_pvalue: float


def fit_marginal_t(
    series: Union[pd.Series, np.ndarray],
    max_df: float = 100.0,
    min_df: float = 2.01,
) -> MarginalFit:
    """Fit univariate Student-t distribution via MLE with bounded degrees of freedom.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        1D array or Series of asset returns.
    max_df : float
        Upper cap on degrees of freedom to prevent numerical instability as fit approaches Gaussianity.
    min_df : float
        Lower floor on degrees of freedom ensuring finite variance (default 2.01).

    Returns
    -------
    MarginalFit
        Fitted df, loc, scale, and Kolmogorov-Smirnov test statistic and p-value.
    """
    if isinstance(series, pd.Series):
        values = series.dropna().values
    else:
        values = np.asarray(series)
        values = values[~np.isnan(values)]

    if len(values) < 30:
        raise ValueError(
            f"Insufficient observations to fit marginal distribution (got {len(values)}, minimum 30 required)."
        )

    # Check for constant series
    std_val = float(np.std(values))
    if std_val < 1e-10 or np.isclose(std_val, 0.0):
        raise ValueError("Cannot fit marginal distribution on a constant or zero-variance series.")

    # Fit Student-t distribution
    df_est, loc_est, scale_est = stats.t.fit(values)

    # Apply bounds on df
    df_bounded = float(np.clip(df_est, min_df, max_df))
    loc_val = float(loc_est)
    scale_val = max(float(scale_est), 1e-8)

    # Kolmogorov-Smirnov diagnostic test
    ks_stat, ks_pvalue = stats.kstest(
        values,
        lambda x: stats.t.cdf(x, df=df_bounded, loc=loc_val, scale=scale_val),
    )

    return MarginalFit(
        df=df_bounded,
        loc=loc_val,
        scale=scale_val,
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
    )


def pit_transform(
    series: Union[pd.Series, np.ndarray],
    fit: MarginalFit,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply Probability Integral Transform (PIT) mapping returns to uniform U ~ (0, 1).

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Asset return values.
    fit : MarginalFit
        Fitted marginal distribution parameters.
    eps : float
        Epsilon bound to prevent exact 0.0 or 1.0 values.

    Returns
    -------
    np.ndarray
        Uniform marginal values strictly within [eps, 1 - eps].
    """
    if isinstance(series, pd.Series):
        vals = series.values
    else:
        vals = np.asarray(series)

    u = stats.t.cdf(vals, df=fit.df, loc=fit.loc, scale=fit.scale)
    return np.clip(u, eps, 1.0 - eps)


def inverse_pit(
    u: Union[np.ndarray, float],
    fit: MarginalFit,
    eps: float = 1e-12,
) -> np.ndarray:
    """Inverse PIT (Quantile / PPF transformation) mapping uniform U in (0, 1) back to return domain.

    Parameters
    ----------
    u : np.ndarray or float
        Uniform marginal values.
    fit : MarginalFit
        Fitted marginal distribution parameters.
    eps : float
        Epsilon clipping threshold.

    Returns
    -------
    np.ndarray
        Returns corresponding to specified uniform quantiles.
    """
    u_clipped = np.clip(np.asarray(u), eps, 1.0 - eps)
    return stats.t.ppf(u_clipped, df=fit.df, loc=fit.loc, scale=fit.scale)


def fit_all_marginals(
    returns_df: pd.DataFrame,
    max_df: float = 100.0,
    log_diagnostics: bool = True,
) -> Dict[str, MarginalFit]:
    """Fit univariate Student-t marginal distributions to all columns in a returns DataFrame and log KS results."""
    fits = {}
    for col in returns_df.columns:
        fit = fit_marginal_t(returns_df[col], max_df=max_df)
        fits[str(col)] = fit
        if log_diagnostics:
            logger.info(
                "Marginal Fit [%s]: df=%.2f, loc=%.6f, scale=%.6f, KS stat=%.4f (p=%.4f)",
                col,
                fit.df,
                fit.loc,
                fit.scale,
                fit.ks_stat,
                fit.ks_pvalue,
            )
    return fits

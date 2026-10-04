"""Rolling backtest engine, exception detection, statistical tests (Kupiec, Christoffersen), Basel traffic light, and disk caching."""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.copula import CopulaFit, calculate_portfolio_risk, fit_copula
from src.marginals import MarginalFit, fit_marginal_t, pit_transform

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    dates: pd.DatetimeIndex
    portfolio_losses: np.ndarray
    var_forecasts: np.ndarray
    es_forecasts: np.ndarray
    exceptions: np.ndarray
    n_exceptions: int
    n_obs: int
    exception_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float
    conditional_coverage_stat: float
    conditional_coverage_pvalue: float
    traffic_light: str  # 'Green', 'Amber', 'Red'
    model_name: str = "Student-t Copula"

    def to_dict(self) -> Dict[str, Any]:
        """Convert backtest results to serializable dictionary."""
        return {
            "dates": [d.strftime("%Y-%m-%d") for d in self.dates],
            "portfolio_losses": self.portfolio_losses.tolist(),
            "var_forecasts": self.var_forecasts.tolist(),
            "es_forecasts": self.es_forecasts.tolist(),
            "exceptions": self.exceptions.tolist(),
            "n_exceptions": int(self.n_exceptions),
            "n_obs": int(self.n_obs),
            "exception_rate": float(self.exception_rate),
            "kupiec_stat": float(self.kupiec_stat),
            "kupiec_pvalue": float(self.kupiec_pvalue),
            "christoffersen_stat": float(self.christoffersen_stat),
            "christoffersen_pvalue": float(self.christoffersen_pvalue),
            "conditional_coverage_stat": float(self.conditional_coverage_stat),
            "conditional_coverage_pvalue": float(self.conditional_coverage_pvalue),
            "traffic_light": self.traffic_light,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestResults":
        """Reconstruct BacktestResults from dictionary."""
        return cls(
            dates=pd.to_datetime(data["dates"]),
            portfolio_losses=np.array(data["portfolio_losses"]),
            var_forecasts=np.array(data["var_forecasts"]),
            es_forecasts=np.array(data["es_forecasts"]),
            exceptions=np.array(data["exceptions"]),
            n_exceptions=data["n_exceptions"],
            n_obs=data["n_obs"],
            exception_rate=data["exception_rate"],
            kupiec_stat=data["kupiec_stat"],
            kupiec_pvalue=data["kupiec_pvalue"],
            christoffersen_stat=data["christoffersen_stat"],
            christoffersen_pvalue=data["christoffersen_pvalue"],
            conditional_coverage_stat=data["conditional_coverage_stat"],
            conditional_coverage_pvalue=data["conditional_coverage_pvalue"],
            traffic_light=data["traffic_light"],
            model_name=data.get("model_name", "Student-t Copula"),
        )


def kupiec_pof_test(n_exceptions: int, n_obs: int, alpha: float = 0.99) -> Tuple[float, float]:
    """Kupiec Proportion of Failures (POF) Likelihood Ratio test.

    H0: p = 1 - alpha.

    LR_pof = -2 ln( ( (1-p)^(N-x) * p^x ) / ( (1 - x/N)^(N-x) * (x/N)^x ) )
    """
    p = 1.0 - alpha
    x = int(n_exceptions)
    n = int(n_obs)

    if n <= 0:
        return 0.0, 1.0

    if x == 0:
        # Limit as x -> 0: (1 - x/N)^(N-x) * (x/N)^x -> 1
        log_lik_null = n * np.log(1.0 - p)
        log_lik_alt = 0.0
        lr_stat = -2.0 * (log_lik_null - log_lik_alt)
    elif x == n:
        # Limit as x -> N
        log_lik_null = n * np.log(p)
        log_lik_alt = 0.0
        lr_stat = -2.0 * (log_lik_null - log_lik_alt)
    else:
        p_hat = x / n
        log_lik_null = (n - x) * np.log(1.0 - p) + x * np.log(p)
        log_lik_alt = (n - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
        lr_stat = -2.0 * (log_lik_null - log_lik_alt)

    lr_stat = max(0.0, float(lr_stat))
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    return lr_stat, p_value


def christoffersen_independence_test(exceptions: np.ndarray) -> Tuple[float, float]:
    """Christoffersen Markov independence test for consecutive exception clustering.

    H0: Exceptions are independent Bernoulli trials.
    """
    if len(exceptions) < 2:
        return 0.0, 1.0

    e = (np.asarray(exceptions) > 0).astype(int)
    n00 = int(np.sum((e[:-1] == 0) & (e[1:] == 0)))
    n01 = int(np.sum((e[:-1] == 0) & (e[1:] == 1)))
    n10 = int(np.sum((e[:-1] == 1) & (e[1:] == 0)))
    n11 = int(np.sum((e[:-1] == 1) & (e[1:] == 1)))

    # If no exceptions occurred at all, or only 1 exception occurred, no clustering can be detected
    if (n01 + n11) == 0:
        return 0.0, 1.0

    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    # If transition state is degenerate
    if pi_0 <= 0.0 and pi_1 <= 0.0:
        return 0.0, 1.0

    eps = 1e-12
    # Compute log-likelihood under H0 (independent)
    ll_null = 0.0
    if (n00 + n10) > 0 and (1.0 - pi) > 0:
        ll_null += (n00 + n10) * np.log(max(1.0 - pi, eps))
    if (n01 + n11) > 0 and pi > 0:
        ll_null += (n01 + n11) * np.log(max(pi, eps))

    # Compute log-likelihood under H1 (first-order Markov)
    ll_alt = 0.0
    if n00 > 0 and (1.0 - pi_0) > 0:
        ll_alt += n00 * np.log(max(1.0 - pi_0, eps))
    if n01 > 0 and pi_0 > 0:
        ll_alt += n01 * np.log(max(pi_0, eps))
    if n10 > 0 and (1.0 - pi_1) > 0:
        ll_alt += n10 * np.log(max(1.0 - pi_1, eps))
    if n11 > 0 and pi_1 > 0:
        ll_alt += n11 * np.log(max(pi_1, eps))

    lr_ind = -2.0 * (ll_null - ll_alt)
    lr_ind = max(0.0, float(lr_ind))
    p_value = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    return lr_ind, p_value


def christoffersen_conditional_coverage_test(
    n_exceptions: int,
    n_obs: int,
    exceptions: np.ndarray,
    alpha: float = 0.99,
) -> Tuple[float, float]:
    """Christoffersen Conditional Coverage test: LR_cc = LR_pof + LR_ind ~ chi2(2)."""
    lr_pof, _ = kupiec_pof_test(n_exceptions, n_obs, alpha=alpha)
    lr_ind, _ = christoffersen_independence_test(exceptions)
    lr_cc = lr_pof + lr_ind
    p_value = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    return float(lr_cc), p_value


def classify_traffic_light(n_exceptions: int, n_obs: int = 250, alpha: float = 0.99) -> str:
    """Basel Committee traffic light zones parameterized for N observations at alpha=0.99.

    For standard N=250:
    - Green: 0 to 4 exceptions (cumulative binomial prob < 95%)
    - Amber: 5 to 9 exceptions (95% to 99.99%)
    - Red: 10+ exceptions (>= 99.99%)
    """
    scale = n_obs / 250.0
    green_cutoff = int(np.floor(4 * scale))
    amber_cutoff = int(np.floor(9 * scale))

    if n_exceptions <= green_cutoff:
        return "Green"
    elif n_exceptions <= amber_cutoff:
        return "Amber"
    else:
        return "Red"


def run_rolling_backtest(
    returns_df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    window_size: int = 252,
    alpha: float = 0.99,
    copula_family: str = "t",
    n_mc_samples: int = 5000,
    cache_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> BacktestResults:
    """Execute rolling out-of-sample backtest with strict no-lookahead guarantee.

    For each day t in test set:
    - Training data uses strictly rows [t - window_size : t] (data up to t-1).
    - Forecast VaR_t and ES_t are evaluated against realized loss on day t: L_t = - R_portfolio,t.
    """
    if cache_path and use_cache and Path(cache_path).exists():
        with open(cache_path, "r") as f:
            data = json.load(f)
            return BacktestResults.from_dict(data)

    n_days, n_assets = returns_df.shape
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    else:
        weights = np.asarray(weights)

    if n_days <= window_size:
        raise ValueError(
            f"Insufficient history: {n_days} days provided, window requires at least {window_size + 1} days."
        )

    test_dates = returns_df.index[window_size:]
    n_test_days = len(test_dates)

    var_forecasts = np.zeros(n_test_days)
    es_forecasts = np.zeros(n_test_days)
    portfolio_losses = np.zeros(n_test_days)

    # Realized daily portfolio return for each day
    realized_portfolio_returns = returns_df.values @ weights

    for i in range(n_test_days):
        # Strictly past data [i : i + window_size] (no look-ahead)
        train_slice = returns_df.iloc[i : i + window_size]

        if copula_family.lower() == "gaussian_var":
            # Traditional Gaussian Variance-Covariance benchmark
            mean_vec = train_slice.mean().values
            cov_mat = train_slice.cov().values
            port_mu = float(weights @ mean_vec)
            port_sigma = float(np.sqrt(weights @ cov_mat @ weights))
            z_alpha = float(stats.norm.ppf(alpha))
            var_t = -(port_mu - z_alpha * port_sigma)
            # Analytic Gaussian ES = -port_mu + port_sigma * phi(z_alpha) / (1 - alpha)
            es_t = -(port_mu) + port_sigma * (stats.norm.pdf(z_alpha) / (1.0 - alpha))
        else:
            # Copula-based modeling
            marginal_fits = [fit_marginal_t(train_slice.iloc[:, col]) for col in range(n_assets)]
            u_train = np.zeros((window_size, n_assets))
            for col in range(n_assets):
                u_train[:, col] = pit_transform(train_slice.iloc[:, col], marginal_fits[col])

            copula_fit = fit_copula(u_train, family=copula_family)

            var_t, es_t = calculate_portfolio_risk(
                weights=weights,
                copula=copula_fit,
                marginal_fits=marginal_fits,
                alpha=alpha,
                n_samples=n_mc_samples,
                seed=42 + i,
            )

        var_forecasts[i] = var_t
        es_forecasts[i] = es_t
        # Realized portfolio loss on day t (index window_size + i)
        portfolio_losses[i] = -realized_portfolio_returns[window_size + i]

    # Exception indicator: 1 if Loss_t > VaR_t
    exceptions = (portfolio_losses > var_forecasts).astype(int)
    n_exceptions = int(np.sum(exceptions))
    exception_rate = float(n_exceptions / n_test_days) if n_test_days > 0 else 0.0

    kupiec_stat, kupiec_p = kupiec_pof_test(n_exceptions, n_test_days, alpha=alpha)
    christoffersen_stat, christoffersen_p = christoffersen_independence_test(exceptions)
    cc_stat, cc_p = christoffersen_conditional_coverage_test(n_exceptions, n_test_days, exceptions, alpha=alpha)
    traffic_light = classify_traffic_light(n_exceptions, n_test_days, alpha=alpha)

    model_label = "Student-t Copula" if copula_family == "t" else ("Gaussian Copula" if copula_family == "gaussian" else "Gaussian Parametric")

    results = BacktestResults(
        dates=test_dates,
        portfolio_losses=portfolio_losses,
        var_forecasts=var_forecasts,
        es_forecasts=es_forecasts,
        exceptions=exceptions,
        n_exceptions=n_exceptions,
        n_obs=n_test_days,
        exception_rate=exception_rate,
        kupiec_stat=kupiec_stat,
        kupiec_pvalue=kupiec_p,
        christoffersen_stat=christoffersen_stat,
        christoffersen_pvalue=christoffersen_p,
        conditional_coverage_stat=cc_stat,
        conditional_coverage_pvalue=cc_p,
        traffic_light=traffic_light,
        model_name=model_label,
    )

    if cache_path:
        cache_p = Path(cache_path)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_p, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

    return results

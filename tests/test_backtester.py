"""Unit and Statistical Validation Tests for Rolling Backtest Engine, Kupiec, Christoffersen, and Basel Traffic Light."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtester import (
    BacktestResults,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    classify_traffic_light,
    kupiec_pof_test,
    run_rolling_backtest,
)
from tests.synthetic import generate_synthetic_returns


def test_no_lookahead_bias():
    """Verify modifying future return data at day t+5 leaves VaR forecast at day t completely unchanged."""
    df_base = generate_synthetic_returns(n_days=100, seed=42)
    # Modify data at the end of the series (day 95+)
    df_altered = df_base.copy()
    df_altered.iloc[90:, :] = df_altered.iloc[90:, :] * 10.0  # extreme shock in the future

    res_base = run_rolling_backtest(df_base, window_size=60, copula_family="gaussian_var")
    res_altered = run_rolling_backtest(df_altered, window_size=60, copula_family="gaussian_var")

    # Forecasts for days before day 90 (index 0 to 29 in test set) must be strictly identical
    np.testing.assert_allclose(res_base.var_forecasts[:29], res_altered.var_forecasts[:29], atol=1e-10)


def test_alignment_of_var_and_realized_loss():
    """Verify that VaR for date t is compared strictly with the loss realized on date t."""
    df_returns = generate_synthetic_returns(n_days=80, seed=42)
    weights = np.array([0.5, 0.3, 0.2])
    res = run_rolling_backtest(df_returns, weights=weights, window_size=50, copula_family="gaussian_var")

    # Check date alignment
    expected_test_dates = df_returns.index[50:]
    pd.testing.assert_index_equal(res.dates, expected_test_dates)

    # Check loss alignment
    portfolio_returns = df_returns.values @ weights
    expected_losses = -portfolio_returns[50:]
    np.testing.assert_allclose(res.portfolio_losses, expected_losses, atol=1e-10)

    # Check exception alignment
    expected_exceptions = (res.portfolio_losses > res.var_forecasts).astype(int)
    np.testing.assert_array_equal(res.exceptions, expected_exceptions)


def test_kupiec_pof_against_closed_form_and_boundary_cases():
    """Verify Kupiec POF Likelihood Ratio test matches exact analytical binomial likelihood ratio."""
    alpha = 0.99
    p = 0.01
    n = 250

    # 1. Expected exceptions (x = 2.5 -> x = 2 or 3)
    # When x = 2.5 (p_hat = p), LR = 0, p-value = 1.0
    lr_0, p_0 = kupiec_pof_test(n_exceptions=0, n_obs=n, alpha=alpha)
    assert lr_0 > 0.0
    assert 0.0 < p_0 < 1.0

    # Hand calculation for x = 0: LR = -2 * 250 * ln(0.99) ~= 5.025
    expected_lr_0 = -2.0 * 250 * np.log(0.99)
    assert np.isclose(lr_0, expected_lr_0, atol=1e-5)

    # 2. Borderline exceptions (x = 6): LR should be positive and reject at some level
    lr_6, p_6 = kupiec_pof_test(n_exceptions=6, n_obs=n, alpha=alpha)
    p_hat = 6.0 / 250.0
    expected_lr_6 = -2.0 * (
        (244 * np.log(0.99) + 6 * np.log(0.01)) - (244 * np.log(1.0 - p_hat) + 6 * np.log(p_hat))
    )
    assert np.isclose(lr_6, expected_lr_6, atol=1e-5)

    # 3. Upper boundary (x = N)
    lr_n, p_n = kupiec_pof_test(n_exceptions=n, n_obs=n, alpha=alpha)
    assert lr_n > 500.0
    assert p_n < 1e-10


def test_christoffersen_independence_clustering_vs_spaced():
    """Verify Christoffersen independence test detects clustering and accepts evenly spaced breaches."""
    # Clustered exception sequence: 4 consecutive breaches
    clustered = np.zeros(250)
    clustered[50:54] = 1  # 4 clustered breaches in a row

    stat_clustered, p_clustered = christoffersen_independence_test(clustered)
    # Should reject independence (low p-value < 0.05)
    assert p_clustered < 0.05

    # Evenly spaced exception sequence: 4 isolated breaches
    spaced = np.zeros(250)
    spaced[20] = 1
    spaced[80] = 1
    spaced[140] = 1
    spaced[200] = 1

    stat_spaced, p_spaced = christoffersen_independence_test(spaced)
    # Should accept independence (high p-value > 0.05)
    assert p_spaced > 0.05

    # Edge cases: 0 exceptions or 1 exception should return p-value = 1.0
    stat_zero, p_zero = christoffersen_independence_test(np.zeros(250))
    assert p_zero == 1.0


@pytest.mark.parametrize(
    "n_exceptions,expected_zone",
    [
        (0, "Green"),
        (2, "Green"),
        (4, "Green"),
        (5, "Amber"),
        (7, "Amber"),
        (9, "Amber"),
        (10, "Red"),
        (15, "Red"),
    ],
)
def test_basel_traffic_light_parameterized(n_exceptions, expected_zone):
    """Verify Basel traffic light categorization at 250 observations."""
    assert classify_traffic_light(n_exceptions, n_obs=250, alpha=0.99) == expected_zone


def test_backtest_serialization_and_disk_cache(tmp_path):
    """Verify BacktestResults to_dict and from_dict serialization and disk caching."""
    df_returns = generate_synthetic_returns(n_days=70, seed=42)
    cache_file = tmp_path / "test_backtest_cache.json"

    res = run_rolling_backtest(
        df_returns,
        window_size=50,
        copula_family="gaussian_var",
        cache_path=cache_file,
        use_cache=True,
    )

    assert cache_file.exists()

    # Load from disk
    cached_res = run_rolling_backtest(
        df_returns,
        window_size=50,
        copula_family="gaussian_var",
        cache_path=cache_file,
        use_cache=True,
    )

    np.testing.assert_allclose(res.var_forecasts, cached_res.var_forecasts)
    assert res.traffic_light == cached_res.traffic_light
    assert res.n_exceptions == cached_res.n_exceptions

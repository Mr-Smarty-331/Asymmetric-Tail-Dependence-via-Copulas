"""Tests for Univariate Marginals: Student-t MLE fit recovery, PIT, round-trip, edge cases, and KS diagnostics."""

import logging
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data_loader import fetch_market_data
from src.marginals import (
    MarginalFit,
    fit_all_marginals,
    fit_marginal_t,
    inverse_pit,
    pit_transform,
)


def test_student_t_parameter_recovery_on_synthetic_data():
    """Verify that MLE estimation recovers known df, loc, and scale parameters on synthetic Student-t samples (n=5,000)."""
    true_df = 5.0
    true_loc = 0.0008
    true_scale = 0.015
    n_samples = 5000

    rng = np.random.default_rng(123)
    synthetic_samples = stats.t.rvs(
        df=true_df,
        loc=true_loc,
        scale=true_scale,
        size=n_samples,
        random_state=rng,
    )

    fit = fit_marginal_t(synthetic_samples, max_df=100.0)

    # Relative tolerance on df within 25% (typical MLE variance for df=5 on 5k obs)
    assert np.isclose(fit.df, true_df, rtol=0.25)
    # Absolute tolerance on loc
    assert np.isclose(fit.loc, true_loc, atol=0.002)
    # Relative tolerance on scale
    assert np.isclose(fit.scale, true_scale, rtol=0.10)
    # KS test p-value on true synthetic data should fail to reject H0 (p > 0.01)
    assert fit.ks_pvalue > 0.01


def test_pit_strictly_in_open_unit_interval():
    """Verify that PIT transforms all returns strictly into (0, 1), with no exact 0.0 or 1.0."""
    fit = MarginalFit(df=4.5, loc=0.0, scale=0.01, ks_stat=0.02, ks_pvalue=0.5)
    extreme_values = np.array([-1e9, -100.0, -0.05, 0.0, 0.05, 100.0, 1e9])

    u = pit_transform(extreme_values, fit)

    assert (u > 0.0).all()
    assert (u < 1.0).all()
    assert np.all(np.isfinite(u))
    # Extreme tails clipped to eps boundaries
    assert u[0] == 1e-12
    assert u[-1] == 1.0 - 1e-12


def test_pit_round_trip_identity():
    """Verify ppf(cdf(x)) ≈ x round-trip identity across a broad range of return values."""
    fit = MarginalFit(df=6.0, loc=0.0005, scale=0.012, ks_stat=0.01, ks_pvalue=0.8)
    original_returns = np.linspace(-0.08, 0.08, 100)

    u = pit_transform(original_returns, fit)
    reconstructed_returns = inverse_pit(u, fit)

    np.testing.assert_allclose(reconstructed_returns, original_returns, atol=1e-7)


def test_handles_short_or_constant_series_gracefully():
    """Verify clear ValueErrors are raised for insufficient observations or zero variance."""
    short_series = pd.Series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError, match="Insufficient observations"):
        fit_marginal_t(short_series)

    constant_series = pd.Series([0.01] * 100)
    with pytest.raises(ValueError, match="constant or zero-variance"):
        fit_marginal_t(constant_series)


def test_real_data_ks_diagnostics_logged(caplog):
    """Verify fitting marginals on real historical market data executes KS test and logs results."""
    caplog.set_level(logging.INFO)
    returns = fetch_market_data(
        tickers=["SPY", "QQQ", "TLT"],
        cache_dir="tests/fixtures",
        use_cache=True,
    )

    fits = fit_all_marginals(returns, log_diagnostics=True)

    assert len(fits) == 3
    for ticker, fit in fits.items():
        assert 2.0 <= fit.df <= 100.0
        assert fit.scale > 0.0
        assert 0.0 <= fit.ks_stat <= 1.0
        assert 0.0 <= fit.ks_pvalue <= 1.0

    # Verify logging output captured
    log_messages = caplog.text
    assert "Marginal Fit [SPY]" in log_messages
    assert "KS stat=" in log_messages

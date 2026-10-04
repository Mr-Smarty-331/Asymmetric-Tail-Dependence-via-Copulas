"""End-to-end smoke test verifying complete pipeline execution on synthetic data."""

import pytest
import numpy as np
import pandas as pd

from tests.synthetic import generate_synthetic_returns
from src.data_loader import compute_log_returns, clean_market_data
from src.marginals import fit_marginal_t, pit_transform, inverse_pit
from src.copula import fit_copula, calculate_portfolio_risk, sample_copula
from src.backtester import run_rolling_backtest, kupiec_pof_test, christoffersen_independence_test
from src.risk_reporter import MockLLMClient, generate_validation_report, calculate_rniv


def test_smoke_end_to_end_pipeline():
    """Gate 0 Smoke Test: Runs data -> marginals -> copula -> backtest -> report -> dashboard import on 300 days of synthetic data."""
    # 1. Synthetic Data Generation (300 days, 3 assets)
    n_days = 300
    n_assets = 3
    df_returns = generate_synthetic_returns(n_days=n_days, n_assets=n_assets, seed=42)
    assert len(df_returns) == n_days
    assert df_returns.shape[1] == n_assets
    assert not df_returns.isna().any().any()

    # 2. Data Cleaning & Log Returns Sanity
    cleaned = clean_market_data(df_returns)
    assert len(cleaned) == n_days

    # 3. Marginals Fitting & PIT Transform
    marginal_fits = []
    u_data = np.zeros_like(cleaned.values)
    for i, col in enumerate(cleaned.columns):
        fit = fit_marginal_t(cleaned[col])
        marginal_fits.append(fit)
        assert 2.0 <= fit.df <= 100.0
        u_data[:, i] = pit_transform(cleaned[col], fit)

    assert (u_data > 0.0).all() and (u_data < 1.0).all()

    # 4. Copula Fitting & Sampling
    copula_fit = fit_copula(u_data, family="t")
    assert copula_fit.family == "t"
    assert copula_fit.corr_matrix.shape == (n_assets, n_assets)
    np.testing.assert_allclose(np.diag(copula_fit.corr_matrix), 1.0, atol=1e-5)

    u_samples = sample_copula(copula_fit, n_samples=1000, seed=42)
    assert u_samples.shape == (1000, n_assets)

    weights = np.array([0.4, 0.3, 0.3])
    var_99, es_99 = calculate_portfolio_risk(weights, copula_fit, marginal_fits, alpha=0.99, n_samples=1000, seed=42)
    assert es_99 >= var_99
    assert var_99 > 0.0

    # 5. Rolling Backtest
    backtest_results = run_rolling_backtest(
        cleaned,
        weights=weights,
        window_size=200,
        alpha=0.99,
        copula_family="t",
        n_mc_samples=1000,
    )
    assert backtest_results.n_obs == 100
    assert 0 <= backtest_results.n_exceptions <= 100
    assert 0.0 <= backtest_results.kupiec_pvalue <= 1.0
    assert 0.0 <= backtest_results.christoffersen_pvalue <= 1.0
    assert backtest_results.traffic_light in ["Green", "Amber", "Red"]

    # 6. Autonomous Model Validation Reporting with Mock LLM
    mock_llm = MockLLMClient()
    report = generate_validation_report(backtest_results, model_name="Student-t Copula VaR", llm_client=mock_llm)
    assert report.model_name == "Student-t Copula VaR"
    assert report.traffic_light_status == backtest_results.traffic_light
    assert report.sign_off_status in ["APPROVED", "CONDITIONAL", "REJECTED"]
    assert len(report.executive_summary) > 0
    assert report.rniv_capital_addon_bps >= 0.0

    # 7. Dashboard Import & Functionality
    import src.app as app
    assert hasattr(app, "render_dashboard")

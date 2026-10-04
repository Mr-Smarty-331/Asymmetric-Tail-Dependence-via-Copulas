"""Comprehensive Unit and Quantitative Statistical Tests for Copula Engine."""

import numpy as np
import pytest
from scipy import stats

from src.copula import (
    CopulaFit,
    calculate_portfolio_risk,
    calculate_t_copula_tail_dependence,
    estimate_t_copula_df,
    fit_copula,
    invert_tau_to_linear_corr,
    kendall_tau_matrix,
    repair_to_nearest_psd,
    sample_copula,
)
from src.data_loader import fetch_market_data
from src.marginals import MarginalFit, fit_all_marginals, pit_transform
from tests.synthetic import generate_synthetic_returns


def test_tau_inversion_known_bivariate_case():
    """Verify Kendall's tau inversion to linear correlation: R = sin(pi * tau / 2)."""
    # For tau = 0.5: R = sin(pi * 0.5 / 2) = sin(pi/4) = 1/sqrt(2) ~= 0.70710678
    tau_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    r_matrix = invert_tau_to_linear_corr(tau_matrix)

    expected_rho = np.sin(np.pi * 0.5 / 2.0)
    assert np.isclose(r_matrix[0, 1], expected_rho, atol=1e-7)
    assert np.isclose(r_matrix[1, 0], expected_rho, atol=1e-7)
    assert np.isclose(r_matrix[0, 0], 1.0)
    assert np.isclose(r_matrix[1, 1], 1.0)


def test_repaired_matrix_is_symmetric_unit_diagonal_and_psd():
    """Verify repair_to_nearest_psd produces a symmetric matrix with unit diagonal and non-negative eigenvalues."""
    # Construct an indefinite / non-PSD symmetric matrix
    non_psd = np.array([
        [1.0, 0.9, 0.9],
        [0.9, 1.0, -0.9],
        [0.9, -0.9, 1.0],
    ])
    # Verify original has a negative eigenvalue
    orig_eigvals = np.linalg.eigvalsh(non_psd)
    assert np.any(orig_eigvals < 0)

    repaired = repair_to_nearest_psd(non_psd, eps=1e-5)

    # 1. Symmetry
    np.testing.assert_allclose(repaired, repaired.T, atol=1e-7)
    # 2. Unit diagonal
    np.testing.assert_allclose(np.diag(repaired), 1.0, atol=1e-7)
    # 3. Positive semi-definite (all eigenvalues >= 0)
    repaired_eigvals = np.linalg.eigvalsh(repaired)
    assert np.all(repaired_eigvals >= 0.0)


def test_gaussian_copula_sample_correlation_matches_target():
    """Verify sample correlation of Gaussian copula simulated uniforms inverted to standard normals matches target R."""
    target_r = np.array([[1.0, 0.6], [0.6, 1.0]])
    copula = CopulaFit(family="gaussian", corr_matrix=target_r)

    u_samples = sample_copula(copula, n_samples=30000, seed=123)
    z_samples = stats.norm.ppf(u_samples)
    sample_corr = np.corrcoef(z_samples, rowvar=False)

    np.testing.assert_allclose(sample_corr[0, 1], target_r[0, 1], atol=0.02)


def test_t_copula_sample_kendall_tau_matches_input():
    """Verify sample Kendall's tau of Student-t copula matches input tau."""
    target_rho = 0.5
    target_r = np.array([[1.0, target_rho], [target_rho, 1.0]])
    copula = CopulaFit(family="t", corr_matrix=target_r, df=5.0)

    u_samples = sample_copula(copula, n_samples=25000, seed=42)
    sample_tau, _ = stats.kendalltau(u_samples[:, 0], u_samples[:, 1])

    expected_tau = (2.0 / np.pi) * np.arcsin(target_rho)
    assert np.isclose(sample_tau, expected_tau, atol=0.03)


def test_tail_dependence_frequency_t_vs_gaussian():
    """Verify empirical lower joint-tail frequency P(U2 < q | U1 < q) matches analytic lambda for t-copula, and is higher than Gaussian."""
    rho = 0.7
    nu = 4.0
    r_mat = np.array([[1.0, rho], [rho, 1.0]])
    analytic_lambda = calculate_t_copula_tail_dependence(nu=nu, rho=rho)

    t_copula = CopulaFit(family="t", corr_matrix=r_mat, df=nu)
    gauss_copula = CopulaFit(family="gaussian", corr_matrix=r_mat)

    n_mc = 100000
    u_t = sample_copula(t_copula, n_samples=n_mc, seed=101)
    u_g = sample_copula(gauss_copula, n_samples=n_mc, seed=101)

    # Tail quantile threshold q = 0.01 (1% joint tail)
    q = 0.01
    p_joint_t = np.mean((u_t[:, 0] < q) & (u_t[:, 1] < q)) / q
    p_joint_g = np.mean((u_g[:, 0] < q) & (u_g[:, 1] < q)) / q

    # t-copula empirical joint tail matches analytic lambda within sampling error
    assert np.isclose(p_joint_t, analytic_lambda, atol=0.08)
    # Student-t joint tail frequency is significantly higher than Gaussian
    assert p_joint_t > p_joint_g * 1.3


def test_t_copula_nu_mle_recovers_known_nu():
    """Verify profile log-likelihood MLE recovers known degrees of freedom nu on synthetic copula data."""
    true_nu = 4.0
    df_returns = generate_synthetic_returns(n_days=1500, copula_type="t", copula_df=true_nu, seed=99)

    # PIT to uniforms
    u_mat = np.zeros_like(df_returns.values)
    for i, col in enumerate(df_returns.columns):
        u_mat[:, i] = stats.t.cdf(df_returns[col], df=5.0 + i * 2.0, loc=0.0005, scale=0.015)

    tau_mat = kendall_tau_matrix(u_mat)
    r_mat = repair_to_nearest_psd(invert_tau_to_linear_corr(tau_mat))

    estimated_nu = estimate_t_copula_df(u_mat, r_mat)
    # Tolerance within 35% on 1500 observations
    assert np.isclose(estimated_nu, true_nu, rtol=0.35)


def test_mc_var_gaussian_copula_matches_analytic_variance_covariance():
    """Verify Monte Carlo VaR with normal marginals & Gaussian copula matches analytic variance-covariance VaR within ~2%."""
    # Under Gaussian assumptions: Portfolio return ~ N(w^T mu, w^T Sigma w)
    weights = np.array([0.5, 0.5])
    mu = np.array([0.0005, 0.0005])
    sigma = np.array([0.015, 0.020])
    rho = 0.6

    cov_matrix = np.array([
        [sigma[0]**2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1]**2],
    ])
    port_var = weights @ cov_matrix @ weights
    port_std = np.sqrt(port_var)
    port_mean = weights @ mu

    alpha = 0.99
    # Loss = -R, so VaR_alpha = - (port_mean - z_alpha * port_std)
    z_alpha = stats.norm.ppf(alpha)
    analytic_var = -(port_mean - z_alpha * port_std)

    # Use large df (e.g. 100) to emulate normal marginals
    marginal_fits = [
        MarginalFit(df=100.0, loc=mu[0], scale=sigma[0], ks_stat=0.01, ks_pvalue=0.9),
        MarginalFit(df=100.0, loc=mu[1], scale=sigma[1], ks_stat=0.01, ks_pvalue=0.9),
    ]
    r_mat = np.array([[1.0, rho], [rho, 1.0]])
    copula = CopulaFit(family="gaussian", corr_matrix=r_mat)

    mc_var, mc_es = calculate_portfolio_risk(
        weights=weights,
        copula=copula,
        marginal_fits=marginal_fits,
        alpha=alpha,
        n_samples=50000,
        seed=42,
    )

    # MC VaR must match analytic within 2.5%
    rel_error = abs(mc_var - analytic_var) / analytic_var
    assert rel_error < 0.025


def test_es_greater_than_var_and_var_monotonic_in_alpha():
    """Verify Expected Shortfall is strictly >= VaR and VaR is monotonic in alpha."""
    marginal_fits = [
        MarginalFit(df=4.0, loc=0.0002, scale=0.015, ks_stat=0.01, ks_pvalue=0.8),
        MarginalFit(df=5.0, loc=0.0002, scale=0.018, ks_stat=0.01, ks_pvalue=0.8),
    ]
    copula = CopulaFit(family="t", corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]), df=4.0)
    weights = np.array([0.5, 0.5])

    var_95, es_95 = calculate_portfolio_risk(weights, copula, marginal_fits, alpha=0.95, n_samples=20000, seed=42)
    var_99, es_99 = calculate_portfolio_risk(weights, copula, marginal_fits, alpha=0.99, n_samples=20000, seed=42)

    assert es_95 >= var_95
    assert es_99 >= var_99
    assert var_99 > var_95
    assert es_99 > es_95


def test_seed_determinism_and_mc_standard_error():
    """Verify seed determinism and that MC standard error across seeds is small at N=10,000."""
    marginal_fits = [
        MarginalFit(df=4.0, loc=0.0002, scale=0.015, ks_stat=0.01, ks_pvalue=0.8),
        MarginalFit(df=4.0, loc=0.0002, scale=0.015, ks_stat=0.01, ks_pvalue=0.8),
    ]
    copula = CopulaFit(family="t", corr_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]), df=4.0)
    weights = np.array([0.5, 0.5])

    # Same seed yields exact same result
    var_1, es_1 = calculate_portfolio_risk(weights, copula, marginal_fits, alpha=0.99, n_samples=10000, seed=777)
    var_2, es_2 = calculate_portfolio_risk(weights, copula, marginal_fits, alpha=0.99, n_samples=10000, seed=777)
    assert var_1 == var_2
    assert es_1 == es_2

    # Standard error across 5 different seeds is small (< 5% of VaR)
    vars_across_seeds = [
        calculate_portfolio_risk(weights, copula, marginal_fits, alpha=0.99, n_samples=10000, seed=1000 + i)[0]
        for i in range(5)
    ]
    std_err = np.std(vars_across_seeds)
    assert std_err / np.mean(vars_across_seeds) < 0.05


def test_real_data_single_day_var_es_t_vs_gaussian():
    """Gate 3 verification: Single-day VaR/ES comparison on real historical market data confirms Student-t Copula model > Gaussian Benchmark in the tail."""
    returns = fetch_market_data(tickers=["SPY", "QQQ", "TLT"], cache_dir="tests/fixtures", use_cache=True)
    marginal_fits = [fit_all_marginals(returns)[col] for col in returns.columns]

    u_mat = np.zeros_like(returns.values)
    for i, col in enumerate(returns.columns):
        u_mat[:, i] = pit_transform(returns[col], marginal_fits[i])

    copula_t = fit_copula(u_mat, family="t")
    copula_g = fit_copula(u_mat, family="gaussian")

    # Gaussian benchmark marginals (df=100)
    gaussian_marginal_fits = [
        MarginalFit(df=100.0, loc=fit.loc, scale=fit.scale, ks_stat=0.01, ks_pvalue=0.9)
        for fit in marginal_fits
    ]

    weights = np.array([0.4, 0.4, 0.2])
    var_t, es_t = calculate_portfolio_risk(weights, copula_t, marginal_fits, alpha=0.99, n_samples=25000, seed=42)
    var_benchmark, es_benchmark = calculate_portfolio_risk(
        weights, copula_g, gaussian_marginal_fits, alpha=0.99, n_samples=25000, seed=42
    )

    # Student-t model captures fat tails, producing significantly higher 99% VaR and ES than Gaussian benchmark
    assert var_t > var_benchmark * 1.15
    assert es_t > es_benchmark * 1.15

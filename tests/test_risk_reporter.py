"""Tests for Autonomous Model Validation Reporter, RNIV Rules, Payload Builder, and Narrative Grounding."""

import json
import os
import numpy as np
import pandas as pd
import pytest

from src.backtester import BacktestResults
from src.risk_reporter import (
    MockLLMClient,
    ModelValidationReport,
    RealLLMClient,
    build_validation_payload,
    calculate_rniv,
    determine_sign_off_status,
    generate_validation_report,
    verify_grounding,
)


def create_sample_backtest_results(traffic_light: str = "Green", n_exceptions: int = 2, christoffersen_p: float = 0.8) -> BacktestResults:
    """Helper creating BacktestResults with specified metrics."""
    dates = pd.bdate_range("2021-01-01", periods=250)
    losses = np.random.normal(0, 0.01, 250)
    var = np.full(250, 0.025)
    es = np.full(250, 0.035)
    exceptions = np.zeros(250)
    if n_exceptions > 0:
        exceptions[:n_exceptions] = 1

    return BacktestResults(
        dates=dates,
        portfolio_losses=losses,
        var_forecasts=var,
        es_forecasts=es,
        exceptions=exceptions,
        n_exceptions=n_exceptions,
        n_obs=250,
        exception_rate=n_exceptions / 250.0,
        kupiec_stat=0.1,
        kupiec_pvalue=0.75,
        christoffersen_stat=0.05,
        christoffersen_pvalue=christoffersen_p,
        conditional_coverage_stat=0.15,
        conditional_coverage_pvalue=0.80,
        traffic_light=traffic_light,
        model_name="Student-t Copula VaR",
    )


def test_payload_builder_snapshot_stability():
    """Verify quantitative payload builder produces an exact, deterministic dictionary snapshot."""
    results = create_sample_backtest_results(traffic_light="Green", n_exceptions=3)
    payload = build_validation_payload(results, model_name="Student-t Copula VaR")

    assert payload["model_name"] == "Student-t Copula VaR"
    assert payload["n_obs"] == 250
    assert payload["n_exceptions"] == 3
    assert payload["expected_exceptions"] == 2.5
    assert payload["traffic_light"] == "Green"
    assert payload["rniv_bps"] == 0.0
    assert len(payload["breach_dates"]) == 3


@pytest.mark.parametrize(
    "traffic_light,n_exceptions,christoffersen_p,expected_rniv",
    [
        ("Green", 2, 0.80, 0.0),       # Green zone, no clustering
        ("Amber", 6, 0.80, 25.0),      # Amber zone (+25 bps)
        ("Amber", 6, 0.01, 45.0),      # Amber zone (+25 bps) + clustering penalty (+20 bps)
        ("Red", 12, 0.80, 100.0),      # Red zone (+100 bps)
        ("Red", 12, 0.02, 120.0),      # Red zone (+100 bps) + clustering penalty (+20 bps)
    ],
)
def test_rniv_deterministic_rules(traffic_light, n_exceptions, christoffersen_p, expected_rniv):
    """Verify regulatory RNIV capital add-on deterministic rules."""
    results = create_sample_backtest_results(
        traffic_light=traffic_light,
        n_exceptions=n_exceptions,
        christoffersen_p=christoffersen_p,
    )
    rniv = calculate_rniv(results)
    assert rniv == expected_rniv


def test_mock_llm_valid_output_and_report_generation():
    """Verify MockLLMClient produces valid ModelValidationReport passing Pydantic validation."""
    results = create_sample_backtest_results(traffic_light="Green", n_exceptions=2)
    mock_client = MockLLMClient()

    report = generate_validation_report(results, llm_client=mock_client)

    assert isinstance(report, ModelValidationReport)
    assert report.traffic_light_status == "Green"
    assert report.sign_off_status == "APPROVED"
    assert "2 exceptions" in report.executive_summary
    assert report.rniv_capital_addon_bps == 0.0


def test_mock_llm_error_retry_and_graceful_fallback():
    """Verify that when MockLLM simulates failures, generate_validation_report retries and falls back gracefully."""
    results = create_sample_backtest_results(traffic_light="Amber", n_exceptions=6)
    failing_client = MockLLMClient(fail_attempts=5)  # always fails

    # Should fall back cleanly without raising unhandled exception
    report = generate_validation_report(results, llm_client=failing_client)

    assert isinstance(report, ModelValidationReport)
    assert report.sign_off_status == "CONDITIONAL"
    assert report.rniv_capital_addon_bps == 25.0


def test_narrative_grounding_verification():
    """Verify verify_grounding accepts factual reports and detects hallucinated numbers."""
    results = create_sample_backtest_results(traffic_light="Green", n_exceptions=3)
    mock_client = MockLLMClient()
    report = generate_validation_report(results, llm_client=mock_client)

    # Valid report matches quantitative payload
    assert verify_grounding(report) is True

    # Hallucinated report (e.g. claims 19 exceptions when only 3 occurred)
    hallucinated_report = report.model_copy(
        update={"executive_summary": "We detected 19 exceptions during the test period."}
    )
    assert verify_grounding(hallucinated_report) is False


def test_real_llm_missing_api_key_graceful_error():
    """Verify RealLLMClient raises clean descriptive error when API key is missing."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("OPENAI_API_KEY", raising=False)
        mp.delenv("GEMINI_API_KEY", raising=False)
        client = RealLLMClient(api_key=None)

        with pytest.raises(RuntimeError, match="Missing API key"):
            client.generate_narrative("Analyze test data")


@pytest.mark.llm
def test_live_llm_call_if_credentials_present():
    """Live LLM call verifying real API connectivity and grounding (manual only)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No OPENAI_API_KEY present in environment.")

    client = RealLLMClient(api_key=api_key)
    results = create_sample_backtest_results(traffic_light="Green", n_exceptions=2)
    report = generate_validation_report(results, llm_client=client)

    assert isinstance(report, ModelValidationReport)
    assert len(report.executive_summary) > 20
    assert verify_grounding(report) is True

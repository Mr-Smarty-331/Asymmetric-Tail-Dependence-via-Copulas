"""Tests for Streamlit Dashboard, Plotly visualization, PDF generation, and AppTest execution."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.app import build_risk_chart, generate_pdf_report, load_or_run_backtest
from src.backtester import BacktestResults
from src.risk_reporter import MockLLMClient, generate_validation_report


def test_build_risk_chart_traces_and_breach_markers():
    """Verify Plotly risk chart has all 4 traces and breach markers match exception count."""
    dates = pd.bdate_range("2021-01-01", periods=100)
    losses = np.random.normal(0, 0.01, 100)
    var = np.full(100, 0.02)
    es = np.full(100, 0.03)
    exceptions = np.zeros(100)
    exceptions[10] = 1
    exceptions[45] = 1
    exceptions[80] = 1

    results = BacktestResults(
        dates=dates,
        portfolio_losses=losses,
        var_forecasts=var,
        es_forecasts=es,
        exceptions=exceptions,
        n_exceptions=3,
        n_obs=100,
        exception_rate=0.03,
        kupiec_stat=0.5,
        kupiec_pvalue=0.48,
        christoffersen_stat=0.1,
        christoffersen_pvalue=0.75,
        conditional_coverage_stat=0.6,
        conditional_coverage_pvalue=0.74,
        traffic_light="Green",
        model_name="Student-t Copula",
    )

    fig = build_risk_chart(results)

    # 4 traces: Realized Loss, VaR, ES, Breaches
    assert len(fig.data) == 4
    trace_names = [t.name for t in fig.data]
    assert any("Realized Daily Loss" in name for name in trace_names)
    assert any("VaR" in name for name in trace_names)
    assert any("ES" in name for name in trace_names)
    assert any("VaR Breaches (3)" in name for name in trace_names)

    # Check breach marker count
    breach_trace = [t for t in fig.data if "VaR Breaches" in t.name][0]
    assert len(breach_trace.x) == 3


def test_pdf_report_generation():
    """Verify generate_pdf_report produces valid PDF bytes."""
    dates = pd.bdate_range("2021-01-01", periods=50)
    results = BacktestResults(
        dates=dates,
        portfolio_losses=np.zeros(50),
        var_forecasts=np.full(50, 0.02),
        es_forecasts=np.full(50, 0.03),
        exceptions=np.zeros(50),
        n_exceptions=0,
        n_obs=50,
        exception_rate=0.0,
        kupiec_stat=1.0,
        kupiec_pvalue=0.31,
        christoffersen_stat=0.0,
        christoffersen_pvalue=1.0,
        conditional_coverage_stat=1.0,
        conditional_coverage_pvalue=0.6,
        traffic_light="Green",
        model_name="Student-t Copula",
    )
    report = generate_validation_report(results, llm_client=MockLLMClient())
    pdf_bytes = generate_pdf_report(report)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_load_or_run_backtest_cached():
    """Verify load_or_run_backtest returns populated BacktestResults and returns DataFrame."""
    results, df = load_or_run_backtest(model_choice="t", alpha=0.99, window_size=252)
    assert isinstance(results, BacktestResults)
    assert results.n_obs > 200
    assert len(df) > 500


def test_streamlit_apptest_headless_execution():
    """Verify Streamlit AppTest executes the entire dashboard headlessly without exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception
    # Check title is rendered
    assert len(at.title) > 0
    assert "Institutional Risk Methodologies Group" in at.title[0].value

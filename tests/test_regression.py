"""Golden Regression Test: Guard against silent math regressions or calibration drifts."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.backtester import run_rolling_backtest
from tests.synthetic import generate_synthetic_returns


def test_golden_regression_backtest_metrics():
    """Verify that a fixed synthetic seed and parameter snapshot reproduces exact metrics matching the golden fixture JSON."""
    golden_path = Path(__file__).parent / "fixtures" / "golden_backtest_metrics.json"
    assert golden_path.exists(), "Golden fixture JSON missing."

    with open(golden_path, "r") as f:
        golden = json.load(f)

    # Re-run identical setup
    df = generate_synthetic_returns(n_days=300, seed=12345)
    res = run_rolling_backtest(df, window_size=200, copula_family="t", n_mc_samples=2000)

    assert res.n_obs == golden["n_obs"]
    assert res.n_exceptions == golden["n_exceptions"]
    assert res.traffic_light == golden["traffic_light"]
    assert np.isclose(res.exception_rate, golden["exception_rate"], atol=1e-5)
    assert np.isclose(res.kupiec_pvalue, golden["kupiec_pvalue"], atol=1e-4)
    assert np.isclose(res.christoffersen_pvalue, golden["christoffersen_pvalue"], atol=1e-4)
    assert np.isclose(float(res.var_forecasts.mean()), golden["mean_var"], atol=1e-4)
    assert np.isclose(float(res.es_forecasts.mean()), golden["mean_es"], atol=1e-4)

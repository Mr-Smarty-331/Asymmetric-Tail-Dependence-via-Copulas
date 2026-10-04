"""Tests for Data Layer: log-returns, forward-fill, caching, monotonicity, and offline fixtures."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    clean_market_data,
    compute_log_returns,
    fetch_market_data,
    forward_fill_prices,
)


def test_log_returns_match_hand_calculated_example():
    """Verify log returns match analytical formula: r_t = ln(P_t / P_{t-1})."""
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    # P0 = 100.0, P1 = 105.0, P2 = 99.75
    # r1 = ln(105/100) = ln(1.05) ~= 0.04879016
    # r2 = ln(99.75/105) = ln(0.95) ~= -0.05129329
    prices = pd.DataFrame({"AAPL": [100.0, 105.0, 99.75]}, index=dates)

    returns = compute_log_returns(prices)

    expected_r1 = np.log(105.0 / 100.0)
    expected_r2 = np.log(99.75 / 105.0)

    assert len(returns) == 2
    assert np.isclose(returns.loc["2020-01-02", "AAPL"], expected_r1, atol=1e-7)
    assert np.isclose(returns.loc["2020-01-03", "AAPL"], expected_r2, atol=1e-7)


def test_forward_fill_preserves_leading_nans_and_fills_gaps():
    """Verify forward-fill fills interior missing data but preserves leading NaNs."""
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    raw_prices = pd.DataFrame(
        {
            "A": [np.nan, 100.0, np.nan, 105.0],  # Leading NaN at index 0, interior gap at index 2
            "B": [50.0, np.nan, 52.0, np.nan],    # Interior gaps at indices 1 and 3
        },
        index=dates,
    )

    ffilled = forward_fill_prices(raw_prices)

    # Leading NaN in column A must still be NaN
    assert np.isnan(ffilled.loc["2020-01-01", "A"])
    # Interior gap in A filled with 100.0
    assert ffilled.loc["2020-01-03", "A"] == 100.0
    # Interior gaps in B filled
    assert ffilled.loc["2020-01-02", "B"] == 50.0
    assert ffilled.loc["2020-01-04", "B"] == 52.0


def test_data_cleaning_guarantees_no_nans_inf_and_monotonic_index():
    """Verify clean_market_data drops leading NaNs, removes inf, and enforces monotonic dates."""
    dates = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-04"])
    dirty_returns = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.02, np.inf],
            "B": [0.02, 0.01, -0.01, 0.03],
        },
        index=dates,
    )

    cleaned = clean_market_data(dirty_returns)

    assert not cleaned.isna().any().any()
    assert not np.isinf(cleaned.values).any()
    assert cleaned.index.is_monotonic_increasing
    assert len(cleaned) > 0


def test_cache_hit_avoids_network_call(tmp_path):
    """Verify that cached data is loaded directly without calling yfinance download."""
    dates = pd.bdate_range("2019-01-01", "2020-01-01")
    cached_df = pd.DataFrame({"SPY": np.random.normal(0, 0.01, len(dates))}, index=dates)
    cache_file = tmp_path / "returns_SPY_2019-01-01_2020-01-01.parquet"
    cached_df.to_parquet(cache_file)

    with patch("yfinance.download") as mock_yf:
        result = fetch_market_data(
            tickers=["SPY"],
            start_date="2019-01-01",
            end_date="2020-01-01",
            cache_dir=tmp_path,
            use_cache=True,
            force_download=False,
        )
        # yfinance download should NOT have been called
        mock_yf.assert_not_called()
        assert len(result) == len(cached_df)
        assert list(result.columns) == ["SPY"]


def test_offline_fixture_loads_successfully_by_default():
    """Verify offline fixture committed in tests/fixtures is loaded cleanly."""
    returns = fetch_market_data(
        tickers=["SPY", "QQQ", "TLT"],
        start_date="2019-01-01",
        end_date="2023-01-01",
        cache_dir="tests/fixtures",
        use_cache=True,
        force_download=False,
    )
    assert len(returns) >= 1000
    assert set(returns.columns) == {"SPY", "QQQ", "TLT"}
    assert returns.index.is_monotonic_increasing
    assert not returns.isna().any().any()


@pytest.mark.network
def test_real_yfinance_download():
    """Network-marked test verifying live yfinance connectivity."""
    df = fetch_market_data(
        tickers=["SPY"],
        start_date="2022-01-01",
        end_date="2022-02-01",
        cache_dir="tests/fixtures",
        force_download=True,
    )
    assert len(df) > 15
    assert "SPY" in df.columns

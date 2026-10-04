"""Market data ingestion, cleaning, log-return transformation, and on-disk caching."""

import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute continuously compounded log returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Log returns with the first row (all NaNs) dropped.
    """
    if prices.empty:
        return pd.DataFrame()
    # Sort index monotonically ascending
    sorted_prices = prices.sort_index()
    log_returns = np.log(sorted_prices / sorted_prices.shift(1))
    return log_returns.dropna(how="all")


def forward_fill_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill internal missing price data without backfilling leading NaNs.

    Parameters
    ----------
    prices : pd.DataFrame
        Raw prices matrix.

    Returns
    -------
    pd.DataFrame
        Forward-filled prices with preserved leading NaNs if series started late.
    """
    return prices.ffill()


def clean_market_data(returns: pd.DataFrame) -> pd.DataFrame:
    """Clean log returns DataFrame:

    - Forward-fill internal gaps (if any intermediate log return was NaN)
    - Drop any rows with remaining NaNs (e.g. before all assets have active data)
    - Remove infinities if present
    - Ensure index is monotonically increasing and unique.
    """
    if returns.empty:
        return returns

    cleaned = returns.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
    # Replace infinities
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    # Forward fill then drop remaining NaNs
    cleaned = cleaned.ffill().dropna()
    return cleaned


def fetch_market_data(
    tickers: Optional[List[str]] = None,
    start_date: str = "2019-01-01",
    end_date: str = "2023-01-01",
    cache_dir: Union[str, Path] = "tests/fixtures",
    use_cache: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch adjusted close prices for tickers, cache on disk as parquet/csv, and return cleaned log returns.

    Parameters
    ----------
    tickers : Optional[List[str]]
        List of tickers. Defaults to ['SPY', 'QQQ', 'TLT'].
    start_date : str
        Start date string (YYYY-MM-DD). Default '2019-01-01' gives 252-day lookback for 2020.
    end_date : str
        End date string (YYYY-MM-DD). Default '2023-01-01'.
    cache_dir : Union[str, Path]
        Directory path for caching. Defaults to 'tests/fixtures'.
    use_cache : bool
        Whether to check and load from on-disk cache.
    force_download : bool
        Whether to bypass cache and force a new yfinance download.

    Returns
    -------
    pd.DataFrame
        Cleaned daily log returns with DatetimeIndex.
    """
    if tickers is None:
        tickers = ["SPY", "QQQ", "TLT"]

    sorted_tickers = sorted(tickers)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"returns_{'_'.join(sorted_tickers)}_{start_date}_{end_date}.parquet"
    csv_fallback = cache_path / f"returns_{'_'.join(sorted_tickers)}_{start_date}_{end_date}.csv"

    # 1. Try loading from cache if permitted
    if use_cache and not force_download:
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index)
            return clean_market_data(df)
        elif csv_fallback.exists():
            df = pd.read_csv(csv_fallback, index_col=0, parse_dates=True)
            return clean_market_data(df)

    # 2. Download via yfinance with auto_adjust=True
    import yfinance as yf
    raw = yf.download(
        tickers=sorted_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No price data returned for tickers: {sorted_tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.levels[0]:
            prices = raw["Close"]
        else:
            prices = raw.xs(raw.columns.levels[0][0], axis=1)
    else:
        prices = raw

    prices = prices[sorted_tickers]
    prices = forward_fill_prices(prices)
    log_returns = compute_log_returns(prices)
    cleaned_returns = clean_market_data(log_returns)

    # 3. Cache to disk
    try:
        cleaned_returns.to_parquet(cache_file)
    except Exception:
        cleaned_returns.to_csv(csv_fallback)

    return cleaned_returns

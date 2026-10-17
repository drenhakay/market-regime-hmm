from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# AbsReturn dropped: it is a function of Return, so the full covariance goes singular
FEATURE_COLS: list[str] = ['Return', 'Vol20', 'Range']


# end is exclusive in yfinance
def load_prices(ticker: str = "^GSPC", start: str = "2000-01-01", end: str = "2026-09-10") -> pd.DataFrame:
    """
    Download daily OHLC (auto-adjusted) from yfinance; cache to data/ as parquet.
    """
    cache = DATA_DIR / f"{ticker}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close']]
    df = df.dropna()
    df.to_parquet(cache)
    return df



def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Causal daily feature frame indexed by date, columns == FEATURE_COLS.
    """
    close = prices['Close']
    log_ret = np.log(close).diff() * 100
    vol_20 = log_ret.rolling(20).std()
    log_range = np.log(prices['High'] / prices['Low']) * 100
    features = pd.DataFrame({
        'Return': log_ret,
        'Vol20': vol_20,
        'Range': log_range,
    })
    features = features.dropna()
    return features



def emissions(prices: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex, pd.Series]:
    """
    Feature matrix X, its dates and next_day_ret, aligned so row i is known at the close of
    dates[i] and next_day_ret[i] is the simple return from close[i] to close[i+1].
    """
    close = prices['Close']
    df = build_features(prices)
    # simple return, since the backtest compounds with (1 + r).cumprod()
    nxt = close.pct_change().shift(-1)
    combined = df.join(nxt.rename('next_ret')).dropna()
    X = combined[FEATURE_COLS].to_numpy()
    dates = combined.index
    next_day_ret = combined['next_ret']
    return (X, dates, next_day_ret)

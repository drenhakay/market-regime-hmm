from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_prices, build_features, FEATURE_COLS


def check_build_features() -> None:
    prices = load_prices()
    feats = build_features(prices)

    # columns are exactly the declared feature set, in the declared order
    assert list(feats.columns) == FEATURE_COLS, feats.columns.tolist()

    # no NaNs left anywhere
    assert feats.isna().sum().sum() == 0

    # rows lost are the warm-up for the 20d vol window (~20)
    print("prices start:", prices.index.min())
    print("feats  start:", feats.index.min())
    print("rows lost   :", len(prices) - len(feats))

    # index stays a clean, sorted, gap-free subset of the price index
    assert feats.index.is_monotonic_increasing
    assert not feats.index.has_duplicates
    assert feats.index.isin(prices.index).all()

    # Vol20 and Range are strictly positive
    assert (feats[["Vol20", "Range"]] > 0).all().all()

    # returns are in %, so daily std is around 1
    print(feats.describe())

    # Vol20 and Range spike in 2008 and Mar 2020
    feats.plot(subplots=True, figsize=(11, 8), title="features")
    plt.tight_layout()
    plt.show()

def check_emissions() -> None:
    from src.data import emissions
    prices = load_prices()
    feats = build_features(prices)
    X, dates, nxt = emissions(prices)

    assert len(X) == len(dates) == len(nxt)
    assert not np.isnan(X).any()
    assert nxt.notna().all()
    assert dates.equals(nxt.index)
    # emissions keeps every feature row that has a valid next-day return,
    # i.e. exactly one fewer than the feature frame (the last row is dropped)
    assert len(nxt) == len(feats) - 1
    assert X.shape[1] == len(FEATURE_COLS)
    print("emissions:", X.shape, dates.min().date(), "->", dates.max().date())

def check_buy_and_hold_matches_price() -> None:
    from src.backtest import baselines
    from src.data import emissions
    prices = load_prices()
    _, dates, nxt = emissions(prices)
    equity = baselines(nxt)["buy_and_hold"]["equity"]
    # equity after row i's next-day return is close[i+1] / close[first row]
    close = prices["Close"]
    expected = close.shift(-1).loc[dates] / close.loc[dates[0]]
    assert np.allclose(equity, expected, rtol=1e-9, atol=0), "buy & hold equity drifts from the price ratio"
    print("buy & hold equity matches the price ratio")


if __name__ == "__main__":
    #check_build_features()
    check_emissions()
    check_buy_and_hold_matches_price()

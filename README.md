# Market Regime Detection with a Hidden Markov Model

This project fits a Hidden Markov Model to daily S&P 500 returns and reads its latent states as
market regimes (calm / choppy / crisis). It then tests whether scaling market exposure by the
regime inferred out of sample improves risk-adjusted return over buy and hold.

The hidden-state process is a discrete-time Markov chain, so the project also estimates its
transition matrix, stationary distribution and expected regime durations. The backtest uses
expanding-window walk-forward, so each day's regime is inferred from past data only.

## Result

Out of sample on the S&P 500 (2004-01-27 to 2026-09-08, monthly refits, forward-filtered states)
a calm / choppy / crisis exposure overlay (1.0 / 0.5 / 0.0) reaches Sharpe 0.71 and a -14.9% max
drawdown, against 0.54 and -56.8% for buy and hold. It gives up return (6.1% a year vs 8.8%), so
it lowers risk and does not add return.

Against a volatility-target baseline it only ties on Sharpe (0.706 vs 0.707) and trails at every
trading-cost level, because it trades more (6.0x vs 4.2x a year). The drawdown edge holds:
-14.9% vs -43.7%, and 673 vs 873 days underwater. Softer exposure maps recover return but grow
the drawdown by 50-80%; four maps were tried on one out-of-sample path, so that table shows a
range and was not used to pick a map. A returns-only Markov-switching model agrees with the HMM
on 78% of days and never lands more than one regime away.

The same pipeline run unchanged on long-dated Treasuries (TLT, 2006-2026 out of sample) cuts the
2022 rate-rise drawdown (-16.9% vs -41.1%) but loses on Sharpe (0.19 vs 0.27). Bond volatility
rises during flight-to-safety rallies, so the crisis state has a positive mean return there and
the overlay sits out the 2008 and 2020 rallies. A mirrored map (more exposure in crisis), fixed
before the run, does no better than a constant 60% position.

Per-market tables are in `results/` (one CSV per ticker and exposure map). Caveats: leverage in
the volatility target (capped at 1.5x) has no financing cost, cash earns nothing in the overlay,
and Sharpe uses no risk-free rate.

## Layout

```
regime_hmm.ipynb        full pipeline for the ticker set in its settings cell
results/                metrics per ticker and exposure map (written by the notebook)
src/data.py             data download + causal features
src/hmm.py              HMM fit, decode, Markov-chain analytics, state selection
src/backtest.py         walk-forward decode, exposure overlay, baselines, metrics, costs
src/ticker_pipeline.py  full pipeline for one ticker: states, backtest table, figures
src/plots.py            figures (saved to figures/ as TICKER_name.png)
src/crosscheck.py       Markov-switching regression cross-check
src/testing.py          ad-hoc checks (run: python -m src.testing)
```

## Data

Daily adjusted OHLC from Yahoo Finance (`yfinance`), cached to `data/` (not committed), up to
2026-09-09 (end date set in `src/data.py`). `^GSPC` (S&P 500) is the main series, from
2000-01-01. `TLT` (iShares 20+ Year Treasury Bond ETF) starts 2002-07-30. Index prices exclude
dividends, while TLT's adjusted prices include its distributions, so compare returns within a
market, not across them.

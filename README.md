# Market Regime Detection with a Hidden Markov Model

This project fits a Hidden Markov Model to daily S&P 500 returns and reads its latent states as
market regimes (calm / choppy / crisis). It then tests whether scaling market exposure by the
regime inferred out of sample improves risk-adjusted return over buy and hold.

The hidden-state process is a discrete-time Markov chain, so the project also estimates its
transition matrix, stationary distribution and expected regime durations. The backtest uses
expanding-window walk-forward, so each day's regime is inferred from past data only.

## Result

Out of sample (2004-2026, monthly refits, forward-filtered states) a calm / choppy / crisis
exposure overlay (1.0 / 0.5 / 0.0) reaches Sharpe 0.71 and a -14.9% max drawdown, against 0.54
and -56.8% for buy and hold. It only ties a volatility-target baseline on Sharpe (0.71 each) and
trails it once trading costs are charged; the drawdown edge (-14.9% vs -43.7%) holds. Full
results in `regime_hmm.ipynb`.

## Layout

```
regime_hmm.ipynb   runs the pipeline top to bottom from src/ and renders the figures
src/data.py        data download + causal features
src/hmm.py         HMM fit, decode, Markov-chain analytics, state selection
src/backtest.py    walk-forward decode, exposure overlay, baselines, metrics, costs
src/plots.py       figures (saved to figures/)
src/crosscheck.py  Markov-switching regression cross-check
src/testing.py     ad-hoc checks (run: python -m src.testing)
```

## Data

Daily adjusted OHLC from Yahoo Finance (`yfinance`), cached to `data/` (not committed).
Only `^GSPC` (S&P 500) is used, from 2000-01-01 to 2026-09-09 (end date set in `src/data.py`).

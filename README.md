# Market Regime Detection with a Hidden Markov Model

This project fits a 3-state Gaussian Hidden Markov Model to daily returns, 20-day volatility and
intraday range, and reads its latent states as market regimes (calm / choppy / crisis). It is
built as an input for trading strategies, not as a strategy itself. Each day, using past data
only, it outputs:

- the probability of each regime (forward-filtered, no backward pass), and
- a next-day forecast of the return's mean and volatility: each state's return distribution
  weighted by tomorrow's state probabilities (today's filtered probabilities times the
  transition matrix).

The model is refit every 21 rows (monthly for exchanges) on an expanding window (walk-forward),
so every output is out of sample. Look-ahead is checked by shifting the states one day into the
future, which must score better than the honest states, and a returns-only Markov-switching
model agrees with the HMM on 78% of S&P 500 days. The hidden-state process is a discrete-time
Markov chain, so the project also estimates its transition matrix, stationary distribution and
expected regime durations.

## How fast does it detect a crisis?

The model is judged on detection, not on trading profit. The yardstick is a price-only
volatility benchmark built from daily High and Low (Parkinson estimator): an episode is a run
of 10 or more days where the centred 11-day average sits above its full-sample 90th percentile,
with start and end dates refined from the daily values and gaps of 5 days or fewer merged. The
benchmark uses hindsight on purpose; it defines what the model should have caught.

| Market | Out of sample | Episodes | Missed | Median entry lag | Median exit lag | False-alarm share of crisis days |
|---|---|---|---|---|---|---|
| S&P 500 | 2004-2026 | 11 | 0 | 1 | 24 | 16% |
| FTSE 100 | 2004-2026 | 13 | 0 | 1 | 15 | 18% |
| Bitcoin | 2017-2026 | 12 | 0 | -0.5 | 14 | 45% |
| Treasuries (TLT) | 2006-2026 | 14 | 0 | 3 | 18 | 31% |

Lags are model minus benchmark in trading days (calendar days for Bitcoin); negative means
early. A false alarm is a crisis spell that touches no episode and does not end within 10 days
before one.

- **Fast in, slow out, in every market.** The model enters a crisis within days (a median of 1
  on equities, 3 on Treasuries) but leaves it two to five weeks after the benchmark does. The
  exit is the main weakness.
- **Most false alarms are short.** On the S&P, 17 of the 24 false alarms last 5 days or
  fewer. Bitcoin is far noisier: 153 of its 179 crisis spells last 5 days or fewer.
- **Holdouts.** The Nasdaq-100 and Ethereum are held back and will be run once, on the final
  model.

## Volatility is not direction

On Treasuries and Bitcoin the high-volatility state also holds rallies: bond volatility rises in
flight-to-safety rallies (2008, 2020) and Bitcoin's 2017 and 2021 run-ups were volatile. Two of
the three features (volatility and range) ignore the sign of a move, so the model cannot tell a
crash from a rebound, which is also why exits are slow. The next stage tests return-aware
features (downside deviation, drawdown from the one-year high) to separate falls from
recoveries.

## Exposure overlay (a yardstick, not the goal)

As a fixed check, exposure is set to 1.0 / 0.5 / 0.0 in calm / choppy / crisis. This map was
fixed in advance and is not tuned.

- **S&P 500:** Sharpe 0.71 and max drawdown -14.9%, against 0.54 and -56.8% for buy and hold,
  but a volatility-target baseline matches it on Sharpe (0.71) with a -43.7% drawdown. It lowers
  risk and does not add return (6.1% a year vs 8.8%).
- **FTSE 100 and Bitcoin:** it cuts the drawdown (-25.9% vs -47.8%, -55.3% vs -83.4%) but trails
  buy and hold on Sharpe (0.24 vs 0.31, 0.39 vs 0.89).
- **Treasuries:** it cuts the 2022 rate-rise drawdown (-16.9% vs -41.1%) but trails on Sharpe
  (0.19 vs 0.27). A mirrored map (more exposure in crisis), fixed before the run, does no better
  than a constant 60% position.
- Weighting the map by the regime probabilities instead of the most likely state raised
  turnover (6.4 vs 6.0 a year on the S&P) without improving Sharpe.

Per-market tables are in `results/`. Caveats: leverage in the volatility target (capped at 1.5x)
has no financing cost, cash earns nothing, and Sharpe uses no risk-free rate.

## Layout

```
regime_hmm.ipynb        full pipeline for the ticker set in its settings cell
results/                metrics and detection lags per ticker (written by the notebook)
src/data.py             data download + causal features
src/hmm.py              HMM fit, decode, Markov-chain analytics, state selection
src/backtest.py         walk-forward probabilities and forecast, overlay, baselines, metrics
src/ticker_pipeline.py  full pipeline for one ticker: states, backtest table, figures
src/plots.py            figures (saved to figures/ as TICKER_name.png)
src/events.py           price-only vol episodes, detection lag and false alarms
src/crosscheck.py       Markov-switching regression cross-check
src/testing.py          ad-hoc checks (run: python -m src.testing)
```

## Data

Daily adjusted OHLC from Yahoo Finance (`yfinance`), cached to `data/` (not committed), up to
2026-09-09 (end date set in `src/data.py`). `^GSPC` (S&P 500) and `^FTSE` (FTSE 100) start
in 2000, `TLT` (iShares 20+ Year Treasury Bond ETF) 2002-07-30 and `BTC-USD` 2014-09-17.
Index prices exclude dividends, while TLT's adjusted prices include its distributions, so
compare returns within a market, not across them. Bitcoin trades every day, so metrics are
annualised by the number of rows per year in each series (365 for Bitcoin, about 252 for the
others).

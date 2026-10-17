from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from itertools import groupby

from src.hmm import fit_hmm, label_states

TRADING_DAYS = 252


def _filtered_states(model, X_seq: np.ndarray, n_past: int) -> np.ndarray:
    """
    Forward filter: the state at row i uses only rows 0..i (no backward pass).

    One log-space forward pass over X_seq; returns the argmax filtered state for the rows
    from n_past on (the new block).
    """
    framelogprob = model._compute_log_likelihood(X_seq)          # (T, K) emission log-probs
    with np.errstate(divide='ignore'):                           # zero probs give -inf, which is fine
        log_startprob = np.log(model.startprob_)
        log_transmat = np.log(model.transmat_)

    T = framelogprob.shape[0]
    alpha = np.empty_like(framelogprob)
    alpha[0] = log_startprob + framelogprob[0]                   # P(z_0, x_0)
    for i in range(1, T):
        # alpha[i, j] = logsumexp_k(alpha[i-1, k] + log P(k -> j)) + emission log-prob[i, j]
        alpha[i] = logsumexp(alpha[i - 1][:, None] + log_transmat, axis=0) + framelogprob[i]

    return alpha[n_past:].argmax(axis=1)


def walk_forward_states(X, dates, min_train=1000, step=21, n_states=3) -> pd.Series:
    """
    Expanding-window walk-forward state inference.

    Each model is fit on X[:t] and forward-filters the next `step` days, with its states
    re-ranked by volatility; t then moves forward and the model is refit.
    Ranked states as a Series indexed by date.
    """
    pieces = []
    t = min_train
    while t < len(X):
        model = fit_hmm(X[:t], n_states=n_states)
        mapping = label_states(model)                            # raw -> vol rank, this fit

        end = min(t + step, len(X))
        # one forward pass over past + new block; filtered states for days t..end-1
        filt = _filtered_states(model, X[:end], n_past=t)
        lut = np.array([mapping[k] for k in range(n_states)])    # same mapping as an array, for indexing
        pieces.append(pd.Series(lut[filt], index=dates[t:end]))
        t = end
    return pd.concat(pieces).sort_index()


def overlay(states: pd.Series, next_day_ret: pd.Series, exposure_map: dict) -> pd.DataFrame:
    """
    Exposure for each day from its state, applied to the next-day return.

    Columns: exposure, strat_ret (exposure * next_day_ret) and equity (cumprod of
    1 + strat_ret), on the dates that states and next_day_ret share.
    """
    combined = pd.concat([states, next_day_ret], axis=1, join='inner')
    combined.columns = ['state', 'next_ret']
    combined['exposure'] = combined['state'].map(exposure_map)
    assert combined['exposure'].notna().all(), 'a state is missing from exposure_map'
    combined['strat_ret'] = combined['exposure'] * combined['next_ret']
    combined['equity'] = (1 + combined['strat_ret']).cumprod()
    return combined[['exposure', 'strat_ret', 'equity']]


def baselines(next_day_ret: pd.Series) -> dict[str, pd.DataFrame]:
    """
    Baseline strategies on the same next-day returns: buy_and_hold (exposure 1.0),
    fixed_60 (0.6) and vol_target (expanding-std target over trailing 20d vol, capped at 1.5).

    One frame per baseline, with the same columns as overlay(): exposure, strat_ret, equity.
    """
    out = {}

    exposure = pd.Series(1.0, index=next_day_ret.index)
    strat_ret = exposure * next_day_ret
    equity = (1 + strat_ret).cumprod()
    out['buy_and_hold'] = pd.DataFrame({'exposure': exposure, 'strat_ret': strat_ret, 'equity': equity})

    exposure = pd.Series(0.6, index=next_day_ret.index)
    strat_ret = exposure * next_day_ret
    equity = (1 + strat_ret).cumprod()
    out['fixed_60'] = pd.DataFrame({'exposure': exposure, 'strat_ret': strat_ret, 'equity': equity})

    daily_ret = next_day_ret.shift(1)                            # return realised on day t, known at the close of t
    trailing_vol = daily_ret.rolling(20).std()
    target_vol = daily_ret.expanding(min_periods=TRADING_DAYS).std()   # expanding, so no full-sample look-ahead
    exposure = (target_vol / trailing_vol).clip(upper=1.5)
    strat_ret = exposure * next_day_ret
    equity = (1 + strat_ret).cumprod()
    out['vol_target'] = pd.DataFrame({'exposure': exposure, 'strat_ret': strat_ret, 'equity': equity})
    return out


def metrics(strat_ret: pd.Series, exposure: pd.Series) -> dict:
    """
    Annualised return & vol, Sharpe, Sortino, max drawdown, drawdown duration (days),
    annual turnover.
    """
    # NaNs would be skipped by cumprod/std but still counted in n_days
    assert strat_ret.notna().all(), 'strat_ret contains NaN; restrict() to the OOS window first'
    equity = (1 + strat_ret).cumprod()
    n_days = len(strat_ret)

    ann_return = equity.iloc[-1] ** (TRADING_DAYS / n_days) - 1
    ann_vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    # Sharpe/Sortino use the arithmetic mean (CAGR would penalise higher-vol strategies)
    ann_mean = strat_ret.mean() * TRADING_DAYS
    sharpe = ann_mean / ann_vol

    # downside deviation over all days (positive days count as 0)
    downside_dev = np.sqrt((strat_ret.clip(upper=0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = ann_mean / downside_dev

    running_max = equity.cummax().clip(lower=1.0)                # 1.0 is the starting peak
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    underwater = drawdown < 0
    runs = [len(list(grp)) for val, grp in groupby(underwater) if val]
    dd_duration = max(runs) if runs else 0
    
    annual_turnover = exposure.diff().abs().mean() * TRADING_DAYS

    return {
    "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
    "sortino": sortino, "max_dd": max_dd, "dd_duration": dd_duration,
    "annual_turnover": annual_turnover,
}


def leakage_check(states: pd.Series, next_day_ret: pd.Series, exposure_map: dict) -> None:
    """
    Look-ahead check. Prints the overlay Sharpe with the states as they are and with the
    states shifted one day into the future. The shifted version is cheating, so it should be
    better; if it is not, the pipeline already has look-ahead.
    """
    honest = overlay(states, next_day_ret, exposure_map)
    honest_metrics = metrics(honest['strat_ret'], honest['exposure'])
    
    cheat_states = states.shift(-1)
    cheat_states.dropna(inplace=True)
    cheat = overlay(cheat_states, next_day_ret, exposure_map)
    cheat_metrics = metrics(cheat['strat_ret'], cheat['exposure'])
    
    print(f"honest Sharpe {honest_metrics['sharpe']:.3f} | shifted Sharpe {cheat_metrics['sharpe']:.3f}")
    if cheat_metrics['sharpe'] <= honest_metrics['sharpe']:
        print('Warning: Potential look-ahead detected')
    else:
        print('No look-ahead detected')


def restrict(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """
    Slice a strategy frame to `index` and restart its equity curve at 1 (used to put the
    baselines on the walk-forward OOS window after they are built on the full history).
    """
    out = frame.loc[index, ['exposure', 'strat_ret']].copy()
    assert not out.isna().any().any(), 'NaN inside the window (strategy warm-up not finished)'
    out['equity'] = (1 + out['strat_ret']).cumprod()
    return out


def apply_costs(strat_ret: pd.Series, exposure: pd.Series, cost_bps: float) -> pd.Series:
    """Strategy returns net of a proportional cost of cost_bps per unit of exposure traded."""
    traded = exposure.diff().abs()
    traded.iloc[0] = abs(exposure.iloc[0])                       # first day: pay to enter from 0
    return strat_ret - traded * cost_bps / 10_000


def sweep_exposure_maps(states: pd.Series, next_day_ret: pd.Series,
                        maps: dict[str, dict], cost_bps: float = 5.0) -> pd.DataFrame:
    """
    One row per exposure map, gross metrics plus Sharpe / return net of cost_bps.
    States are passed in so the walk-forward HMM is never refit per map.
    """
    rows = []
    for name, exposure_map in maps.items():
        res = overlay(states, next_day_ret, exposure_map)
        gross = metrics(res['strat_ret'], res['exposure'])
        net = metrics(apply_costs(res['strat_ret'], res['exposure'], cost_bps), res['exposure'])
        rows.append({'map': name, 'ann_return': gross['ann_return'], 'sharpe': gross['sharpe'],
                     'max_dd': gross['max_dd'], 'annual_turnover': gross['annual_turnover'],
                     'net_ann_return': net['ann_return'], 'net_sharpe': net['sharpe']})
    return pd.DataFrame(rows).set_index('map')



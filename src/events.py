from __future__ import annotations

import pandas as pd
import numpy as np


def spells(states: pd.Series, value: int) -> pd.DataFrame:
    """
    Unbroken runs where states equal value: one row per run with start, end and length.
    """
    run_id = (states != states.shift()).cumsum()
    frame = pd.DataFrame({'states': states, 'date': states.index})
    runs = frame.groupby(run_id).agg(states=('states', 'first'), start=('date', 'min'),
                                     end=('date', 'max'), length=('date', 'size'))
    spell = runs[runs['states'] == value].drop(columns='states').reset_index(drop=True)
    return spell


def vol_episodes(prices: pd.DataFrame, q: float = 0.9, window: int = 11, min_len: int = 10,
                 pad: int = 5) -> pd.Series:
    """
    Crisis benchmark from prices alone, using a hybrid of smoothed and daily Parkinson vol:
    2 on days in a high-vol episode, 0 otherwise.

    The centred smoothed vol finds episodes (min_len+ days above its q quantile); the daily vol
    sets each start and end within pad days of the smoothed ones. Gaps of pad days or fewer
    between episodes are merged.
    """
    high = prices['High']
    low = prices['Low']
    daily = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
    smoothed = daily.rolling(window, center=True).mean()
    threshold = smoothed.quantile(q)
    states = (smoothed > threshold) * 2
    spell = spells(states, 2)
    qualifying = spell[spell['length'] >= min_len].reset_index(drop=True)
    episodes = []
    for ep in qualifying.itertuples():
        start_pos = prices.index.get_loc(ep.start)
        end_pos = prices.index.get_loc(ep.end)
        start_pos = max(start_pos - pad, 0)
        end_pos = min(end_pos + pad, len(daily) - 1)
        search = daily.iloc[start_pos : end_pos + 1]
        search = search[search > threshold]
        start = search.index[0]
        end = search.index[-1]
        episodes.append((start, end))
    episode_states = pd.Series(0, index=prices.index)
    for start, end in episodes:
        episode_states.loc[start:end] = 2
    calm = spells(episode_states, 0)
    gaps = calm[(calm['length'] <= pad)
                & (calm['start'] != episode_states.index[0])
                & (calm['end'] != episode_states.index[-1])]
    for g in gaps.itertuples():
        episode_states.loc[g.start : g.end] = 2
    return episode_states


def entry_exit_lags(wf_states: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """
    One row per benchmark episode in the walk-forward period: when the model entered and left
    the crisis state around it.

    Lags are model minus benchmark in trading days (negative = early, positive = late). An episode
    no model crisis spell overlaps is missed, with lags and dates left as NaN.
    """
    bench_cut = bench.loc[wf_states.index]
    episodes = spells(bench_cut, 2)
    model = spells(wf_states, 2)
    rows = []
    for ep in episodes.itertuples():
        overlap = model[(model['end'] >= ep.start) & (model['start'] <= ep.end)]
        if not overlap.empty:
            first = overlap.iloc[0]
            last = overlap.iloc[-1]
            first_eps = episodes[(episodes['start'] <= first['end'])
                                 & (episodes['end'] >= first['start'])]
            last_eps = episodes[(episodes['start'] <= last['end'])
                                & (episodes['end'] >= last['start'])]
            detected = overlap['start'].iloc[0]
            model_exit = overlap['end'].iloc[-1]
            start_pos = wf_states.index.get_loc(ep.start)
            det_pos = wf_states.index.get_loc(detected)
            end_pos = wf_states.index.get_loc(ep.end)
            exit_pos = wf_states.index.get_loc(model_exit)
            entry_lag = det_pos - start_pos
            exit_lag = exit_pos - end_pos
            missed = False
            shared = len(first_eps) > 1 or len(last_eps) > 1
        else:
            detected = np.nan
            model_exit = np.nan
            entry_lag = np.nan
            exit_lag = np.nan
            missed = True
            shared = False
        values = {'start': ep.start, 'end': ep.end, 'length': ep.length, 'detected': detected,
                  'entry_lag': entry_lag, 'model_exit': model_exit, 'exit_lag': exit_lag,
                  'missed': missed, 'shared': shared}
        rows.append(values)
    return pd.DataFrame(rows)


def false_alarms(wf_states: pd.Series, bench: pd.Series, warn: int = 10,
                 flip_len: int = 5) -> pd.DataFrame:
    """
    One row per walk-forward crisis spell, classed as exactly one of: overlaps (touches a
    benchmark episode), early_warning (touches none but ends 1 to warn trading days before one
    starts) or false_alarm (neither).

    flip marks spells of flip_len trading days or fewer, whatever their class.
    """
    bench_cut = bench.loc[wf_states.index]
    episodes = spells(bench_cut, 2)
    model = spells(wf_states, 2)
    ep_start_pos = wf_states.index.get_indexer(episodes['start'])
    rows = []
    for sp in model.itertuples():
        touching = episodes[(episodes['start'] <= sp.end) & (episodes['end'] >= sp.start)]
        overlaps = len(touching) > 0
        end_pos = wf_states.index.get_loc(sp.end)
        gap = ep_start_pos - end_pos
        early_warning = not overlaps and ((gap >= 1) & (gap <= warn)).any()
        false_alarm = not overlaps and not early_warning
        flip = sp.length <= flip_len
        values = {'start': sp.start, 'end': sp.end, 'length': sp.length, 'overlaps': overlaps,
                  'early_warning': early_warning, 'false_alarm': false_alarm, 'flip': flip}
        rows.append(values)
    return pd.DataFrame(rows)

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.data import build_features


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


def fall_recovery(states: pd.Series, probs: pd.DataFrame, bench: pd.Series,
                  close: pd.Series) -> pd.DataFrame:
    """
    One row per benchmark episode in the walk-forward period, split at its lowest close: the
    fall runs from the start to the trough (trough included), the recovery from the next day to
    the end.

    For each phase: its length in trading days, the share of days in the crisis state and the
    mean filtered P(crisis). An empty recovery (trough on the last day) gives NaN.
    """
    wf_frame = probs.assign(state=states)
    bench_cut = bench.loc[wf_frame.index]
    episodes = spells(bench_cut, 2)
    rows = []
    for ep in episodes.itertuples():
        close_vals = close.loc[ep.start : ep.end]
        min_close_idx = close_vals.idxmin()
        fall_rows = wf_frame.loc[ep.start : min_close_idx]
        recov_rows = wf_frame[(wf_frame.index > min_close_idx) & (wf_frame.index <= ep.end)]
        fall_len = len(fall_rows)
        recov_len = len(recov_rows)
        fall_crisis = (fall_rows['state'] == 2).mean()
        recov_crisis = (recov_rows['state'] == 2).mean()
        fall_p = fall_rows['p2'].mean()
        recov_p = recov_rows['p2'].mean()
        temp_dict = {'start': ep.start, 'trough': min_close_idx, 'end': ep.end,
                     'fall_length': fall_len, 'recov_length': recov_len,
                     'fall_crisis_share': fall_crisis, 'recov_crisis_share': recov_crisis,
                     'fall_p': fall_p, 'recov_p': recov_p}
        rows.append(temp_dict)
    return pd.DataFrame(rows)


def threshold_states(measure: pd.Series, q: float = 0.9, min_periods: int = 1000) -> pd.Series:
    """
    Threshold detector: 2 on a day measured above q quantile on past values (expanding window),
    0 otherwise. Days before min_periods have 0 as value.
    """
    threshold = measure.expanding(min_periods=min_periods).quantile(q).shift(1)
    states = (measure > threshold) * 2
    return states


def alert_summary(states: pd.Series, bench: pd.Series) -> pd.Series:
    """
    One detector's scorecard against the benchmark: episodes, missed, entry and exit lags,
    crisis spells by class, flips and the false-alarm share of crisis days. Works for any
    0/2 states (the HMM or a rival rule).
    """
    lags = entry_exit_lags(states, bench)
    fa = false_alarms(states, bench)
    # medians skip missed episodes (NaN lags); shared episodes reuse one model spell
    unshared = lags[~lags['shared']]
    return pd.Series({
        'episodes': len(lags),
        'missed': int(lags['missed'].sum()),
        'median entry lag': lags['entry_lag'].median(),
        'median entry lag (not shared)': unshared['entry_lag'].median(),
        'median exit lag': lags['exit_lag'].median(),
        'median exit lag (not shared)': unshared['exit_lag'].median(),
        'model crisis spells': len(fa),
        'real calls': int(fa['overlaps'].sum()),
        'early warnings': int(fa['early_warning'].sum()),
        'false alarms': int(fa['false_alarm'].sum()),
        'flips': int(fa['flip'].sum()),
        'false-alarm share of crisis days': round(fa.loc[fa['false_alarm'], 'length'].sum()
                                                  / fa['length'].sum(), 3),
    }, dtype=object)


def spell_starts(states: pd.Series, bench: pd.Series, prices: pd.DataFrame,
                 min_periods: int = 1000) -> pd.DataFrame:
    """
    One row per crisis spell, indexed by start date. The type of spell, real, flip or
    long_false, is under group. Spike and shock compare the first day's range and absolute
    return with the previous 60 days, calm_before is whether Vol20 the day before was under
    its expanding 75th percentile, next is the following day's range over the same baseline.

    Day t jump needs spike >= 2 and calm_before.
    Jump day needs next < 1.5 and depends on day_t_jump.
    """
    false_a = false_alarms(states, bench)
    false_a = false_a[~false_a['early_warning']]
    real = false_a['overlaps']
    flip = false_a['false_alarm'] & false_a['flip']
    long_false = false_a['false_alarm'] & ~false_a['flip']
    false_a['group'] = np.select([real, flip, long_false], ['real', 'flip', 'long_false'],
                                 default='other')

    dailies = build_features(prices)
    dailies['usual_range'] = dailies['Range'].rolling(60).median().shift(1)
    dailies['usual_return'] = dailies['Return'].rolling(60).std().shift(1)
    dailies['vol20_threshold'] = (dailies['Vol20'].expanding(min_periods=min_periods)
                                  .quantile(0.75).shift(1))
    dailies['spike'] = dailies['Range'] / dailies['usual_range']
    dailies['shock'] = dailies['Return'].abs() / dailies['usual_return']
    dailies['calm_before'] = dailies['Vol20'].shift(1) < dailies['vol20_threshold']
    dailies['next'] = dailies['Range'].shift(-1) / dailies['usual_range']

    false_a = false_a.set_index('start')
    false_a = false_a.join(dailies[['spike', 'shock', 'calm_before', 'next']], how='left')
    false_a['day_t_jump'] = (false_a['spike'] >= 2) & false_a['calm_before']
    false_a['jump_day'] = false_a['day_t_jump'] & (false_a['next'] < 1.5)
    return false_a


def reliability(probs: pd.Series, outcome: pd.Series,
                bins: tuple = (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)) -> pd.DataFrame:
    """
    Reliability table for probabilities against what happened. One row per bin of probs.
    Number of days, mean probability and observed share are columns. Gap is observed minus mean_p,
    negative where the probabilities were overconfident.
    """
    intervals = pd.cut(probs, bins=bins, include_lowest=True)
    day_data = pd.DataFrame({'probs': probs, 'outcome': outcome})
    table = day_data.groupby(intervals, observed=False).agg(days=('probs', 'size'),
                                                            mean_p=('probs', 'mean'),
                                                            observed=('outcome', 'mean'))
    table['gap'] = table['observed'] - table['mean_p']
    return table


def recalibrate(p_fit: pd.Series, y_fit: pd.Series, p_apply: pd.Series) -> pd.Series:
    """
    Isotonic map from raw probabilities to calibrated ones. Learned on p_fit against y_fit,
    then applied to p_apply.
    """
    model = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
    model.fit(p_fit, y_fit)
    results = model.predict(p_apply)
    return pd.Series(results, index=p_apply.index)
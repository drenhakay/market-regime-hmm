import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_prices, emissions, DATA_DIR, clean_ticker, FEATURE_COLS
from src.backtest import walk_forward_states, overlay, restrict, baselines, metrics, leakage_check
from src.hmm import fit_hmm, label_states, decode, transition_report
from src.plots import plot_drawdowns, plot_equity_curves, plot_regime_price, plot_transition_heatmap

# exposure per vol-ranked state: calm 1.0, choppy 0.5, crisis 0.0
HEADLINE_MAP = {0: 1.0, 1: 0.5, 2: 0.0}


def hmm_pipeline(ticker: str = '^GSPC', min_train: int = 1000, step: int = 21, n_states: int = 3,
                 exposure_map: dict = HEADLINE_MAP, plots: bool = True) -> dict:
    """
    Full pipeline for a given ticker: walk-forward states, overlay vs baselines, full-sample fit.

    Walk-forward states are cached in data/ per ticker and settings. Figures are saved to figures/
    only when plots is True. Returns a dict with the metrics table, states, results, regime shares,
    state table, transition report and full-sample ranked states.
    """
    prices = load_prices(ticker)
    X, dates, next_day_ret = emissions(prices)
    clean = clean_ticker(ticker)

    # ticker and settings in name to distinguish and reuse
    cache = DATA_DIR / f'{clean}_wf_states_{min_train}_{step}_{n_states}.parquet'
    if cache.exists():
        states = pd.read_parquet(cache)['state']
    else:
        states = walk_forward_states(X, dates, min_train=min_train, step=step, n_states=n_states)
        states.to_frame('state').to_parquet(cache)
    assert states.index.equals(dates[min_train:]), 'stale cache: delete it and re-run'
    regime_share = states.map({0: 'calm', 1: 'choppy', 2: 'crisis'}).value_counts(normalize=True)

    res_overlay = overlay(states, next_day_ret, exposure_map)
    # baselines use the full history for warm-up, then get cut to the OOS days
    base = {k: restrict(v, res_overlay.index) for k, v in baselines(next_day_ret).items()}
    results = {'regime overlay': res_overlay, 'buy & hold': base['buy_and_hold'],
               'fixed 60%': base['fixed_60'], 'vol target': base['vol_target']}
    table = pd.DataFrame({k: metrics(v['strat_ret'], v['exposure']) for k, v in results.items()}).T
    table['dd_duration'] = table['dd_duration'].astype(int)
    table['ticker'] = clean

    # assumes knowing tomorrow's state helps: a map that adds exposure in crisis can warn
    leakage_check(states, next_day_ret, exposure_map)

    # full sample fit for interpretation only, nothing in backtest uses it
    model = fit_hmm(X, n_states)
    mapping = label_states(model)                           # raw state -> vol rank
    viterbi, _ = decode(model, X)
    lut = np.array([mapping[k] for k in range(n_states)])
    ranked = lut[viterbi]
    transitions = transition_report(model, viterbi)
    # rows sorted by mean Vol20 so they line up with calm / choppy / crisis
    order = np.argsort(model.means_[:, FEATURE_COLS.index('Vol20')])
    state_table = pd.DataFrame(model.means_[order], columns=FEATURE_COLS,
                               index=['calm', 'choppy', 'crisis'])
    state_table['stationary'] = transitions['stationary']
    state_table['implied_dwell'] = transitions['implied_dwell']
    state_table['realised_dwell'] = transitions['realised_dwell']

    if plots:
        plt.close(plot_regime_price(prices, dates, ranked, ticker=ticker))
        plt.close(plot_transition_heatmap(transitions['P'], ticker=ticker))
        plt.close(plot_equity_curves(results, ticker=ticker))
        plt.close(plot_drawdowns(results, ticker=ticker))

    return {'ticker': clean, 'table': table, 'states': states, 'results': results,
            'regime_share': regime_share, 'state_table': state_table, 'transitions': transitions,
            'ranked': ranked}

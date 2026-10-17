from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def fit_markov_switching(returns: pd.Series, k_regimes: int = 3):
    """
    MarkovRegression on the 100x log return with switching mean and variance.
    """
    # integer index avoids statsmodels' no-frequency warning on the date index
    model = MarkovRegression(returns.reset_index(drop=True), k_regimes=k_regimes, trend="c",
                             switching_variance=True)
    # seeded restarts; the default 100 iterations can stop short of convergence
    res = model.fit(search_reps=20, maxiter=500, rng=0)
    if not res.mle_retvals.get("converged", False):
        raise RuntimeError("Markov-switching fit did not converge; try k_regimes=2 or more restarts")
    return res


def ranked_regime_path(res, index: pd.Index) -> pd.Series:
    """
    Most likely regime per day (smoothed, full-sample), relabelled 0 = lowest variance.
    """
    k = res.model.k_regimes
    sigma2 = np.array([res.params[f"sigma2[{i}]"] for i in range(k)])
    rank = np.argsort(np.argsort(sigma2))                # rank[raw] = variance rank (0 = calmest)
    raw_path = np.asarray(res.smoothed_marginal_probabilities).argmax(axis=1)
    return pd.Series(rank[raw_path], index=index)


def agreement(hmm_path: pd.Series, ms_path: pd.Series) -> dict:
    """
    Same-day agreement, confusion matrix and crisis-day overlap on the shared dates.
    """
    idx = hmm_path.index.intersection(ms_path.index)
    a, b = hmm_path.loc[idx], ms_path.loc[idx]
    crisis_a, crisis_b = a == a.max(), b == b.max()      # highest rank = highest variance
    return {
        "same_day_pct": float((a == b).mean() * 100),
        "confusion": pd.crosstab(a, b, rownames=["hmm"], colnames=["markov_switching"]),
        "crisis_jaccard": float((crisis_a & crisis_b).sum() / (crisis_a | crisis_b).sum()),
    }

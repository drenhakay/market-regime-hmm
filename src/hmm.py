from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from src.data import FEATURE_COLS
from itertools import groupby

def fit_hmm(X: np.ndarray, n_states: int = 3, seed: int = 0, n_restarts: int = 8):
    """
    Full-covariance GaussianHMM, best of n_restarts seeds.

    EM is non-convex, so each seed can end at a different optimum; the fitted model with the
    highest log-likelihood on X is the one returned.
    """
    fits = []
    for i in range(n_restarts):
        random_state = seed + i
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200,
                                random_state=random_state)
            model.fit(X)
            fits.append((model.score(X), model))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not fits:
        raise RuntimeError(f'All {n_restarts} restarts failed.')
    return max(fits, key=lambda x: x[0])[1]


def label_states(model) -> dict[int, int]:
    """
    Mapping {raw_state: ranked_state}, with states ordered by mean Vol20 (0 = calmest,
    n-1 = most volatile). Raw state ids are arbitrary and can permute between fits.
    """
    vol_col = FEATURE_COLS.index('Vol20')
    means = model.means_[:, vol_col]
    order = np.argsort(means)
    mapping = {raw: rank for rank, raw in enumerate(order)}
    return mapping


def decode(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Viterbi path and posteriors: the most likely state sequence and the per-day
    forward-backward state probabilities. Both use the whole series, so they are for
    interpretation only; the backtest uses forward filtering.
    """
    viterbi_path = model.predict(X)
    posteriors = model.predict_proba(X)
    return viterbi_path, posteriors


def transition_report(model, viterbi_path: np.ndarray) -> dict:
    """
    Markov-chain summary of the fitted model, with states in vol-rank order.

    P is the transition matrix, stationary its left eigenvector for eigenvalue 1 (normalised
    to sum to 1), implied_dwell is 1 / (1 - P_ii) per state and realised_dwell the mean run
    length of each state in viterbi_path.
    """
    order = np.argsort(model.means_[:, FEATURE_COLS.index("Vol20")])
    p = model.transmat_[np.ix_(order, order)]

    vals, vecs = np.linalg.eig(p.T)
    k = np.argmin(np.abs(vals - 1))
    stationary = np.real(vecs[:, k])
    stationary = stationary / stationary.sum()

    implied_dwell = 1 / (1- np.diag(p))

    ranked_path = np.argsort(order)[viterbi_path]                # inverse permutation: raw state -> vol rank

    runs = [(state, len(list(grp))) for state, grp in groupby(ranked_path)]

    realised_dwell = np.array([
        np.mean([length for st, length in runs if st == k])
        for k in range(model.n_components)
        ])
    return {"P": p, "stationary": stationary,
        "implied_dwell": implied_dwell, "realised_dwell": realised_dwell}

def _n_params(K: int, d: int) -> int:
    """
    Free parameters in a full-covariance Gaussian HMM (used for BIC).
    """
    trans = K * (K - 1)
    start = K - 1
    means = K * d
    covs = K * d * (d + 1) // 2
    return trans + start + means + covs

def select_n_states(X: np.ndarray, train_end: int, candidates=(2, 3, 4, 5)) -> list[dict]:
    """
    State-count comparison: one row per candidate n, fit on X[:train_end], with the BIC and
    the out-of-sample log-likelihood on X[train_end:].
    """
    d = X.shape[1]
    rows = []
    for n in candidates:
        model = fit_hmm(X[:train_end], n_states=n)
        ll_train = model.score(X[:train_end])
        ll_test = model.score(X[train_end:])

        k = _n_params(n, d)
        bic = k * np.log(train_end) - 2 * ll_train
        rows.append({"n": n, "ll_train": ll_train, "ll_test": ll_test,
                     "bic": bic, "n_params": k, "converged": model.monitor_.converged})
    return rows

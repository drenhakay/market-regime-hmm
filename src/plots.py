from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

REGIME_LABELS = ["calm", "choppy", "crisis"]

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# fixed colour per strategy across figures
STRATEGY_COLORS = {"regime overlay": "#2a78d6", "buy & hold": "#eb6834",
                   "fixed 60%": "#1baf7a", "vol target": "#eda100"}
# opaque pale tints, cool to hot (calm blue, choppy amber, crisis red), light enough for the price line
REGIME_COLORS = ["#cde2fb", "#f7d27a", "#e7837f"]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)


def _plain_log_axis(ax: plt.Axes, ticks: list[float], fmt: str) -> None:
    """
    Plain numeric ticks on a log y-axis.
    """
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=150, facecolor=SURFACE, bbox_inches="tight")


def plot_regime_price(prices: pd.DataFrame, dates: pd.DatetimeIndex,
                      ranked_states: np.ndarray) -> plt.Figure:
    """
    Close price (log axis) with the background shaded by full-sample Viterbi regime.
    ranked_states must already be vol-ranked. Interpretation only, not a trading signal.
    """
    close = prices["Close"].loc[dates]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    _style(ax)
    ax.semilogy(close.index, close.values, color=INK, lw=1.0)
    ax.set_xlim(dates[0], dates[-1])                             # axvspan otherwise pads the x-axis
    ax.set_ylim(close.min() * 0.85, close.max() * 1.1)
    _plain_log_axis(ax, [600, 1000, 2000, 4000, 8000], "{:,.0f}")

    # one span per run of the same state
    starts = np.r_[0, np.flatnonzero(np.diff(ranked_states)) + 1]
    ends = np.r_[starts[1:], len(ranked_states)]
    for s, e in zip(starts, ends):
        # zorder 0 keeps the spans under the gridlines
        ax.axvspan(dates[s], dates[min(e, len(dates) - 1)], color=REGIME_COLORS[ranked_states[s]],
                   lw=0, zorder=0)

    # mark the two big crises
    for label, day in [("2008", "2008-10-10"), ("Mar 2020", "2020-03-16")]:
        ts = pd.Timestamp(day)
        ax.annotate(label, (ts, close.asof(ts)), xytext=(0, -28), textcoords="offset points",
                    ha="center", fontsize=9, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))

    # solid legend box so the swatches aren't drawn over the bands
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(REGIME_COLORS, REGIME_LABELS)],
              loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=1,
              labelcolor=INK_2, fontsize=9, title="regime", title_fontsize=9)
    ax.set_title("S&P 500 by inferred regime (full-sample Viterbi)", loc="left", color=INK, fontsize=12)
    ax.set_ylabel("index level (log)", color=INK_2, fontsize=9)
    _save(fig, "regime_price.png")
    return fig


def _common_returns(results: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """
    Strategy returns on the dates every strategy shares.
    """
    shared = None
    for frame in results.values():
        shared = frame.index if shared is None else shared.intersection(frame.index)
    return {name: frame.loc[shared, "strat_ret"] for name, frame in results.items()}


def plot_equity_curves(results: dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Growth of 1 for each strategy on a log axis, rebased on the shared start date.
    """
    rets = _common_returns(results)
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    _style(ax)
    for name, r in rets.items():
        eq = (1 + r).cumprod()
        ax.semilogy(eq.index, eq.values, lw=1.6, color=STRATEGY_COLORS.get(name, INK_2),
                    label=f"{name} ({eq.iloc[-1]:.1f}x)")
    _plain_log_axis(ax, [0.5, 1, 2, 5, 10], "{:g}x")
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, fontsize=9)
    ax.set_title("Walk-forward equity, growth of 1", loc="left", color=INK, fontsize=12)
    ax.set_ylabel("equity (log)", color=INK_2, fontsize=9)
    _save(fig, "equity_curves.png")
    return fig


def plot_drawdowns(results: dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Underwater chart. The first entry (the overlay) is filled, the baselines are lines only.
    """
    rets = _common_returns(results)
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=SURFACE)
    _style(ax)
    for i, (name, r) in enumerate(rets.items()):
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax().clip(lower=1.0) - 1                # peak starts at 1.0, as in metrics()
        color = STRATEGY_COLORS.get(name, INK_2)
        ax.plot(dd.index, dd.values * 100, lw=1.4, color=color, label=f"{name} ({dd.min():.1%})")
        if i == 0:
            ax.fill_between(dd.index, dd.values * 100, 0, color=color, alpha=0.2, lw=0)
    ax.legend(loc="lower left", frameon=False, labelcolor=INK_2, fontsize=9, title="max drawdown",
              title_fontsize=9)
    ax.set_title("Drawdown from running peak", loc="left", color=INK, fontsize=12)
    ax.set_ylabel("drawdown (%)", color=INK_2, fontsize=9)
    _save(fig, "drawdowns.png")
    return fig


def plot_transition_heatmap(P: np.ndarray, labels: list[str] = REGIME_LABELS) -> plt.Figure:
    """
    Annotated heatmap of the ranked transition matrix (rows = from, columns = to).
    """
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#0d366b"])
    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=SURFACE)
    ax.imshow(P, cmap=cmap, vmin=0, vmax=1)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            # white text on the dark cells, ink on the light ones
            ax.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if P[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)), labels, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, color=INK_2)
    ax.set_xlabel("to", color=INK_2, fontsize=9)
    ax.set_ylabel("from", color=INK_2, fontsize=9)
    ax.set_title("Regime transition probabilities", loc="left", color=INK, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "transition_heatmap.png")
    return fig

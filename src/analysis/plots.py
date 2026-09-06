"""Figures for the results section: violin plots for TTC/jerk distributions
by strategy, a success-rate heatmap over strategy x density, and stacked
failure-mode bars. No CARLA dependency; consumes results/raw_episodes.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats import clip_unbounded_metrics

DENSITY_ORDER = ["light", "medium", "heavy"]


def _ordered_strategies(df: pd.DataFrame) -> list[str]:
    preferred = ["egoistic", "rule_based", "negotiation", "learned"]
    present = [s for s in preferred if s in df["strategy"].unique()]
    extra = [s for s in df["strategy"].unique() if s not in present]
    return present + sorted(extra)


def violin_by_strategy(df: pd.DataFrame, metric: str, density: str, out_path: Path, ylabel: str = None):
    subset = clip_unbounded_metrics(df[df["density"] == density])
    data_by_strategy = [
        (s, subset[subset["strategy"] == s][metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
        for s in _ordered_strategies(subset)
    ]
    # A strategy with no finite values for this metric (e.g. min_ttc_s is
    # +inf for every episode where nothing ever closed) must be dropped from
    # the LABELS as well as from the data — dropping it from only one of the
    # two silently shifts every remaining violin onto the wrong label.
    plotted = [(s, d) for s, d in data_by_strategy if d.size > 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    if plotted:
        strategies = [s for s, _ in plotted]
        ax.violinplot([d for _, d in plotted], showmeans=True)
        ax.set_xticks(range(1, len(strategies) + 1))
        ax.set_xticklabels(strategies, rotation=20)
    else:
        # every strategy was non-finite: emit a labelled empty axes rather
        # than raising out of the middle of a figure-generation run
        ax.set_xticks([])
        ax.text(0.5, 0.5, f"no finite {metric} values", ha="center", va="center",
                transform=ax.transAxes)
    dropped = [s for s, d in data_by_strategy if d.size == 0]
    title = f"{metric} by strategy — density={density}"
    if dropped:
        title += "\n(no finite values: " + ", ".join(dropped) + ")"
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def success_rate_heatmap(df: pd.DataFrame, out_path: Path):
    strategies = _ordered_strategies(df)
    densities = [d for d in DENSITY_ORDER if d in df["density"].unique()]
    matrix = np.zeros((len(strategies), len(densities)))
    for i, s in enumerate(strategies):
        for j, d in enumerate(densities):
            cell = df[(df["strategy"] == s) & (df["density"] == d)]
            matrix[i, j] = cell["merge_success"].mean() if len(cell) else np.nan

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(densities)))
    ax.set_xticklabels(densities)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    for i in range(len(strategies)):
        for j in range(len(densities)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Merge success rate")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def failure_mode_stacked_bars(df: pd.DataFrame, out_path: Path):
    strategies = _ordered_strategies(df)
    modes = sorted(df["failure_mode"].unique())
    counts = pd.crosstab(df["strategy"], df["failure_mode"], normalize="index").reindex(strategies)

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(strategies))
    for mode in modes:
        values = counts[mode].to_numpy() if mode in counts else np.zeros(len(strategies))
        ax.bar(strategies, values, bottom=bottom, label=mode)
        bottom += values
    ax.set_ylabel("proportion of episodes")
    ax.set_title("Failure-mode breakdown by strategy")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_figures(df: pd.DataFrame, out_dir: Path):
    df = clip_unbounded_metrics(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    for density in [d for d in DENSITY_ORDER if d in df["density"].unique()]:
        violin_by_strategy(df, "min_ttc_s", density, out_dir / f"ttc_{density}.png", "min TTC (s)")
        violin_by_strategy(
            df, "jerk_rms_main_mps3", density, out_dir / f"jerk_{density}.png", "RMS jerk (m/s^3)"
        )
    success_rate_heatmap(df, out_dir / "success_rate_heatmap.png")
    failure_mode_stacked_bars(df, out_dir / "failure_modes.png")

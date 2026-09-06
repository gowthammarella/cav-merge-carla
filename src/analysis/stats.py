"""Statistical analysis of results/raw_episodes.csv: bootstrap confidence
intervals per (strategy, density) cell, and non-parametric significance
tests between strategies (Mann-Whitney U pairwise, Kruskal-Wallis overall)
— appropriate because TTC and jerk distributions are rarely normal, and
because most CARLA merging papers report single-run numbers with no CI at
all. This module has no CARLA dependency and runs anywhere.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

DEFAULT_METRICS = (
    "min_ttc_s",
    "completion_time_s",
    "hard_braking_main",
    "jerk_rms_main_mps3",
    "string_effect_speed_drop_mps",
)

# `metrics.min_ttc` returns +inf when the vehicles never close, and a very
# large finite value when they close almost imperceptibly (a near-zero
# denominator). Both are correct as raw TTC but neither is meaningful as a
# safety number, and a handful of them drags a cell's mean into the
# thousands of seconds while collapsing its bootstrap CI onto that one
# outlier. Anything past this bound is "not a safety-relevant interaction",
# so it is clipped to the bound rather than dropped — dropping it would
# silently reduce n and bias the cell toward its riskiest episodes.
TTC_CLIP_S = 30.0
UNBOUNDED_METRIC_CLIPS = {"min_ttc_s": TTC_CLIP_S}


def clip_unbounded_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Bounds metrics that are legitimately unbounded above (see
    `TTC_CLIP_S`). Idempotent, and returns a copy — callers never mutate
    the DataFrame they were handed.
    """
    out = df.copy()
    for metric, upper in UNBOUNDED_METRIC_CLIPS.items():
        if metric in out.columns:
            out[metric] = (
                pd.to_numeric(out[metric], errors="coerce")
                .replace([-np.inf], np.nan)
                .clip(upper=upper)
            )
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) using a percentile bootstrap. Ignores
    NaN/inf entries (e.g. completion_time_s is NaN for failed merges;
    min_ttc_s can be +inf for episodes with no closing interaction —
    callers should filter/clip before calling if that's not desired).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [rng.choice(values, size=values.size, replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lower), float(upper)


def summarize(df: pd.DataFrame, metrics: Iterable[str] = DEFAULT_METRICS) -> pd.DataFrame:
    """One row per (strategy, density), with success rate and a bootstrap
    mean/CI for each metric in `metrics`.
    """
    df = clip_unbounded_metrics(df)
    rows = []
    for (strategy, density), group in df.groupby(["strategy", "density"]):
        row = {
            "strategy": strategy,
            "density": density,
            "n_episodes": len(group),
            "success_rate": group["merge_success"].mean(),
            "collision_rate": group["collision"].mean(),
        }
        for metric in metrics:
            mean, lo, hi = bootstrap_ci(group[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["density", "strategy"]).reset_index(drop=True)


def kruskal_wallis_by_density(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Overall test per density level: does strategy have any effect on
    `metric` at all? Run this before pairwise tests to avoid multiple-
    comparison fishing.
    """
    df = clip_unbounded_metrics(df)
    rows = []
    for density, group in df.groupby("density"):
        samples = [
            g[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            for _, g in group.groupby("strategy")
        ]
        samples = [s for s in samples if s.size > 0]
        if len(samples) < 2:
            continue
        stat, p = scipy_stats.kruskal(*samples)
        rows.append({"density": density, "metric": metric, "H": stat, "p_value": p})
    return pd.DataFrame(rows)


def pairwise_mannwhitney(df: pd.DataFrame, metric: str, density: str) -> pd.DataFrame:
    """Pairwise Mann-Whitney U between every strategy pair, within one
    density level, for one metric. Caller is responsible for correcting
    for multiple comparisons (e.g. Bonferroni) across the resulting rows
    if reporting many metrics at once.
    """
    subset = clip_unbounded_metrics(df)[df["density"] == density]
    strategies = sorted(subset["strategy"].unique())
    rows = []
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1 :]:
            a = subset[subset["strategy"] == s1][metric].replace([np.inf, -np.inf], np.nan).dropna()
            b = subset[subset["strategy"] == s2][metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(a) == 0 or len(b) == 0:
                continue
            stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(
                {
                    "density": density,
                    "metric": metric,
                    "strategy_a": s1,
                    "strategy_b": s2,
                    "U": stat,
                    "p_value": p,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_correct(p_values: pd.Series) -> pd.Series:
    n = len(p_values)
    return (p_values * n).clip(upper=1.0)

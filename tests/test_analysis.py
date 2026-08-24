import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.stats import (
    bootstrap_ci,
    kruskal_wallis_by_density,
    pairwise_mannwhitney,
    summarize,
)


def _synthetic_df(seed=0, n_per_cell=15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    strategies = {"egoistic": 1.5, "rule_based": 2.5, "negotiation": 3.2}
    for strategy, ttc_shift in strategies.items():
        for density in ["light", "medium", "heavy"]:
            for seed_i in range(n_per_cell):
                success = bool(rng.random() < 0.85)
                rows.append(
                    {
                        "strategy": strategy,
                        "density": density,
                        "seed": seed_i,
                        "merge_success": success,
                        "collision": bool(rng.random() < 0.02),
                        "failure_mode": "success_clean" if success else "no_gap_found",
                        "min_ttc_s": max(0.1, rng.normal(ttc_shift, 0.5)),
                        "completion_time_s": rng.normal(6.0, 1.0) if success else np.nan,
                        "hard_braking_main": rng.integers(0, 3),
                        "jerk_rms_main_mps3": abs(rng.normal(1.0, 0.3)),
                        "string_effect_speed_drop_mps": abs(rng.normal(0.5, 0.4)),
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_ci_basic():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=1)
    assert lo <= mean <= hi


def test_bootstrap_ci_handles_inf_and_nan():
    values = np.array([1.0, np.inf, np.nan, 3.0])
    mean, lo, hi = bootstrap_ci(values, n_boot=200, seed=1)
    assert np.isfinite(mean)


def test_bootstrap_ci_empty_returns_nan():
    mean, lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(mean)


def test_summarize_shape():
    df = _synthetic_df()
    summary = summarize(df)
    assert set(summary["strategy"].unique()) == {"egoistic", "rule_based", "negotiation"}
    assert set(summary["density"].unique()) == {"light", "medium", "heavy"}
    assert "min_ttc_s_mean" in summary.columns


def test_kruskal_wallis_detects_shifted_distributions():
    df = _synthetic_df(n_per_cell=40)
    kw = kruskal_wallis_by_density(df, "min_ttc_s")
    # TTC means were deliberately shifted apart across strategies —
    # expect a significant effect at every density
    assert (kw["p_value"] < 0.05).all()


def test_pairwise_mannwhitney_returns_all_pairs():
    df = _synthetic_df()
    result = pairwise_mannwhitney(df, "min_ttc_s", "light")
    # 3 strategies -> 3 pairs
    assert len(result) == 3
    assert set(result["p_value"] <= 1.0)

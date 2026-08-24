"""Generates a results.md summary (tables only, no prose) from
results/raw_episodes.csv — a starting point to paste into the paper's
results section once experiments have run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.stats import DEFAULT_METRICS, kruskal_wallis_by_density, summarize


def _fmt(mean: float, lo: float, hi: float) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def generate_report(results_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(results_csv)
    summary = summarize(df)

    lines = ["# Results summary", "", f"Total episodes: {len(df)}", ""]

    for density in sorted(summary["density"].unique()):
        lines.append(f"## Density: {density}")
        lines.append("")
        header = ["strategy", "n", "success_rate", "collision_rate"] + list(DEFAULT_METRICS)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        subset = summary[summary["density"] == density]
        for _, row in subset.iterrows():
            cells = [
                row["strategy"],
                str(int(row["n_episodes"])),
                f"{row['success_rate']:.2f}",
                f"{row['collision_rate']:.2f}",
            ]
            for metric in DEFAULT_METRICS:
                cells.append(_fmt(row[f"{metric}_mean"], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Kruskal-Wallis (strategy effect per density, per metric)")
    lines.append("")
    for metric in DEFAULT_METRICS:
        kw = kruskal_wallis_by_density(df, metric)
        if kw.empty:
            continue
        lines.append(f"### {metric}")
        lines.append("| density | H | p_value |")
        lines.append("|---|---|---|")
        for _, row in kw.iterrows():
            lines.append(f"| {row['density']} | {row['H']:.2f} | {row['p_value']:.4f} |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

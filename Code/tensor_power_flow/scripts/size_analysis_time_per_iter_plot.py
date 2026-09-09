# =============================================================================
# size_analysis_time_per_iter_plot.py  —  Box plot of time per iteration vs n_bus
# =============================================================================
"""
Plots time per iteration (ms) vs n_bus as box plots.
One box per network size, combining all data (0%, 25%, 50%, 75% PV, all seeds).

Usage:
    python size_analysis_time_per_iter_plot.py
    python size_analysis_time_per_iter_plot.py --csv scripts/test_size_analysis_v2/size_analysis_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    csv: str = "scripts/test_size_analysis_v2/size_analysis_results.csv",
    max_bus: int = 1001,
    out: str = "size_analysis_time_per_iter",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    df = df[df["n_bus"] <= max_bus]
    df = df[df["converged"] == True]
    print(f"Filtered to {len(df)} converged rows with n_bus <= {max_bus}")

    sizes = sorted(df["n_bus"].unique())
    print(f"Network sizes: {sizes}")

    fig, ax = plt.subplots(figsize=(5.9, 4.5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    box_data = [df[df["n_bus"] == n]["t_per_iter_ms"].values for n in sizes]

    box_widths = [n * 0.1 for n in sizes]

    bp = ax.boxplot(box_data, positions=sizes, widths=box_widths,
                    patch_artist=True, showfliers=False)

    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.7)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)

    ax.set_xlabel(r"$n_{\mathrm{bus}}$", fontsize=12)
    ax.set_xlim(17, 1200)
    ax.set_ylabel(r"$\mathrm{Zeit\,\,pro \,\,Iteration \,\,[ms]}$", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    out_dir = csv_path.parent
    pgf_path = out_dir / f"{out}.pgf"
    png_path = out_dir / f"{out}.png"
    USE_PGF=True
    if USE_PGF:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
    fig.savefig(pgf_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pgf_path}")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {png_path}")

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="test_size_analysis_v2/size_analysis_results.csv")
    parser.add_argument("--max-bus", type=int, default=1001)
    parser.add_argument("--out", default="size_analysis_time_per_iter")
    args = parser.parse_args()
    main(csv=args.csv, max_bus=args.max_bus, out=args.out)

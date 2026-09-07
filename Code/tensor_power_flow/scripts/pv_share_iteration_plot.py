# =============================================================================
# pv_share_iteration_plot.py  —  Plot iterations vs network size
# =============================================================================
"""
Plots total iterations vs n_bus.
Two points per network size:
  - 0% PV nodes
  - All PV cases combined (25%, 50%, 75%)

Usage:
    python pv_share_iteration_plot.py
    python pv_share_iteration_plot.py --csv test_size_analysis_v2/test_size_analysis_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(
    csv: str = "test_size_analysis_v2/size_analysis_results.csv",
    max_bus: int = 1001,
    out: str = "pv_share_iteration",
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
    print(df)
    df = df[df["converged"] == True]
    print(f"Filtered to {len(df)} converged rows with n_bus <= {max_bus}")

    sizes = sorted(df["n_bus"].unique())

    fig, ax = plt.subplots(figsize=(5.9, 4.5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    df_no_pv = df[df["pv_percent"] == 0]
    df_with_pv = df[df["pv_percent"] > 0]

    for label, subset, color, marker in [
        ("0% PV", df_no_pv, "#1f77b4", "o"),
        ("PV (25-75%)", df_with_pv, "#ff7f0e", "s"),
    ]:
        means = []
        stds = []
        x_vals = []

        for n in sizes:
            data = subset[subset["n_bus"] == n]["iterations"]
            if len(data) > 0:
                means.append(data.mean())
                stds.append(data.std() if len(data) > 1 else 0)
                x_vals.append(n)

        x_arr = np.array(x_vals)
        y_arr = np.array(means)
        y_err = np.array(stds)

        ax.errorbar(x_arr, y_arr, yerr=y_err, fmt=f"{marker}-", color=color,
                    label=label, markersize=8, capsize=4, capthick=1.5,
                    linewidth=2, alpha=0.8)

    ax.set_xlabel(r"$n_{\mathrm{bus}}$", fontsize=12)
    ax.set_ylabel("Iterations", fontsize=12)
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()

    out_dir = csv_path.parent
    pgf_path = out_dir / f"{out}.pgf"
    png_path = out_dir / f"{out}.png"

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
    parser.add_argument("--out", default="pv_share_iteration")
    args = parser.parse_args()
    main(csv=args.csv, max_bus=args.max_bus, out=args.out)

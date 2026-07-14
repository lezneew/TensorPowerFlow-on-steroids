# tensor_power_flow/scripts/plot_tau_results.py
"""
Plot tau scaling results from CSV files.
Usage: python plot_tau_results.py <csv_files> [options]
"""
import sys, os, argparse
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
import re

def parse_csv(csv_path: str):
    """Load tau scaling data from CSV file."""
    tau, t_tpf, t_nr, speedup, tpf_per, nr_per = [], [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tau.append(int(row["tau"]))
            t_tpf.append(float(row["t_tpf"]))
            t_nr.append(float(row["t_nr"]))
            speedup.append(float(row["speedup"]))
            tpf_per.append(float(row["tpf_per_scenario"]))
            nr_per.append(float(row["nr_per_scenario"]))
    return {
        "tau": np.array(tau),
        "t_tpf": np.array(t_tpf),
        "t_nr": np.array(t_nr),
        "speedup": np.array(speedup),
        "tpf_per_scenario": np.array(tpf_per),
        "nr_per_scenario": np.array(nr_per),
    }


def extract_network_name(csv_path: str) -> str:
    """Extract network name from CSV filename."""
    basename = os.path.basename(csv_path)
    if basename.startswith("tau_scaling_"):
        parts = basename.replace(".csv", "").split("_")
        if len(parts) >= 3:
            return "_".join(parts[2:])
    return basename.replace(".csv", "")


def plot_time_vs_tau(data_list, names, output_dir=None, use_log=True, save_only=False):
    """Plot computing time vs tau (log-log or linear)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    for i, (data, name) in enumerate(zip(data_list, names)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(data["tau"], data["t_tpf"], color=color, marker=marker,
                lw=2, markersize=6, label=f"TPF - {name}")
        ax.plot(data["tau"], data["t_nr"], color=color, marker=marker,
                lw=2, markersize=6, linestyle="--", alpha=0.7, label=f"NR - {name}")

    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Computing time [s]", fontsize=12)
    ax.set_title("Computing Time vs tau", fontsize=14)

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"tau_time_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if not save_only:
        plt.show()
    plt.close()


def plot_time_per_scenario(data_list, names, output_dir=None, use_log=True, save_only=False):
    """Plot time per scenario vs tau."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    for i, (data, name) in enumerate(zip(data_list, names)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(data["tau"], data["tpf_per_scenario"], color=color, marker=marker,
                lw=2, markersize=6, label=f"TPF - {name}")
        ax.plot(data["tau"], data["nr_per_scenario"], color=color, marker=marker,
                lw=2, markersize=6, linestyle="--", alpha=0.7, label=f"NR - {name}")

    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Time per scenario [ms]", fontsize=12)
    ax.set_title("Time per Scenario vs tau", fontsize=14)

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"tau_per_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if not save_only:
        plt.show()
    plt.close()


def plot_speedup(data_list, names, output_dir=None, use_log=True, save_only=False):
    """Plot speedup (TPF/NR) vs tau."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    for i, (data, name) in enumerate(zip(data_list, names)):
        name = "_".join(name.split("_")[:3])
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(data["tau"], data["speedup"], color=color, marker=marker,
                lw=2, markersize=6, label=name)

    ax.axhline(y=1, color="black", linestyle="-", lw=1, alpha=0.5)
    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Speedup (NR time / TPF time)", fontsize=12)
    ax.set_title("TPF Speedup vs NR", fontsize=14)

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"tau_speedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if not save_only:
        plt.show()
    plt.close()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "results"))


def scan_csv_files(directory: str):
    """Scan directory for tau_scaling CSV files."""
    csv_files = []
    if os.path.isdir(directory):
        for f in os.listdir(directory):
            if f.startswith("tau_scaling_") and f.endswith(".csv"):
                csv_files.append(os.path.join(directory, f))
    return sorted(csv_files)


def main():
    parser = argparse.ArgumentParser(description="Plot tau scaling results from CSV files")
    parser.add_argument("csv_files", nargs="*", help="CSV file paths (optional if --scan used)")
    parser.add_argument("--scan", "-s", action="store_true",
                        help="Scan results directory for CSV files")
    parser.add_argument("--output-dir", "-d", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Output directory for plot (default: ../../results)")
    parser.add_argument("--type", "-t", type=str, choices=["time", "time_per_scenario", "speedup"],
                        default="time", help="Plot type (default: time)")
    parser.add_argument("--linear", "-l", action="store_true",
                        help="Use linear scale instead of log-log")
    parser.add_argument("--save-only", action="store_true",
                        help="Save plot to file without displaying")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title")
    args = parser.parse_args()

    csv_paths = args.csv_files

    if args.scan:
        csv_paths = scan_csv_files(DEFAULT_RESULTS_DIR)
        if not csv_paths:
            print(f"No CSV files found in: {DEFAULT_RESULTS_DIR}")
            return
        print(f"Found {len(csv_paths)} CSV files:")
        for p in csv_paths:
            print(f"  - {os.path.basename(p)}")

    if not csv_paths:
        csv_paths = scan_csv_files(DEFAULT_RESULTS_DIR)
        if not csv_paths:
            print(f"Error: No CSV files specified and none found in default directory: {DEFAULT_RESULTS_DIR}")
            return
        print(f"Using {len(csv_paths)} CSV files from default directory:")
        for p in csv_paths:
            print(f"  - {os.path.basename(p)}")

    for p in csv_paths:
        if not os.path.isfile(p):
            print(f"Error: File not found: {p}")
            return

    data_list = []
    names = []
    for p in csv_paths:
        data_list.append(parse_csv(p))
        names.append(extract_network_name(p))

    output_dir = args.output_dir

    use_log = not args.linear
    save_only = args.save_only

    if args.type == "time":
        plot_time_vs_tau(data_list, names, output_dir, use_log, save_only)
    elif args.type == "time_per_scenario":
        plot_time_per_scenario(data_list, names, output_dir, use_log, save_only)
    elif args.type == "speedup":
        plot_speedup(data_list, names, output_dir, use_log, save_only)


if __name__ == "__main__":
    main()
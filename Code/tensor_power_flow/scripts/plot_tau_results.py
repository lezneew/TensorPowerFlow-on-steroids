# tensor_power_flow/scripts/plot_tau_results.py
"""
Plot tau scaling results from CSV files.
Usage: python plot_tau_results.py <csv_files> [options]
"""
import sys, os, argparse, re
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt


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


def extract_grid_size(csv_path: str) -> int:
    """Extract grid size (bus count) from CSV filename. E.g., sz_500_r000 -> 500."""
    basename = os.path.basename(csv_path)
    match = re.search(r'sz_(\d+)_', basename)
    if match:
        return int(match.group(1))
    return 0


def group_by_grid_size(data_list, names, csv_paths):
    """Group data by grid size. Returns dict {size: [(data, name), ...]}."""
    grouped = {}
    for data, name, path in zip(data_list, names, csv_paths):
        size = extract_grid_size(path)
        if size not in grouped:
            grouped[size] = []
        grouped[size].append((data, name))
    return dict(sorted(grouped.items()))


# ============================================================================
# Individual Line Plots (Original)
# ============================================================================

def plot_time_vs_tau(data_list, names, output_dir=None, use_log_x=True, use_log_y=True, save_only=False):
    """Plot computing time vs tau (log-log or linear) - individual lines."""
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

    ax.set_xscale("log" if use_log_x else "linear")
    ax.set_yscale("log" if use_log_y else "linear")
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


def plot_time_per_scenario(data_list, names, output_dir=None, use_log_x=True, use_log_y=True, save_only=False):
    """Plot time per scenario vs tau - individual lines."""
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

    ax.set_xscale("log" if use_log_x else "linear")
    ax.set_yscale("log" if use_log_y else "linear")
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


def plot_speedup(data_list, names, output_dir=None, use_log_x=True, use_log_y=True, save_only=False):
    """Plot speedup (TPF/NR) vs tau - individual lines."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    for i, (data, name) in enumerate(zip(data_list, names)):
        name_short = "_".join(name.split("_")[:3])
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(data["tau"], data["speedup"], color=color, marker=marker,
                lw=2, markersize=6, label=name_short)

    ax.axhline(y=1, color="black", linestyle="-", lw=1, alpha=0.5)
    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Speedup (NR time / TPF time)", fontsize=12)
    ax.set_title("TPF Speedup vs NR", fontsize=14)

    ax.set_xscale("log" if use_log_x else "linear")
    ax.set_yscale("log" if use_log_y else "linear")
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


# ============================================================================
# Band Plots (NEW)
# ============================================================================

def compute_band_stats(data_list, key):
    """Compute min, max, mean across all data for each tau value.
    Handles different tau values across different CSV files.
    """
    # Collect all unique tau values across all datasets
    all_tau_values = set()
    for d in data_list:
        all_tau_values.update(d["tau"].tolist())
    tau_unique = np.array(sorted(all_tau_values))

    # For each tau, collect values from all networks that have this tau
    min_vals = []
    max_vals = []
    mean_vals = []

    for tau_val in tau_unique:
        vals_at_tau = []
        for d in data_list:
            mask = d["tau"] == tau_val
            if np.any(mask):
                vals_at_tau.append(d[key][mask][0])
        if vals_at_tau:
            min_vals.append(np.min(vals_at_tau))
            max_vals.append(np.max(vals_at_tau))
            mean_vals.append(np.mean(vals_at_tau))
        else:
            min_vals.append(np.nan)
            max_vals.append(np.nan)
            mean_vals.append(np.nan)

    return tau_unique, np.array(min_vals), np.array(max_vals), np.array(mean_vals)


def plot_time_vs_tau_bands(data_by_size, nr_data_list, output_dir=None, use_log_x=True, use_log_y=True, save_only=False, band_alpha=0.25):
    """Plot computing time vs tau with bands: NR=1 band, TPF=separate bands per size."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors
    sizes = sorted(data_by_size.keys())

    # Plot NR band (all sizes combined)
    if nr_data_list:
        nr_tau, nr_min, nr_max, nr_mean = compute_band_stats(nr_data_list, "t_nr")
        ax.fill_between(nr_tau, nr_min, nr_max, color="gray", alpha=band_alpha, label="NR (all sizes)")
        ax.plot(nr_tau, nr_mean, color="gray", linestyle="--", lw=1)

    # Plot TPF bands (separate per grid size)
    for i, size in enumerate(sizes):
        color = colors[i % len(colors)]
        data_list = [d for d, _ in data_by_size[size]]
        tau, tmin, tmax, tmean = compute_band_stats(data_list, "t_tpf")
        ax.fill_between(tau, tmin, tmax, color=color, alpha=band_alpha, label=f"TPF sz_{size}")
        ax.plot(tau, tmean, color=color, linestyle="-", lw=1)

    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Computing time [s]", fontsize=12)
    ax.set_title("Computing Time vs tau (Band Plot)", fontsize=14)

    ax.set_xscale("log" if use_log_x else "linear")
    ax.set_yscale("log" if use_log_y else "linear")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"tau_time_bands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if not save_only:
        plt.show()
    plt.close()


def plot_time_per_scenario_bands(data_by_size, nr_data_list, output_dir=None, use_log_x=True, use_log_y=True, save_only=False, band_alpha=0.25):
    """Plot time per scenario vs tau with bands."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors
    sizes = sorted(data_by_size.keys())

    # Plot NR band
    if nr_data_list:
        nr_tau, nr_min, nr_max, nr_mean = compute_band_stats(nr_data_list, "nr_per_scenario")
        ax.fill_between(nr_tau, nr_min, nr_max, color="gray", alpha=band_alpha, label="NR (all sizes)")
        ax.plot(nr_tau, nr_mean, color="gray", linestyle="--", lw=1)

    # Plot TPF bands
    for i, size in enumerate(sizes):
        color = colors[i % len(colors)]
        data_list = [d for d, _ in data_by_size[size]]
        tau, tmin, tmax, tmean = compute_band_stats(data_list, "tpf_per_scenario")
        ax.fill_between(tau, tmin, tmax, color=color, alpha=band_alpha, label=f"TPF sz_{size}")
        ax.plot(tau, tmean, color=color, linestyle="-", lw=1)

    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Time per scenario [ms]", fontsize=12)
    ax.set_title("Time per Scenario vs tau (Band Plot)", fontsize=14)

    ax.set_xscale("log" if use_log_x else "linear")
    ax.set_yscale("log" if use_log_y else "linear")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"tau_per_scenario_bands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if not save_only:
        plt.show()
    plt.close()


def plot_speedup_bands(data_by_size, output_dir=None, use_log_x=True, use_log_y=True, save_only=False, band_alpha=0.25):
    """Plot speedup vs tau with bands grouped by grid size."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.tab10.colors
    sizes = sorted(data_by_size.keys())

    # Plot speedup bands per grid size
    for i, size in enumerate(sizes):
        color = colors[i % len(colors)]
        data_list = [d for d, _ in data_by_size[size]]
        tau, smin, smax, smean = compute_band_stats(data_list, "speedup")
        ax.fill_between(tau, smin, smax, color=color, alpha=band_alpha, label=f"sz_{size}")
        ax.plot(tau, smean, color=color, linestyle="-", lw=1)

    ax.axhline(y=1, color="black", linestyle="-", lw=1, alpha=0.5)
    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Speedup (NR time / TPF time)", fontsize=12)
    ax.set_title("TPF Speedup vs NR (Band Plot)", fontsize=14)

    ax.set_xscale("log" if use_log_x else "linear")
    ax.set_yscale("log" if use_log_y else "linear")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"tau_speedup_bands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    if not save_only:
        plt.show()
    plt.close()


# ============================================================================
# Main
# ============================================================================

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
    parser.add_argument("--linear", "-l", type=str,
                        choices=["x", "y", "both", "none"],
                        default="none",
                        help="Which axis to use linear scale: x, y, both, or none (default: none = log-log)")
    parser.add_argument("--save-only", action="store_true",
                        help="Save plot to file without displaying")
    parser.add_argument("--bands", "-b", action="store_true",
                        help="Plot as bands (NR=1 band, TPF bands per grid size)")
    parser.add_argument("--band-alpha", type=float, default=0.25,
                        help="Band transparency (default: 0.25)")
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
    linear_choice = args.linear
    use_log_x = linear_choice not in ["x", "both"]
    use_log_y = linear_choice not in ["y", "both"]
    save_only = args.save_only
    band_alpha = args.band_alpha

    if args.bands:
        # Band mode: group by grid size
        data_by_size = group_by_grid_size(data_list, names, csv_paths)
        nr_data_list = data_list  # All NR data for combined band

        print(f"\nGrid sizes found: {sorted(data_by_size.keys())}")
        for size, items in data_by_size.items():
            print(f"  sz_{size}: {len(items)} networks")

        if args.type == "time":
            plot_time_vs_tau_bands(data_by_size, nr_data_list, output_dir, use_log_x, use_log_y, save_only, band_alpha)
        elif args.type == "time_per_scenario":
            plot_time_per_scenario_bands(data_by_size, nr_data_list, output_dir, use_log_x, use_log_y, save_only, band_alpha)
        elif args.type == "speedup":
            plot_speedup_bands(data_by_size, output_dir, use_log_x, use_log_y, save_only, band_alpha)
    else:
        # Individual lines mode (original)
        if args.type == "time":
            plot_time_vs_tau(data_list, names, output_dir, use_log_x, use_log_y, save_only)
        elif args.type == "time_per_scenario":
            plot_time_per_scenario(data_list, names, output_dir, use_log_x, use_log_y, save_only)
        elif args.type == "speedup":
            plot_speedup(data_list, names, output_dir, use_log_x, use_log_y, save_only)


if __name__ == "__main__":
    main()
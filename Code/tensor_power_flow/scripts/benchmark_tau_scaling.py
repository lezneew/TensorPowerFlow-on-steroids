# tensor_power_flow/scripts/benchmark_tau_scaling.py
"""
Benchmark TPF vs NR for varying tau (number of time scenarios).
Measures computing time and convergence for different network sizes.
"""
import sys, os, time, argparse, traceback
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import csv

from tpf.builders.from_pandapower import build_network_from_pandapower, build_s_batch_timeseries
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.network_generator_salazar import (
    get_salazar_scaling_networks,
    get_salazar_low_rx05_networks,
    get_salazar_low_rx10_networks,
    get_salazar_low_vm_networks,
)
from tpf.generators.profile_generators import generate_pv_profile, generate_load_profile


SALAZAR_SCALING = get_salazar_scaling_networks()
SALAZAR_LOW_RX05 = get_salazar_low_rx05_networks()
SALAZAR_LOW_RX10 = get_salazar_low_rx10_networks()
SALAZAR_LOW_VM = get_salazar_low_vm_networks()

ALL_SALAZAR_NETWORKS = {**SALAZAR_SCALING, **SALAZAR_LOW_RX05, **SALAZAR_LOW_RX10, **SALAZAR_LOW_VM}
FEWER_SALAZAR_NETWORKS = {}

def run_single_benchmark(network_name: str, tau: int, seed: int = 42):
    """Run TPF and NR for a single tau value. Returns None on error."""
    try:
        network_dict = ALL_SALAZAR_NETWORKS[network_name]
        net = network_dict["constructor"]()
        network = build_network_from_pandapower(net, include_pv=True)

        p_load, q_load = generate_load_profile(net, tau, "daily_double_peak", seed=seed + 1)
        p_pv = None
        if network.has_pv:
            p_pv = generate_pv_profile(
                network.n_pv, tau, "daily_cosine",
                capacity_factor=0.25,
                p_nom_mw=net.gen["p_mw"].values,
                seed=seed + 2,
            )
        s_batch = build_s_batch_timeseries(network, net, p_load, q_load, p_pv)

        tpf = TPFDensePVMethodA(
            tol=1e-11, tol_pv=1e-9, omega=1.0,
            max_iter_inner=100, max_iter_outer=50
        )
        t0 = time.perf_counter()
        res_tpf = tpf.solve_timeseries(network, s_batch, verbose=False)
        t_tpf = time.perf_counter() - t0

        nr = PandapowerNRSolver(tol=1e-10, max_iter=100)
        t0 = time.perf_counter()
        res_nr = nr.solve_timeseries(net, p_load, q_load, p_pv, verbose=False)
        t_nr = time.perf_counter() - t0

        return {
            "tau": tau,
            "t_tpf": t_tpf,
            "t_nr": t_nr,
            "speedup": t_nr / t_tpf if t_tpf > 0 else 0,
            "tpf_per_scenario": t_tpf / tau * 1000,
            "nr_per_scenario": t_nr / tau * 1000,
            "tpf_converged": res_tpf.converged,
            "nr_converged": res_nr.converged,
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def run_all_benchmarks(network_name: str, tau_values: list[int], seed: int = 42):
    """Run benchmarks for all tau values. Skips failed benchmarks."""
    results = []
    for tau in tau_values:
        print(f"  Running tau={tau:,}...", end=" ", flush=True)
        r = run_single_benchmark(network_name, tau, seed)
        if r is None:
            print("FAILED")
            continue
        results.append(r)
        print(f"TPF: {r['t_tpf']:.2f}s, NR: {r['t_nr']:.2f}s, Speedup: {r['speedup']:.1f}x")
    return results


def print_table(results: list[dict]):
    """Print results as a formatted table."""
    print("\n" + "="*95)
    print(f"  {'tau':>8} | {'TPF [s]':>10} | {'NR [s]':>10} | {'Speedup':>8} | "
          f"{'TPF ms/t':>10} | {'NR ms/t':>10} | {'Conv TPF':>8} | {'Conv NR':>8}")
    print("-"*95)
    for r in results:
        print(f"  {r['tau']:>8,} | {r['t_tpf']:>10.2f} | {r['t_nr']:>10.2f} | "
              f"{r['speedup']:>7.1f}x | {r['tpf_per_scenario']:>10.3f} | "
              f"{r['nr_per_scenario']:>10.3f} | {'Yes' if r['tpf_converged'] else 'No':>8} | "
              f"{'Yes' if r['nr_converged'] else 'No':>8}")
    print("="*95)


def save_results_to_file(results: list[dict], network_name: str, output_dir: str, timestamp: str = None):
    """Save all results to txt and csv files in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = f"tau_scaling_{network_name}_{timestamp}"
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    csv_path = os.path.join(output_dir, f"{base_name}.csv")

    with open(txt_path, "w") as f:
        f.write(f"tau_scaling Benchmark Results\n")
        f.write(f"Network: {network_name}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 95 + "\n")
        f.write(f"  {'tau':>8} | {'TPF [s]':>10} | {'NR [s]':>10} | {'Speedup':>8} | "
                f"{'TPF ms/t':>10} | {'NR ms/t':>10} | {'Conv TPF':>8} | {'Conv NR':>8}\n")
        f.write("-" * 95 + "\n")
        for r in results:
            f.write(f"  {r['tau']:>8,} | {r['t_tpf']:>10.2f} | {r['t_nr']:>10.2f} | "
                    f"{r['speedup']:>7.1f}x | {r['tpf_per_scenario']:>10.3f} | "
                    f"{r['nr_per_scenario']:>10.3f} | {'Yes' if r['tpf_converged'] else 'No':>8} | "
                    f"{'Yes' if r['nr_converged'] else 'No':>8}\n")
        f.write("=" * 95 + "\n")
    print(f"Results saved: {txt_path}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "t_tpf", "t_nr", "speedup", "tpf_per_scenario", "nr_per_scenario",
                         "tpf_converged", "nr_converged"])
        for r in results:
            writer.writerow([r["tau"], r["t_tpf"], r["t_nr"], r["speedup"],
                             r["tpf_per_scenario"], r["nr_per_scenario"],
                             r["tpf_converged"], r["nr_converged"]])
    print(f"CSV saved: {csv_path}")

    return txt_path, csv_path


def plot_results(results: list[dict], network_name: str, output_dir: str = None):
    """Create log-log plot of tau vs computing time."""
    tau_vals = np.array([r["tau"] for r in results])
    t_tpf = np.array([r["t_tpf"] for r in results])
    t_nr = np.array([r["t_nr"] for r in results])
    tpf_per = np.array([r["tpf_per_scenario"] for r in results])
    nr_per = np.array([r["nr_per_scenario"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.loglog(tau_vals, t_tpf, "b-o", lw=2, markersize=6, label="TPF")
    ax.loglog(tau_vals, t_nr, "r-s", lw=2, markersize=6, label="NR")
    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Computing time [s]", fontsize=12)
    ax.set_title(f"Computing Time vs tau - {network_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.loglog(tau_vals, tpf_per, "b-o", lw=2, markersize=6, label="TPF")
    ax.loglog(tau_vals, nr_per, "r-s", lw=2, markersize=6, label="NR")
    ax.set_xlabel("tau (number of scenarios)", fontsize=12)
    ax.set_ylabel("Time per scenario [ms]", fontsize=12)
    ax.set_title("Time per Scenario vs tau", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"tau_scaling_{network_name}_{timestamp}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    plt.show()


def get_completed_networks(output_dir: str) -> set:
    """Scan output directory for existing benchmark results. Returns set of network names."""
    completed = set()
    if not os.path.isdir(output_dir):
        return completed
    for f in os.listdir(output_dir):
        if f.startswith("tau_scaling_") and (f.endswith(".txt") or f.endswith(".csv")):
            parts = f.split("_")
            if len(parts) >= 3:
                network_name = "_".join(parts[2:-1])
                completed.add(network_name)
    return completed


def check_convergence(network_name: str, tau: int, seed: int = 42):
    """Quick convergence check at a single tau value. Returns (tpf_converged, nr_converged, error)."""
    try:
        r = run_single_benchmark(network_name, tau, seed)
        if r is None:
            return None, None, "Benchmark failed"
        return r["tpf_converged"], r["nr_converged"], None
    except Exception as e:
        return None, None, str(e)


def run_batch_benchmark(network_names: list[str], tau_values: list[int], output_dir: str,
                        convergence_tau: int = 10, seed: int = 42):
    """Run batch benchmark for multiple networks. Returns summary list."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    failed_networks = []
    skipped_networks = []
    already_exists = []

    completed = get_completed_networks(output_dir)
    networks_to_run = [n for n in network_names if n not in completed]
    already_exists = [n for n in network_names if n in completed]

    if already_exists:
        print(f"\n{'='*80}")
        print(f"  Found {len(already_exists)} existing results, will skip:")
        for net in already_exists[:10]:
            print(f"    - {net}")
        if len(already_exists) > 10:
            print(f"    ... and {len(already_exists) - 10} more")
        print(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print(f"  BATCH BENCHMARK: {len(networks_to_run)} networks to run ({len(network_names)} total)")
    print(f"  Tau values: {tau_values}")
    print(f"  Convergence check tau: {convergence_tau}")
    print(f"{'='*80}\n")

    for i, network_name in enumerate(networks_to_run, 1):
        print(f"[{i}/{len(networks_to_run)}] Network: {network_name}")

        tpf_conv, nr_conv, err = check_convergence(network_name, convergence_tau, seed)

        if err is not None:
            print(f"  -> ERROR during convergence check: {err}")
            failed_networks.append((network_name, err))
            continue

        if not tpf_conv or not nr_conv:
            print(f"  -> SKIPPED: TPF converged={tpf_conv}, NR converged={nr_conv}")
            skipped_networks.append(network_name)
            continue

        print(f"  -> Convergence OK, running full benchmark...")

        try:
            results = run_all_benchmarks(network_name, tau_values, seed)
            if not results:
                print(f"  -> FAILED: No results")
                failed_networks.append((network_name, "No results"))
                continue

            save_results_to_file(results, network_name, output_dir, timestamp)

            avg_speedup = sum(r["speedup"] for r in results) / len(results)
            all_results.append({
                "network": network_name,
                "n_tau": len(results),
                "avg_speedup": avg_speedup,
                "max_speedup": max(r["speedup"] for r in results),
                "min_speedup": min(r["speedup"] for r in results),
            })
            print(f"  -> DONE: avg speedup={avg_speedup:.1f}x")

        except Exception as e:
            print(f"  -> ERROR: {e}")
            failed_networks.append((network_name, str(e)))

    save_summary(all_results, failed_networks, skipped_networks, already_exists, output_dir, timestamp)
    return all_results, failed_networks, skipped_networks


def save_summary(all_results: list, failed_networks: list, skipped_networks: list,
                 already_exists: list, output_dir: str, timestamp: str):
    """Save summary file for batch run."""
    summary_path = os.path.join(output_dir, f"batch_summary_{timestamp}.txt")

    with open(summary_path, "w") as f:
        f.write(f"Batch Benchmark Summary\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"SUCCESSFUL: {len(all_results)} networks\n")
        if all_results:
            f.write("-" * 80 + "\n")
            f.write(f"  {'Network':<25} | {'#tau':>6} | {'Avg Speedup':>12} | {'Min':>8} | {'Max':>8}\n")
            f.write("-" * 80 + "\n")
            for r in all_results:
                f.write(f"  {r['network']:<25} | {r['n_tau']:>6} | {r['avg_speedup']:>11.1f}x | "
                        f"{r['min_speedup']:>7.1f}x | {r['max_speedup']:>7.1f}x\n")

        f.write(f"\nSKIPPED (no convergence): {len(skipped_networks)} networks\n")
        if skipped_networks:
            for net in skipped_networks:
                f.write(f"  - {net}\n")

        f.write(f"\nALREADY EXISTS: {len(already_exists)} networks\n")
        if already_exists:
            for net in already_exists:
                f.write(f"  - {net}\n")

        f.write(f"\nFAILED (errors): {len(failed_networks)} networks\n")
        if failed_networks:
            for net, err in failed_networks:
                f.write(f"  - {net}: {err}\n")

    print(f"\nSummary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TPF vs NR for varying tau")
    parser.add_argument("--network", "-n", type=str, default="sz_75_r010",
                        help="Network name (default: sz_75_r010)")
    parser.add_argument("--tau", "-t", type=str,
                        default="1, 10,50,100,500,1000,5000,10000,50000,100000",
                        help="Comma-separated tau values (default:1,10,50,100,500,1000,5000,10000,50000,100000)")
    parser.add_argument("--output-dir", "-d", type=str, default=None,
                        help="Output directory for results (default: ./tau_benchmark_results)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Run batch mode: all networks in suite")
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Filter networks by prefix (e.g., 'sz_', 'ieee_')")
    parser.add_argument("--convergence-tau", "-c", type=int, default=10,
                        help="Tau value for convergence check (default: 10)")
    args = parser.parse_args()

    tau_values = [int(x.strip()) for x in args.tau.split(",")]
    output_dir = args.output_dir if args.output_dir else "tau_benchmark_results"

    if args.batch:
        network_names = list(ALL_SALAZAR_NETWORKS.keys())
        if args.filter:
            network_names = [n for n in network_names if n.startswith(args.filter)]
        if not network_names:
            print(f"Error: No networks match filter '{args.filter}'")
            return

        run_batch_benchmark(network_names, tau_values, output_dir,
                            convergence_tau=args.convergence_tau)
        return

    if args.network not in ALL_SALAZAR_NETWORKS:
        print(f"Error: Network '{args.network}' not found.")
        print(f"Available networks: {list(ALL_SALAZAR_NETWORKS.keys())[:10]}... (and more)")
        return

    print(f"=================================================================================")
    print(f"  BENCHMARK: TPF vs NR - tau Scaling Analysis")
    print(f"  Network: {args.network:<50}")
    print(f"  Output:  {output_dir:<50}")
    print(f"=================================================================================")

    results = run_all_benchmarks(args.network, tau_values)
    print_table(results)
    save_results_to_file(results, args.network, output_dir)
    plot_results(results, args.network, output_dir)


if __name__ == "__main__":
    main()
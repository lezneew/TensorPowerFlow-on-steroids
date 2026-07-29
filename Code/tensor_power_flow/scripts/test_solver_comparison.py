# tensor_power_flow/scripts/test_solver_comparison.py
"""
Comparison of TPFDensePVMethodA configurations vs Newton-Raphson
================================================================

Compares three solver methods:
1. Full: Method A with no inner iteration optimization
2. Adaptive: Method A with adaptive_inner=True
3. NR: Newton-Raphson from pandapower

Usage:
    python scripts/test_solver_comparison.py --suite quick
    python scripts/test_solver_comparison.py --suite salazar_scaling --show-plot
    python scripts/test_solver_comparison.py --network case14 --show-plot
"""

import numpy as np
import sys
import os
import warnings
import argparse
import json

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.radial_network import get_quick_test_networks
from tpf.generators.network_generator_salazar import (
    create_salazar_network,
    get_salazar_scaling_networks,
    get_salazar_all_networks,
    get_salazar_pv_networks,
    get_salazar_small_pv_networks,
    get_salazar_large_pv_networks,
    get_salazar_paper_networks,
    get_salazar_low_rx05_networks,
    get_salazar_low_rx10_networks,
)

VALID_SUITES = [
    "quick",
    "salazar_scaling",
    "salazar_all",
    "salazar_pv",
    "salazar_small_pv",
    "salazar_large_pv",
    "salazar_paper",
]


def get_suite_networks(suite_name: str) -> dict:
    """Get networks for a single suite name."""
    suites = {
        "quick": get_quick_test_networks,
        "salazar_scaling": get_salazar_scaling_networks,
        "salazar_all": get_salazar_all_networks,
        "salazar_pv": get_salazar_pv_networks,
        "salazar_small_pv": get_salazar_small_pv_networks,
        "salazar_large_pv": get_salazar_large_pv_networks,
        "salazar_paper": get_salazar_paper_networks,
    }
    if suite_name not in suites:
        raise ValueError(f"Unknown suite: '{suite_name}'. Valid: {VALID_SUITES}")
    return suites[suite_name]()

def test_full_method_a(network, max_inner=50, omega=1.0):
    """Test Method A with no inner iteration optimization."""
    solver = TPFDensePVMethodA(
        tol=1e-6,
        max_iter_inner=max_inner,
        max_iter_outer=30,
        tol_pv=1e-6,
        omega=omega,
        enforce_q_lims=False,
        cold_start=False,
        max_iter_inner_per_outer=None,
        adaptive_inner=False,
    )

    try:
        result = solver.solve(network)
    except Exception as e:
        return {
            "converged": False,
            "iterations": 0,
            "time_ms": 0.0,
            "error": np.inf,
            "outer_iters": 0,
            "inner_total": 0,
            "total_iterations": 0,
            "inner_per_outer": [],
            "error_history": [],
            "inner_v_change_all": [],
        }

    pv_info = solver.pv_info
    outer_iters = pv_info.outer_iterations if pv_info else 0
    inner_total = pv_info.inner_iterations_total if pv_info else 0
    total_iters = outer_iters + inner_total
    return {
        "converged": result.converged,
        "iterations": total_iters,
        "time_ms": result.elapsed_time_s * 1000,
        "error": result.max_mismatch,
        "outer_iters": outer_iters,
        "inner_total": inner_total,
        "total_iterations": total_iters,
        "inner_per_outer": pv_info.inner_iterations_per_outer if pv_info else [],
        "error_history": pv_info.pv_v_error_history if pv_info else [],
        "inner_v_change_all": pv_info.inner_v_change_all if pv_info else [],
    }


def test_adaptive_method_a(network, max_inner=50, omega=1.0):
    """Test Method A with adaptive inner iteration control."""
    solver = TPFDensePVMethodA(
        tol=1e-6,
        max_iter_inner=max_inner,
        max_iter_outer=30,
        tol_pv=1e-6,
        omega=omega,
        enforce_q_lims=False,
        cold_start=False,
        max_iter_inner_per_outer=None,
        adaptive_inner=True,
    )

    try:
        result = solver.solve(network)
    except Exception as e:
        return {
            "converged": False,
            "iterations": 0,
            "time_ms": 0.0,
            "error": np.inf,
            "outer_iters": 0,
            "inner_total": 0,
            "total_iterations": 0,
            "inner_per_outer": [],
            "error_history": [],
            "inner_v_change_all": [],
        }

    pv_info = solver.pv_info
    outer_iters = pv_info.outer_iterations if pv_info else 0
    inner_total = pv_info.inner_iterations_total if pv_info else 0
    total_iters = outer_iters + inner_total
    return {
        "converged": result.converged,
        "iterations": total_iters,
        "time_ms": result.elapsed_time_s * 1000,
        "error": result.max_mismatch,
        "outer_iters": outer_iters,
        "inner_total": inner_total,
        "total_iterations": total_iters,
        "inner_per_outer": pv_info.inner_iterations_per_outer if pv_info else [],
        "error_history": pv_info.pv_v_error_history if pv_info else [],
        "inner_v_change_all": pv_info.inner_v_change_all if pv_info else [],
    }


def test_nr_solver(net):
    """Test Newton-Raphson from pandapower."""
    nr_solver = PandapowerNRSolver(tol=1e-6, max_iter=100)

    try:
        result = nr_solver.solve_from_net(net)
    except Exception as e:
        return {
            "converged": False,
            "iterations": 0,
            "time_ms": 0.0,
            "error": np.inf,
        }

    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "time_ms": result.elapsed_time_s * 1000,
        "error": result.max_mismatch,
    }


def test_suite(suite_name: str, max_inner: int = 50, omega: float = 1.0):
    """Test all networks in a suite with all three methods."""
    networks = get_suite_networks(suite_name)
    all_results = {}

    for net_name, net_info in networks.items():
        print(f"\n  Testing {net_name}...", end="", flush=True)

        net_constructor = net_info["constructor"]
        try:
            net = net_constructor()
            network = build_network_from_pandapower(net, include_pv=True)
        except Exception as e:
            print(f" ERROR (skipping)")
            continue

        full_result = test_full_method_a(network, max_inner, omega)
        adaptive_result = test_adaptive_method_a(network, max_inner, omega)
        nr_result = test_nr_solver(net)

        all_results[net_name] = {
            "net": net,
            "network": network,
            "n_buses": network.n_bus_phases,
            "n_pv": network.n_pv,
            "full": full_result,
            "adaptive": adaptive_result,
            "nr": nr_result,
        }

        conv_status = []
        if full_result["converged"]:
            conv_status.append("F")
        if adaptive_result["converged"]:
            conv_status.append("A")
        if nr_result["converged"]:
            conv_status.append("N")
        print(f" [{','.join(conv_status) if conv_status else 'NONE'}]", end="")
        print(f" {full_result['time_ms']:.1f}ms / {adaptive_result['time_ms']:.1f}ms / {nr_result['time_ms']:.1f}ms")

    return all_results


def test_single_network(net_name: str, max_inner: int = 50, omega: float = 1.0):
    """Test a single network with all three methods."""
    networks = get_suite_networks("quick")

    if net_name not in networks:
        networks = get_suite_networks("salazar_scaling")
        if net_name not in networks:
            raise ValueError(f"Unknown network: {net_name}")

    net_info = networks[net_name]
    net_constructor = net_info["constructor"]
    net = net_constructor()
    network = build_network_from_pandapower(net, include_pv=True)

    print(f"\n  Testing {net_name}...")

    full_result = test_full_method_a(network, max_inner, omega)
    adaptive_result = test_adaptive_method_a(network, max_inner, omega)
    nr_result = test_nr_solver(net)

    return {
        net_name: {
            "net": net,
            "network": network,
            "n_buses": network.n_bus_phases,
            "n_pv": network.n_pv,
            "full": full_result,
            "adaptive": adaptive_result,
            "nr": nr_result,
        }
    }


def print_summary_table(all_results: dict):
    """Print summary table comparing all methods."""
    print(f"\n{'='*200}")
    print(f"  SOLVER COMPARISON RESULTS")
    print(f"{'='*200}")

    header = (f"  {'Network':<22} {'Buses':<6} {'PV':<3} "
              f"{'Full (O+I=Total)':<28} {'Adaptive (O+I=Total)':<28} {'NR':<20} {'Spd FvsA':<10} {'Spd FvsNR':<10}")
    print(header)
    print(f"  {'-'*198}")

    for net_name, data in all_results.items():
        n_buses = data["n_buses"]
        n_pv = data["n_pv"]

        full = data["full"]
        adaptive = data["adaptive"]
        nr = data["nr"]

        if full["converged"]:
            full_str = f"{full['outer_iters']}+{full['inner_total']}={full['total_iterations']}/{full['time_ms']:.1f}ms"
        else:
            full_str = "DIVERGED"

        if adaptive["converged"]:
            adaptive_str = f"{adaptive['outer_iters']}+{adaptive['inner_total']}={adaptive['total_iterations']}/{adaptive['time_ms']:.1f}ms"
        else:
            adaptive_str = "DIVERGED"

        nr_str = f"{nr['iterations']}it/{nr['time_ms']:.1f}ms" if nr["converged"] else "DIVERGED"

        speedup_fa = ""
        if full["converged"] and adaptive["converged"] and adaptive["time_ms"] > 0:
            speedup_fa = f"{full['time_ms'] / adaptive['time_ms']:.2f}x"

        speedup_fnr = ""
        if full["converged"] and nr["converged"] and nr["time_ms"] > 0:
            speedup_fnr = f"{nr['time_ms'] / full['time_ms']:.2f}x"

        print(f"  {net_name:<22} {n_buses:<6} {n_pv:<3} "
              f"{full_str:<28} {adaptive_str:<28} {nr_str:<20} {speedup_fa:<10} {speedup_fnr:<10}")

    print(f"{'='*200}")

    full_times = [d["full"]["time_ms"] for d in all_results.values() if d["full"]["converged"]]
    adaptive_times = [d["adaptive"]["time_ms"] for d in all_results.values() if d["adaptive"]["converged"]]
    nr_times = [d["nr"]["time_ms"] for d in all_results.values() if d["nr"]["converged"]]

    print(f"\n  SUMMARY:")
    print(f"    Networks tested: {len(all_results)}")
    if full_times:
        print(f"    Avg time (Full):    {np.mean(full_times):.2f}ms")
    if adaptive_times:
        print(f"    Avg time (Adaptive): {np.mean(adaptive_times):.2f}ms")
    if nr_times:
        print(f"    Avg time (NR):      {np.mean(nr_times):.2f}ms")

    if full_times and nr_times:
        print(f"    Avg speedup (Full vs NR): {np.mean(nr_times) / np.mean(full_times):.2f}x")
    if adaptive_times and nr_times:
        print(f"    Avg speedup (Adaptive vs NR): {np.mean(nr_times) / np.mean(adaptive_times):.2f}x")
    if full_times and adaptive_times:
        print(f"    Avg speedup (Adaptive vs Full): {np.mean(full_times) / np.mean(adaptive_times):.2f}x")


def plot_comparison(all_results: dict, suite_name: str, save_path: str = None):
    """Create comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    network_names = list(all_results.keys())
    n_networks = len(network_names)

    if n_networks == 0:
        print("  (no results to plot)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = plt.cm.tab10.colors

    full_times = []
    adaptive_times = []
    nr_times = []
    full_iters = []
    adaptive_iters = []
    nr_iters = []

    for net_name in network_names:
        d = all_results[net_name]
        full_times.append(d["full"]["time_ms"] if d["full"]["converged"] else 0)
        adaptive_times.append(d["adaptive"]["time_ms"] if d["adaptive"]["converged"] else 0)
        nr_times.append(d["nr"]["time_ms"] if d["nr"]["converged"] else 0)
        full_iters.append(d["full"]["iterations"] if d["full"]["converged"] else 0)
        adaptive_iters.append(d["adaptive"]["iterations"] if d["adaptive"]["converged"] else 0)
        nr_iters.append(d["nr"]["iterations"] if d["nr"]["converged"] else 0)

    x = np.arange(n_networks)
    width = 0.35

    ax_a = axes[0, 0]
    ax_a.bar(x - width/2, full_times, width, label="Full", color=colors[0], alpha=0.8)
    ax_a.bar(x + width/2, adaptive_times, width, label="Adaptive", color=colors[1], alpha=0.8)
    ax_a.set_xlabel("Network", fontsize=11)
    ax_a.set_ylabel("Time (ms)", fontsize=11)
    ax_a.set_title("(a) Computation Time: Full vs Adaptive", fontsize=12)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([n[:12] for n in network_names], rotation=45, ha="right", fontsize=8)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3, axis="y")

    ax_b = axes[0, 1]
    ax_b.bar(x - width/2, full_iters, width, label="Full", color=colors[0], alpha=0.8)
    ax_b.bar(x + width/2, adaptive_iters, width, label="Adaptive", color=colors[1], alpha=0.8)
    ax_b.set_xlabel("Network", fontsize=11)
    ax_b.set_ylabel("Total Iterations (Outer + Inner)", fontsize=11)
    ax_b.set_title("(b) Total Iteration Count: Full vs Adaptive", fontsize=12)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([n[:12] for n in network_names], rotation=45, ha="right", fontsize=8)
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3, axis="y")

    ax_c = axes[1, 0]
    for i, net_name in enumerate(network_names):
        d = all_results[net_name]
        full_inner = d["full"]["inner_per_outer"]
        adaptive_inner = d["adaptive"]["inner_per_outer"]

        if full_inner:
            outer_iters_full = list(range(1, len(full_inner) + 1))
            ax_c.plot(outer_iters_full, full_inner, marker="o", linewidth=2, markersize=6,
                     label=f"{net_name[:10]} (Full)", color=colors[i % len(colors)], linestyle="-")

        if adaptive_inner:
            outer_iters_adaptive = list(range(1, len(adaptive_inner) + 1))
            ax_c.plot(outer_iters_adaptive, adaptive_inner, marker="s", linewidth=2, markersize=6,
                     label=f"{net_name[:10]} (Adaptive)", color=colors[i % len(colors)], linestyle="--")

    ax_c.set_xlabel("Outer Iteration", fontsize=11)
    ax_c.set_ylabel("Inner Iterations", fontsize=11)
    ax_c.set_title("(c) Inner Iterations per Outer Iteration", fontsize=12)
    ax_c.legend(fontsize=7, loc="upper right", ncol=2)
    ax_c.grid(True, alpha=0.3)

    ax_d = axes[1, 1]
    for i, net_name in enumerate(network_names):
        d = all_results[net_name]
        full_v_change = d["full"]["inner_v_change_all"]
        adaptive_v_change = d["adaptive"]["inner_v_change_all"]

        if full_v_change:
            cumsum = list(range(1, len(full_v_change) + 1))
            ax_d.semilogy(cumsum, full_v_change, marker="o", linewidth=1.5, markersize=4,
                         label=f"{net_name[:10]} (Full)", color=colors[i % len(colors)], linestyle="-")

        if adaptive_v_change:
            cumsum = list(range(1, len(adaptive_v_change) + 1))
            ax_d.semilogy(cumsum, adaptive_v_change, marker="s", linewidth=1.5, markersize=4,
                         label=f"{net_name[:10]} (Adaptive)", color=colors[i % len(colors)], linestyle="--")

    ax_d.set_xlabel("Cumulative Iterations (Inner FPI iterations)", fontsize=11)
    ax_d.set_ylabel("Max |dV| (log scale)", fontsize=11)
    ax_d.set_title("(d) Convergence: Error vs Cumulative Inner Iterations", fontsize=12)
    ax_d.legend(fontsize=7, loc="upper right", ncol=2)
    ax_d.grid(True, alpha=0.3)

    fig.suptitle(f"Solver Comparison: {suite_name}", fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare Full vs Adaptive Method A vs Newton-Raphson"
    )
    parser.add_argument(
        "--suite", type=str, default="quick",
        help=f"Suite name (default: quick). Valid: {VALID_SUITES}"
    )
    parser.add_argument(
        "--network", type=str, default=None,
        help="Single network name (overrides --suite)"
    )
    parser.add_argument(
        "--max-inner", type=int, default=50,
        help="Max inner iterations (default: 50)"
    )
    parser.add_argument(
        "--omega", type=float, default=1.0,
        help="Relaxation factor omega (default: 1.0)"
    )
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Show plots"
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="Save plot to file"
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"  SOLVER COMPARISON: Full vs Adaptive vs NR")
    if args.network:
        print(f"  Network: {args.network}")
    else:
        print(f"  Suite: {args.suite}")
        print(f"  Networks: {list(get_suite_networks(args.suite).keys())}")
    print(f"  Max inner: {args.max_inner}, Omega: {args.omega}")
    print(f"{'#'*80}")

    if args.network:
        all_results = test_single_network(args.network, args.max_inner, args.omega)
        suite_name = args.network
    else:
        all_results = test_suite(args.suite, args.max_inner, args.omega)
        suite_name = args.suite

    print_summary_table(all_results)

    if args.show_plot or args.save_plot:
        plot_comparison(all_results, suite_name, args.save_plot)

    if args.save_json:
        json_results = {}
        for net_name, data in all_results.items():
            json_results[net_name] = {
                "n_buses": data["n_buses"],
                "n_pv": data["n_pv"],
                "full": {
                    "converged": data["full"]["converged"],
                    "iterations": data["full"]["iterations"],
                    "time_ms": data["full"]["time_ms"],
                    "error": data["full"]["error"],
                    "inner_per_outer": data["full"]["inner_per_outer"],
                },
                "adaptive": {
                    "converged": data["adaptive"]["converged"],
                    "iterations": data["adaptive"]["iterations"],
                    "time_ms": data["adaptive"]["time_ms"],
                    "error": data["adaptive"]["error"],
                    "inner_per_outer": data["adaptive"]["inner_per_outer"],
                },
                "nr": {
                    "converged": data["nr"]["converged"],
                    "iterations": data["nr"]["iterations"],
                    "time_ms": data["nr"]["time_ms"],
                    "error": data["nr"]["error"],
                },
            }
        with open(args.save_json, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
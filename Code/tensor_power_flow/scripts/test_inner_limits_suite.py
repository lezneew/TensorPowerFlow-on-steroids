# tensor_power_flow/scripts/test_inner_limits_suite.py
"""
Test max_iter_inner_per_outer across a whole suite of networks
==============================================================

Generates comprehensive plots showing:
- Inner iterations per outer iteration (per network)
- Inner iterations vs outer iteration (all networks)
- Computing time vs inner limit (all networks)

Usage:
    python scripts/test_inner_limits_suite.py --suite quick
    python scripts/test_inner_limits_suite.py --suite salazar --show-plot
    python scripts/test_inner_limits_suite.py --suite radial --inner-limits None,10,5,2
"""

import numpy as np
import sys
import os
import warnings
import argparse
import json
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.generators.radial_network import (
    get_quick_test_networks,
    get_radial_only_networks,
    get_comprehensive_networks,
)
from tpf.generators.network_generator_salazar import (
    get_salazar_pv_networks,
    get_salazar_scaling_networks,
    get_salazar_low_vm_networks,
    get_salazar_low_rx05_networks,
    get_salazar_low_rx10_networks,
)
from tpf.generators.ieee_pegase_networks import (
    get_ieee_networks,
    get_pegase_networks,
    get_rte_networks,
    get_large_networks,
    get_all_standard_networks,
)

VALID_SUITES = [
    "quick", "radial", "salazar", "salazar_scaling",
    "salazar_low_vm", "", "salazar_low_rx10",
    "ieee", "pegase", "rte", "large", "standard", "full"
]


def estimate_m_apriori(network, s_batch, beta=0.2, v_operating=1.0, exact_below=2000):
    """
    A-priori determination of max_iter_inner_per_outer (m*) BEFORE simulation.

    Uses only K (from Y_dd) and the power vector S to compute spectral radius.

    Returns: {"rho_in": float, "m_star": int, "method": str, "worst_col": int}
    """
    Z_B = np.linalg.inv(network.Y_dd)
    K = -Z_B
    bphi = K.shape[0]

    if s_batch.ndim == 1:
        s_batch = s_batch.reshape(-1, 1)

    load_per_col = np.sum(np.abs(s_batch), axis=0)
    worst = int(np.argmax(load_per_col))
    s_col = s_batch[:, worst]

    scale = np.conj(s_col) / (v_operating ** 2)
    M = K * scale.reshape(1, -1)

    if bphi <= exact_below:
        eigenvalues = np.linalg.eigvals(M)
        rho_in = float(np.max(np.abs(eigenvalues)))
        method = "exact_eigvals"
    else:
        rho_in = float(np.max(np.sum(np.abs(M), axis=1)))
        method = "inf_norm_bound"

    # rho_in = min(rho_in, 0.999)

    m_star = int(np.ceil(np.log(beta) / np.log(rho_in)))
    # m_star = max(1, m_star)

    return {"rho_in": rho_in, "m_star": m_star, "method": method, "worst_col": worst}


def get_suite_networks(suite_name: str) -> dict:
    """Get networks for a single suite name."""
    if suite_name == "quick":
        return get_quick_test_networks()
    elif suite_name == "radial":
        return get_radial_only_networks()
    elif suite_name == "salazar":
        return get_salazar_pv_networks()
    elif suite_name == "salazar_scaling":
        return get_salazar_scaling_networks()
    elif suite_name == "salazar_low_vm":
        return get_salazar_low_vm_networks()
    elif suite_name == "salazar_low_rx05":
        return get_salazar_low_rx05_networks()
    elif suite_name == "salazar_low_rx10":
        return get_salazar_low_rx10_networks()
    elif suite_name == "ieee":
        return get_ieee_networks()
    elif suite_name == "pegase":
        return get_pegase_networks()
    elif suite_name == "rte":
        return get_rte_networks()
    elif suite_name == "large":
        return get_large_networks()
    elif suite_name == "standard":
        return get_all_standard_networks()
    elif suite_name == "full":
        return get_comprehensive_networks()
    else:
        raise ValueError(f"Unknown suite: '{suite_name}'. Valid: {VALID_SUITES}")


def test_inner_limit(net, network, max_inner, inner_limit, omega=1.0):
    """Test Method A with a specific inner limit."""
    limit_val = inner_limit if inner_limit is not None else "full"

    solver = TPFDensePVMethodA(
        tol=1e-6,
        max_iter_inner=max_inner,
        max_iter_outer=30,
        tol_pv=1e-6,
        omega=omega,
        enforce_q_lims=False,
        cold_start=False,
        max_iter_inner_per_outer=inner_limit,
    )

    try:
        result = solver.solve(network)
    except Exception as e:
        return {
            "inner_limit": limit_val,
            "converged": False,
            "outer_iters": 0,
            "inner_total": 0,
            "inner_per_outer": [],
            "pv_error": np.inf,
            "time_ms": 0.0,
        }

    pv_info = solver.pv_info

    return {
        "inner_limit": limit_val,
        "converged": result.converged,
        "outer_iters": pv_info.outer_iterations if pv_info else 0,
        "inner_total": pv_info.inner_iterations_total if pv_info else 0,
        "inner_per_outer": pv_info.inner_iterations_per_outer if pv_info else [],
        "pv_error": pv_info.pv_v_error_final if pv_info else np.inf,
        "time_ms": result.elapsed_time_s * 1000,
    }


def test_suite(suite_name: str, inner_limits: list, max_inner: int = 50):
    """Test all networks in a suite with different inner limits."""
    networks = get_suite_networks(suite_name)

    all_results = {}

    for net_name, net_info in networks.items():
        print(f"\n  Testing {net_name}...", end="", flush=True)

        net_constructor = net_info["constructor"]
        try:
            net = net_constructor()
            network = build_network_from_pandapower(net, include_pv=True)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        s_batch = network.s_nom.reshape(-1, 1)
        apr = estimate_m_apriori(network, s_batch, beta=0.03)

        results = []
        for limit in inner_limits:
            r = test_inner_limit(net, network, max_inner, limit)
            results.append(r)

        all_results[net_name] = {
            "results": results,
            "n_buses": network.n_bus_phases,
            "n_pv": network.n_pv,
            "rho_in_apriori": apr["rho_in"],
            "m_star_apriori": apr["m_star"],
            "apriori_method": apr["method"],
        }

        conv_count = sum(1 for r in results if r["converged"])
        print(f" {conv_count}/{len(results)} converged")

    return all_results


def plot_suite_results(all_results: dict, inner_limits: list, suite_name: str, save_path: str = None):
    """Create comprehensive plots for suite results."""
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

    colors = plt.cm.tab20.colors

    # ══════════════════════════════════════════════════════════════
    # Plot A: Inner iterations per outer iteration (grouped bar chart)
    # ══════════════════════════════════════════════════════════════
    ax_a = axes[0, 0]

    x = np.arange(len(network_names))
    width = 0.12

    for i, limit in enumerate(inner_limits):
        inner_totals = []
        for net_name in network_names:
            results = all_results[net_name]["results"]
            r = results[i] if i < len(results) else None
            if r and r["converged"]:
                inner_totals.append(r["inner_total"])
            else:
                inner_totals.append(0)

        offset = (i - len(inner_limits)/2 + 0.5) * width
        bars = ax_a.bar(x + offset, inner_totals, width, label=str(limit), alpha=0.8)

    ax_a.set_xlabel("Network", fontsize=11)
    ax_a.set_ylabel("Total Inner Iterations", fontsize=11)
    ax_a.set_title("(a) Total Inner Iterations per Network", fontsize=12)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([n[:15] for n in network_names], rotation=45, ha="right", fontsize=8)
    ax_a.legend(title="Inner Limit", fontsize=8, loc="upper right")
    ax_a.grid(True, alpha=0.3, axis="y")

    # ══════════════════════════════════════════════════════════════
    # Plot B: Inner iterations vs outer iteration (line plot)
    # ══════════════════════════════════════════════════════════════
    ax_b = axes[0, 1]

    for i, net_name in enumerate(network_names):
        results = all_results[net_name]["results"]
        full_result = results[0] if results and results[0]["inner_limit"] == "full" else None

        if full_result and full_result["inner_per_outer"]:
            color = colors[i % len(colors)]
            outer_iters = list(range(1, len(full_result["inner_per_outer"]) + 1))
            ax_b.plot(outer_iters, full_result["inner_per_outer"],
                     marker="o", linewidth=2, markersize=5,
                     label=net_name[:20], color=color)

    ax_b.set_xlabel("Outer Iteration", fontsize=11)
    ax_b.set_ylabel("Inner Iterations", fontsize=11)
    ax_b.set_title("(b) Inner Iterations vs Outer (full limit)", fontsize=12)
    ax_b.legend(fontsize=7, loc="upper right", ncol=2)
    ax_b.grid(True, alpha=0.3)

    # ══════════════════════════════════════════════════════════════
    # Plot C: Computing time vs inner limit (line plot)
    # ══════════════════════════════════════════════════════════════
    ax_c = axes[1, 0]

    limit_labels = [str(l) for l in inner_limits]

    for i, net_name in enumerate(network_names):
        results = all_results[net_name]["results"]
        color = colors[i % len(colors)]

        times = [r["time_ms"] for r in results]
        converged = [r["converged"] for r in results]

        for j, (t, c) in enumerate(zip(times, converged)):
            marker = "o" if c else "x"
            ax_c.loglog(j, t, marker=marker, markersize=4, color=color, alpha=0.7)

        ax_c.plot(range(len(times)), times, linewidth=1.5, color=color, alpha=0.5)

    ax_c.set_xticks(range(len(limit_labels)))
    ax_c.set_xticklabels(limit_labels)
    ax_c.set_xlabel("Inner Limit", fontsize=11)
    ax_c.set_ylabel("Time (ms)", fontsize=11)
    ax_c.set_title("(c) Computing Time vs Inner Limit", fontsize=12)
    ax_c.grid(True, alpha=0.3)

    # Add legend for first few networks only
    for i, net_name in enumerate(network_names[:5]):
        ax_c.plot([], [], color=colors[i % len(colors)], label=net_name[:20])
    ax_c.legend(fontsize=7, loc="upper right")

    # ══════════════════════════════════════════════════════════════
    # Plot D: Optimal inner limit analysis
    # ══════════════════════════════════════════════════════════════
    ax_d = axes[1, 1]

    valid_network_names = []
    optimal_limits = []
    full_times = []
    optimal_times = []
    speedups = []

    for net_name in network_names:
        results = all_results[net_name]["results"]

        full_result = results[0] if results and results[0]["inner_limit"] == "full" else None
        if not full_result or not full_result["converged"]:
            continue

        valid_network_names.append(net_name)
        full_time = full_result["time_ms"]
        full_times.append(full_time)

        converged_results = [r for r in results if r["converged"]]
        if converged_results:
            best = min(converged_results, key=lambda x: x["time_ms"])
            optimal_time = best["time_ms"]
            speedup = full_time / optimal_time if optimal_time > 0 else 1.0

            optimal_limits.append(best["inner_limit"])
            optimal_times.append(optimal_time)
            speedups.append(speedup)
        else:
            optimal_limits.append("DIV")
            optimal_times.append(full_time)
            speedups.append(1.0)

    x_pos = np.arange(len(valid_network_names))
    width = 0.35

    ax_d.bar(x_pos - width/2, full_times, width, label="Full", color="steelblue", alpha=0.8)
    ax_d.bar(x_pos + width/2, optimal_times, width, label="Optimal", color="forestgreen", alpha=0.8)

    ax_d.set_xlabel("Network", fontsize=11)
    ax_d.set_ylabel("Time (ms)", fontsize=11)
    ax_d.set_title("(d) Full vs Optimal Time", fontsize=12)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([n[:15] for n in valid_network_names], rotation=45, ha="right", fontsize=8)
    ax_d.legend(fontsize=9)
    ax_d.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Inner Iteration Limit Suite Test: {suite_name}", fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()

    return optimal_limits, full_times, optimal_times, speedups


def print_summary(all_results: dict, inner_limits: list):
    """Print summary table."""
    print(f"\n{'='*155}")
    print(f"  SUITE SUMMARY")
    print(f"{'='*155}")

    header = (f"  {'Network':<22} {'Buses':<6} {'PV':<3} {'rho_in':<8} {'m*_calc':<8} "
              f"{'m*_emp':<8} {'Full Time':<10} {'Opt Time':<10} {'Speedup':<8}")
    print(header)
    print(f"  {'-'*153}")

    network_names = list(all_results.keys())

    for net_name in network_names:
        results = all_results[net_name]["results"]
        n_buses = all_results[net_name]["n_buses"]
        n_pv = all_results[net_name]["n_pv"]
        rho_in = all_results[net_name].get("rho_in_apriori", 0)
        m_calc = all_results[net_name].get("m_star_apriori", 0)

        full_result = results[0] if results and results[0]["inner_limit"] == "full" else None
        if not full_result or not full_result["converged"]:
            print(f"  {net_name:<22} {n_buses:<6} {n_pv:<3} {'DIV':<8}")
            continue

        full_time = full_result["time_ms"]

        converged_results = [r for r in results if r["converged"]]
        if converged_results:
            best = min(converged_results, key=lambda x: x["time_ms"])
            optimal_limit = best["inner_limit"]
            optimal_time = best["time_ms"]
            speedup = full_time / optimal_time if optimal_time > 0 else 1.0
        else:
            optimal_limit = "DIV"
            optimal_time = full_time
            speedup = 1.0

        m_emp = optimal_limit if isinstance(optimal_limit, int) else ("full" if optimal_limit is None else str(optimal_limit))
        rho_str = f"{rho_in:.4f}" if rho_in else "N/A"

        print(f"  {net_name:<22} {n_buses:<6} {n_pv:<3} {rho_str:<8} {m_calc:<8} {str(m_emp):<8} "
              f"{full_time:<10.2f} {optimal_time:<10.2f} {speedup:<8.2f}x")

    print(f"{'='*155}")

    all_speedups = []
    all_full_times = []
    all_optimal_times = []

    for net_name in network_names:
        results = all_results[net_name]["results"]
        full_result = results[0] if results and results[0]["inner_limit"] == "full" else None
        if not full_result or not full_result["converged"]:
            continue

        full_time = full_result["time_ms"]
        converged_results = [r for r in results if r["converged"]]
        if converged_results:
            best = min(converged_results, key=lambda x: x["time_ms"])
            optimal_time = best["time_ms"]
            speedup = full_time / optimal_time if optimal_time > 0 else 1.0
            all_speedups.append(speedup)
            all_full_times.append(full_time)
            all_optimal_times.append(optimal_time)

    if all_speedups:
        print(f"\n  OVERALL:")
        print(f"    Networks tested: {len(all_speedups)}")
        print(f"    Avg time (full): {np.mean(all_full_times):.2f}ms")
        print(f"    Avg time (optimal): {np.mean(all_optimal_times):.2f}ms")
        print(f"    Avg speedup: {np.mean(all_speedups):.2f}x")
        print(f"    Max speedup: {max(all_speedups):.2f}x")

    print(f"{'='*140}")

    # Overall stats
    all_speedups = []
    all_full_times = []
    all_optimal_times = []

    for net_name in network_names:
        results = all_results[net_name]["results"]
        full_result = results[0] if results and results[0]["inner_limit"] == "full" else None
        if not full_result or not full_result["converged"]:
            continue

        full_time = full_result["time_ms"]
        converged_results = [r for r in results if r["converged"]]
        if converged_results:
            best = min(converged_results, key=lambda x: x["time_ms"])
            optimal_time = best["time_ms"]
            speedup = full_time / optimal_time if optimal_time > 0 else 1.0
            all_speedups.append(speedup)
            all_full_times.append(full_time)
            all_optimal_times.append(optimal_time)

    if all_speedups:
        print(f"\n  OVERALL:")
        print(f"    Networks tested: {len(all_speedups)}")
        print(f"    Avg time (full): {np.mean(all_full_times):.2f}ms")
        print(f"    Avg time (optimal): {np.mean(all_optimal_times):.2f}ms")
        print(f"    Avg speedup: {np.mean(all_speedups):.2f}x")
        print(f"    Max speedup: {max(all_speedups):.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Test inner iteration limits across a suite of networks"
    )
    parser.add_argument(
        "--suite", type=str, default="quick",
        help=f"Suite name (default: quick). Valid: {VALID_SUITES}"
    )
    parser.add_argument(
        "--max-inner", type=int, default=50,
        help="Full inner iterations for Phase 2 (default: 50)"
    )
    parser.add_argument(
        "--inner-limits", type=str, default="None,10,5,3,2,1",
        help="Comma-separated inner limits to test (default: None,20,10,5,3,2,1)"
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

    inner_limits = []
    for val in args.inner_limits.split(","):
        val = val.strip()
        if val.lower() == "none":
            inner_limits.append(None)
        else:
            try:
                inner_limits.append(int(val))
            except ValueError:
                print(f"Warning: ignoring invalid inner limit '{val}'")

    if not inner_limits:
        print("Error: no valid inner limits specified")
        return

    print(f"\n{'#'*80}")
    print(f"  SUITE INNER ITERATION LIMIT TEST: {args.suite}")
    print(f"  Networks: {list(get_suite_networks(args.suite).keys())}")
    print(f"  Inner limits: {inner_limits}")
    print(f"{'#'*80}")

    all_results = test_suite(args.suite, inner_limits, args.max_inner)

    print_summary(all_results, inner_limits)

    if args.show_plot or args.save_plot:
        plot_suite_results(all_results, inner_limits, args.suite, args.save_plot)

    if args.save_json:
        json_results = {}
        for net_name, data in all_results.items():
            json_results[net_name] = {
                "n_buses": data["n_buses"],
                "n_pv": data["n_pv"],
                "rho_in_apriori": data.get("rho_in_apriori"),
                "m_star_apriori": data.get("m_star_apriori"),
                "apriori_method": data.get("apriori_method"),
                "results": data["results"],
            }
        with open(args.save_json, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.save_json}")


if __name__ == "__main__":
    main()
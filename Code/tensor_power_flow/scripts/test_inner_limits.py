# tensor_power_flow/scripts/test_inner_limits.py
"""
Test different max_iter_inner_per_outer values for Method A
============================================================

Tests the two-phase approach: reduced inner iterations until outer converges,
then one final pass with full inner iterations.

Usage:
    python scripts/test_inner_limits.py
    python scripts/test_inner_limits.py --list
    python scripts/test_inner_limits.py --network 4bus_1pv --suite quick --show-plot
    python scripts/test_inner_limits.py --network case30 --suite ieee --inner-limits None,10,5,2
"""

import numpy as np
import sys
import os
import warnings
import argparse

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
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
    "salazar_low_vm", "salazar_low_rx05", "salazar_low_rx10",
    "ieee", "pegase", "rte", "large", "standard", "full"
]


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


def list_all_networks():
    """List all available networks from all suites."""
    print("\nAvailable networks:")
    print("=" * 80)

    for suite in VALID_SUITES:
        try:
            networks = get_suite_networks(suite)
            if networks:
                print(f"\n{suite.upper()} ({len(networks)} networks):")
                for name, info in networks.items():
                    n_pv = info.get("n_pv", "?")
                    desc = info.get("description", "")
                    print(f"  {name:<35} PV={n_pv:<2} - {desc}")
        except Exception as e:
            print(f"  Error loading {suite}: {e}")


def run_nr_reference(net):
    """Run NR solver to get reference solution."""
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
    try:
        result = nr_solver.solve_from_net(net)
        return result if result.converged else None
    except Exception:
        return None


def test_inner_limit(net, network, nr_result, max_inner, inner_limit, omega=1.0, verbose=False):
    """
    Test Method A with a specific inner limit.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network (for d_idx mapping)
    network : NetworkData
        The network to solve
    nr_result : PowerFlowResult
        NR reference result for accuracy comparison
    max_inner : int
        Full inner iterations for Phase 2
    inner_limit : int or None
        Inner iterations per outer for Phase 1
    omega : float
        Relaxation factor

    Returns
    -------
    dict : Results for this configuration
    """
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
            "error": str(e)[:50],
            "outer_iters": 0,
            "inner_total": 0,
            "inner_per_outer": [],
            "pv_error": np.inf,
            "time_ms": 0.0,
            "max_dv": np.inf,
            "pv_error_history": [],
            "inner_v_change_all": [],
            "outer_start_indices": [],
            "inner_fpi_time_ms": [],
            "total_inner_fpi_time_ms": 0.0,
        }

    pv_info = solver.pv_info

    # Calculate voltage error vs NR
    max_dv = np.inf
    if nr_result is not None and result.voltages is not None:
        # Get proper d_idx mapping
        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        pv_idx_ppc = np.where(bus_types == 2)[0]
        pq_idx_ppc = np.where(bus_types == 1)[0]
        d_idx = np.sort(np.concatenate([pq_idx_ppc, pv_idx_ppc]))

        v_tpf = result.voltages.flatten()[:len(d_idx)]
        v_nr = nr_result.voltages[:len(d_idx)]
        if len(v_tpf) == len(v_nr):
            max_dv = float(np.max(np.abs(np.abs(v_tpf) - np.abs(v_nr))))

    return {
        "inner_limit": limit_val,
        "converged": result.converged,
        "error": None,
        "outer_iters": pv_info.outer_iterations if pv_info else 0,
        "inner_total": pv_info.inner_iterations_total if pv_info else 0,
        "inner_per_outer": pv_info.inner_iterations_per_outer if pv_info else [],
        "pv_error": pv_info.pv_v_error_final if pv_info else np.inf,
        "time_ms": result.elapsed_time_s * 1000,
        "max_dv": max_dv,
        "pv_error_history": pv_info.pv_v_error_history if pv_info else [],
        "inner_v_change_all": pv_info.inner_v_change_all if pv_info else [],
        "outer_start_indices": pv_info.outer_start_indices if pv_info else [],
        "inner_fpi_time_ms": pv_info.inner_fpi_time_ms if pv_info else [],
        "total_inner_fpi_time_ms": pv_info.total_inner_fpi_time_ms if pv_info else 0.0,
    }


def print_results_table(results, network_name, max_inner):
    """Print formatted comparison table."""
    print(f"\n{'='*120}")
    print(f"  INNER ITERATION LIMIT COMPARISON: {network_name}")
    print(f"  max_iter_inner (Phase 2): {max_inner}")
    print(f"{'='*120}")

    header = (f"  {'Inner Limit':<12} {'Conv':<5} {'Outer':<6} {'Inner':<7} "
              f"{'Inner per Outer':<25} {'PV Error':<12} {'Time ms':<10} {'Inner FPI ms':<14} {'Max |dV|':<12}")
    print(header)
    print(f"  {'-'*139}")

    for r in results:
        limit_str = str(r["inner_limit"])
        conv_str = "YES" if r["converged"] else "NO"
        outer_str = str(r["outer_iters"])
        inner_str = str(r["inner_total"])

        inner_per_outer = r["inner_per_outer"]
        if inner_per_outer:
            # Highlight Phase 2: last entry should be >= max_inner if two-phase worked
            inner_outer_str = str(inner_per_outer)
        else:
            inner_outer_str = "-"

        pv_err = r["pv_error"]
        pv_err_str = f"{pv_err:.2e}" if pv_err < 100 else "-"

        time_str = f"{r['time_ms']:.2f}" if r['time_ms'] > 0 else "-"

        inner_fpi_time = r.get("total_inner_fpi_time_ms", 0.0)
        inner_fpi_str = f"{inner_fpi_time:.2f}" if inner_fpi_time > 0 else "-"

        max_dv = r["max_dv"]
        max_dv_str = f"{max_dv:.2e}" if max_dv < 100 else "-"

        status = "DIV" if not r["converged"] else "OK"

        print(f"  {limit_str:<12} {conv_str:<5} {outer_str:<6} {inner_str:<7} "
              f"{inner_outer_str:<25} {pv_err_str:<12} {time_str:<10} {inner_fpi_str:<14} {max_dv_str:<12} {status}")

    print(f"{'='*139}")

    # Summary
    converged = [r for r in results if r["converged"]]
    diverged = [r for r in results if not r["converged"]]

    print(f"\n  SUMMARY:")
    print(f"    Converged: {len(converged)}/{len(results)}")

    if converged:
        inner_iters = [r["inner_total"] for r in converged]
        times = [r["time_ms"] for r in converged]

        print(f"    Inner iterations: min={min(inner_iters)}, max={max(inner_iters)}, "
              f"avg={np.mean(inner_iters):.1f}")
        print(f"    Time (ms): min={min(times):.2f}, max={max(times):.2f}, "
              f"avg={np.mean(times):.2f}")

        # Best config
        best = min(converged, key=lambda x: x["inner_total"])
        print(f"    Best (fewest inner): inner_limit={best['inner_limit']}, "
              f"total={best['inner_total']}, time={best['time_ms']:.2f}ms")

    if diverged:
        print(f"    Diverged: {[r['inner_limit'] for r in diverged]}")


def plot_convergence(results, network_name, save_path=None):
    """Plot convergence comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping plot)")
        return

    # Get all results with history data (both converged and diverged)
    all_results = [r for r in results if r.get("pv_error_history") and r.get("inner_v_change_all")]
    if not all_results:
        print("  (no results with history to plot)")
        return

    # Find full result for speedup calculation
    full_result = next((r for r in results if r["inner_limit"] == "full"), None)
    full_time = full_result["time_ms"] if full_result and full_result["converged"] else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ══════════════════════════════════════════════════════════════
    # Plot 1 (a): PV voltage error vs outer iteration (log-log)
    # ══════════════════════════════════════════════════════════════
    ax1 = axes[0, 0]
    colors = plt.cm.tab10.colors

    for i, r in enumerate(all_results):
        errors = r["pv_error_history"]
        if not errors:
            continue
        iters = list(range(1, len(errors) + 1))
        label = f"limit={r['inner_limit']}"
        color = colors[i % len(colors)]

        # Different style for converged vs diverged
        marker = "o" if r["converged"] else "x"
        linestyle = "-" if r["converged"] else "--"
        alpha = 0.85 if r["converged"] else 0.6

        ax1.loglog(iters, errors, marker=marker, markersize=4,
                   label=label, linewidth=1.5, color=color,
                   linestyle=linestyle, alpha=alpha)

    ax1.set_xlabel("Outer Iteration l", fontsize=11)
    ax1.set_ylabel("max ||V_PV| - V_spec|| [p.u.]", fontsize=11)
    ax1.set_title("(a) PV-Spannungsfehler vs. Outer-Iteration", fontsize=12)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim(bottom=1e-7, top=1e0)
    ax1.legend(fontsize=9)

    # ══════════════════════════════════════════════════════════════
    # Plot 2 (b): Voltage change vs cumulative inner iteration (semilog)
    # ══════════════════════════════════════════════════════════════
    ax2 = axes[0, 1]

    for i, r in enumerate(all_results):
        v_changes = r["inner_v_change_all"]
        outer_starts = r["outer_start_indices"]
        if not v_changes:
            continue

        x = list(range(1, len(v_changes) + 1))
        label = f"limit={r['inner_limit']}"
        color = colors[i % len(colors)]

        # Different style for converged vs diverged
        marker = "o" if r["converged"] else "x"
        linestyle = "-" if r["converged"] else "--"
        alpha = 0.8 if r["converged"] else 0.5

        ax2.semilogy(x, v_changes, marker=marker, markersize=3,
                     label=label, linewidth=1.2, color=color,
                     linestyle=linestyle, alpha=alpha)

        # Vertical lines at outer iteration boundaries (skip first)
        for idx_start in outer_starts[1:]:
            if idx_start < len(v_changes):
                ax2.axvline(x=idx_start + 1, color=color, linestyle=":", linewidth=0.4, alpha=0.3)

    ax2.set_xlabel("Cumulative Inner Iteration", fontsize=11)
    ax2.set_ylabel("max ||V_new| - |V_old|| [p.u.]", fontsize=11)
    ax2.set_title("(b) Network Convergence (all buses)", fontsize=12)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(bottom=1e-12, top=1e1)
    ax2.set_xlim(left=0.5)
    ax2.legend(fontsize=9)

    # ══════════════════════════════════════════════════════════════
    # Plot 3 (c): Computing time vs inner limit (line plot)
    # ══════════════════════════════════════════════════════════════
    ax3 = axes[1, 0]

    limit_labels = [str(r["inner_limit"]) for r in results]
    times = [r["time_ms"] for r in results]
    converged = [r["converged"] for r in results]

    x_vals = range(len(limit_labels))

    # Line plot connecting all points
    ax3.plot(x_vals, times, marker="o", markersize=8, linewidth=2,
             color="steelblue", label="Time")

    # Add markers for converged (circle) vs diverged (x)
    for i, (t, c) in enumerate(zip(times, converged)):
        marker = "o" if c else "x"
        color = "green" if c else "red"
        ax3.plot(i, t, marker=marker, markersize=10, color=color,
                markeredgecolor="black", markeredgewidth=1.5)

    ax3.set_xticks(x_vals)
    ax3.set_xticklabels(limit_labels)
    ax3.set_xlabel("Inner Limit", fontsize=11)
    ax3.set_ylabel("Computing Time (ms)", fontsize=11)
    ax3.set_title("(c) Computing Time vs Inner Limit", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # ══════════════════════════════════════════════════════════════
    # Plot 4 (d): Speedup vs full configuration (line plot)
    # ══════════════════════════════════════════════════════════════
    ax4 = axes[1, 1]

    if full_time and full_time > 0:
        speedups = []
        for r in results:
            if r["converged"] and r["time_ms"] > 0:
                speedups.append(full_time / r["time_ms"])
            else:
                speedups.append(0.0)

        # Line plot
        ax4.plot(x_vals, speedups, marker="s", markersize=8, linewidth=2,
                 color="darkorange", label="Speedup")

        # Add markers for converged vs diverged
        for i, (s, c) in enumerate(zip(speedups, converged)):
            marker = "o" if c else "x"
            color = "green" if c else "red"
            ax4.plot(i, s, marker=marker, markersize=5, color=color, markeredgewidth=1.5)

        # Reference line at y=1 (no speedup)
        ax4.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5,
                   alpha=0.7, label="No speedup (1.0x)")

        ax4.set_xticks(x_vals)
        ax4.set_xticklabels(limit_labels)
        ax4.set_xlabel("Inner Limit", fontsize=11)
        ax4.set_ylabel("Speedup (full time / limited time)", fontsize=11)
        ax4.set_title("(d) Speedup vs Full Configuration", fontsize=12)
        ax4.grid(True, alpha=0.3)

        # Find and annotate best speedup
        max_speedup = max(speedups)
        best_idx = speedups.index(max_speedup)
        # ax4.annotate(f"Max: {max_speedup:.2f}x", xy=(best_idx, max_speedup),
        #             xytext=(best_idx + 0.3, max_speedup * 1.1),
        #             fontsize=10, fontweight="bold",
        #             arrowprops=dict(arrowstyle="->", color="black", lw=1))
    else:
        ax4.text(0.5, 0.5, "Full time not available\n(cannot compute speedup)",
                ha="center", va="center", fontsize=12, transform=ax4.transAxes)
        ax4.set_title("(d) Speedup vs Full Configuration", fontsize=12)

    fig.suptitle(f"Inner Iteration Limit Test: {network_name}", fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Test different max_iter_inner_per_outer values for Method A"
    )
    parser.add_argument(
        "--network", type=str, default="4bus_1pv",
        help="Network name (default: 4bus_1pv)"
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
    )in me
    parser.add_argument(
        "--show-plot", action="store_true",
        help="Show convergence plot"
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="Save plot to file"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available networks and exit"
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        list_all_networks()
        return

    # Parse inner limits
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

    # Load network
    try:
        networks = get_suite_networks(args.suite)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.network not in networks:
        print(f"Error: network '{args.network}' not found in suite '{args.suite}'")
        print(f"Available networks in {args.suite}: {list(networks.keys())}")
        return

    net_info = networks[args.network]
    net_constructor = net_info["constructor"]

    print(f"\nLoading network: {args.network} from {args.suite}")
    print(f"  Description: {net_info.get('description', 'N/A')}")
    print(f"  Expected PV: {net_info.get('n_pv', 'N/A')}")
    print(f"  Testing inner limits: {inner_limits}")
    print(f"  Max inner (Phase 2): {args.max_inner}")

    # Build network
    try:
        net = net_constructor()
    except Exception as e:
        print(f"Error creating network: {e}")
        return

    try:
        network = build_network_from_pandapower(net, include_pv=True)
    except Exception as e:
        print(f"Error building network: {e}")
        return

    print(f"  Network: {network.n_bus_phases} buses, {network.n_pv} PV")

    # Run NR reference
    print("\nRunning NR reference...")
    nr_result = run_nr_reference(net)
    if nr_result is None:
        print("  Warning: NR did not converge, skipping accuracy comparison")
    else:
        print(f"  NR converged in {nr_result.iterations} iterations")

    # Test each inner limit
    print("\nTesting inner limits...")
    results = []

    for limit in inner_limits:
        limit_str = "None" if limit is None else str(limit)
        print(f"  Testing inner_limit = {limit_str}... ", end="", flush=True)

        r = test_inner_limit(net, network, nr_result, args.max_inner, limit, args.omega)
        results.append(r)

        status = "CONV" if r["converged"] else "DIV"
        print(f"{status} (outer={r['outer_iters']}, inner={r['inner_total']}, "
              f"time={r['time_ms']:.2f}ms)")

    # Print results table
    print_results_table(results, args.network, args.max_inner)

    # Plot if requested
    if args.show_plot or args.save_plot:
        plot_convergence(results, args.network, args.save_plot)


if __name__ == "__main__":
    main()
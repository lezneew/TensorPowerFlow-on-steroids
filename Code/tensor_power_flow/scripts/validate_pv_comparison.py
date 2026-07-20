# tensor_power_flow/scripts/validate_pv_comparison.py
"""
Vergleich: NR, Methode A und Methode B auf einem einzelnen Netz
================================================================

Vergleicht drei Solver auf einem Netz und plottet Konvergenz:
- Newton-Raphson (NR) - pandapower Referenz
- Methode A - TPF mit äußerer Q-Schleife
- Methode B - TPF mit Single-Pass + Q-Korrektur

Aufruf:
    python scripts/validate_pv_comparison.py
    python scripts/validate_pv_comparison.py --network 4bus_1pv --suite quick
    python scripts/validate_pv_comparison.py --suite salazar_scaling --list
    python scripts/validate_pv_comparison.py --network salazar_34bus_3pv --suite salazar --save comparison.png
"""

import numpy as np
import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.tpf_pv_method_b import TPFDensePVMethodB
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


def run_all_solvers(net_constructor, name, omega=1.0, omega_q=1.0, verbose=True):
    """
    Führt alle drei Solver aus und sammelt Ergebnisse.
    """
    results = {
        "name": name,
        "nr": {"converged": False, "iterations": 0, "time_ms": 0.0, "voltages": None, "error": None},
        "method_a": {"converged": False, "iterations": 0, "time_ms": 0.0, "voltages": None, "error": None,
                     "pv_errors": [], "v_changes": []},
        "method_b": {"converged": False, "iterations": 0, "time_ms": 0.0, "voltages": None, "error": None,
                     "pv_errors": [], "v_changes": []},
        "d_idx": None,
    }

    # 1. Netz erzeugen
    try:
        net = net_constructor()
    except Exception as e:
        results["nr"]["error"] = f"Constructor: {str(e)[:50]}"
        if verbose:
            print(f"  X {name}: FEHLER (Constructor): {e}")
        return results

    # 2. NR-Referenz
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=200)
    try:
        nr_result = nr_solver.solve_from_net(net)
        results["nr"]["converged"] = nr_result.converged
        results["nr"]["iterations"] = nr_result.iterations
        results["nr"]["time_ms"] = nr_result.elapsed_time_s * 1000
        results["nr"]["voltages"] = nr_result.voltages
    except Exception as e:
        results["nr"]["error"] = f"NR: {str(e)[:50]}"
        if verbose:
            print(f"  X {name}: FEHLER (NR): {e}")
        return results

    if not nr_result.converged:
        results["nr"]["error"] = "NR divergiert"
        if verbose:
            print(f"  X {name}: NR divergiert")
        return results

    # 3. Netzwerk aufbauen
    try:
        network = build_network_from_pandapower(net, include_pv=True)
    except Exception as e:
        results["nr"]["error"] = f"Builder: {str(e)[:50]}"
        if verbose:
            print(f"  X {name}: FEHLER (Builder): {e}")
        return results

    n_pv = network.n_pv
    if verbose:
        print(f"  Netz: {name} | Busse: {network.n_bus_phases} | PV: {n_pv}")

    # Compute d_idx for voltage comparison
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    pv_idx_ppc = np.where(bus_types == 2)[0]
    pq_idx_ppc = np.where(bus_types == 1)[0]
    results["d_idx"] = np.sort(np.concatenate([pq_idx_ppc, pv_idx_ppc]))

    # 4. Methode A
    solver_a = TPFDensePVMethodA(
        tol=1e-8, max_iter_inner=20, max_iter_outer=50,
        omega=omega, enforce_q_lims=False,
    )
    try:
        result_a = solver_a.solve(network)
        results["method_a"]["converged"] = result_a.converged
        results["method_a"]["iterations"] = result_a.iterations
        results["method_a"]["time_ms"] = result_a.elapsed_time_s * 1000
        results["method_a"]["voltages"] = result_a.voltages.flatten()

        if solver_a.pv_info:
            results["method_a"]["pv_errors"] = solver_a.pv_info.pv_v_error_history or []
            results["method_a"]["v_changes"] = solver_a.pv_info.v_change_history or solver_a.pv_info.inner_v_change_all or []
    except Exception as e:
        results["method_a"]["error"] = f"Method A: {str(e)[:50]}"
        if verbose:
            print(f"    Method A FEHLER: {e}")

    # 5. Methode B
    solver_b = TPFDensePVMethodB(
        tol=1e-8, max_iter=200,
        tol_pv=1e-6, omega=omega, omega_q=omega_q, enforce_q_lims=False,
    )
    try:
        result_b = solver_b.solve(network)
        results["method_b"]["converged"] = result_b.converged
        results["method_b"]["iterations"] = result_b.iterations
        results["method_b"]["time_ms"] = result_b.elapsed_time_s * 1000
        results["method_b"]["voltages"] = result_b.voltages.flatten()

        if solver_b.pv_info:
            results["method_b"]["pv_errors"] = solver_b.pv_info.pv_v_error_history or []
            results["method_b"]["v_changes"] = solver_b.pv_info.voltage_change_history or []
    except Exception as e:
        results["method_b"]["error"] = f"Method B: {str(e)[:50]}"
        if verbose:
            print(f"    Method B FEHLER: {e}")

    return results


def print_comparison_table(results, tol_pass=1e-4):
    """Druckt Vergleichstabelle."""
    print(f"\n{'='*100}")
    print(f"  VERGLEICH: {results['name']}")
    print(f"{'='*100}")

    # Header
    hdr = (f"  {'Solver':<12} {'Conv':<5} {'Iter':<6} {'Zeit[ms]':<10} "
           f"{'max|dV|':<12} {'max|dTheta|':<12} {'Status'}")
    print(hdr)
    print(f"  {'-'*98}")

    nr = results["nr"]
    ma = results["method_a"]
    mb = results["method_b"]
    d_idx = results.get("d_idx")

    def get_errors(tpf_v, nr_v, d_idx):
        if tpf_v is None or nr_v is None or d_idx is None:
            return np.inf, np.inf
        nr_v_d = nr_v[d_idx]
        if tpf_v.shape[0] != nr_v_d.shape[0]:
            return np.inf, np.inf
        mag_err = np.abs(np.abs(tpf_v) - np.abs(nr_v_d))
        angle_err = np.abs(np.angle(tpf_v, deg=True) - np.angle(nr_v_d, deg=True))
        return float(np.max(mag_err)), float(np.max(angle_err))

    # NR
    print(f"  {'NR':<12} {'Yes' if nr['converged'] else 'No':<5} "
          f"{nr['iterations']:<6} {nr['time_ms']:<10.2f} {'—':<12} {'—':<12} {'REF':<6}")

    # Method A
    ma_max_v, ma_max_theta = get_errors(ma.get("voltages"), nr.get("voltages"), d_idx)
    ma_status = "PASS" if ma_max_v < tol_pass and ma['converged'] else "FAIL"
    if ma.get("error"):
        ma_status = "ERR"
    elif not ma['converged']:
        ma_status = "DIV"
    ma_v_str = f"{ma_max_v:.2e}" if ma_max_v < 100 else "—"
    ma_t_str = f"{ma_max_theta:.4f}" if ma_max_theta < 100 else "—"
    print(f"  {'Methode A':<12} {'Yes' if ma['converged'] else 'No':<5} "
          f"{ma['iterations']:<6} {ma['time_ms']:<10.2f} {ma_v_str:<12} {ma_t_str:<12} {ma_status:<6}")

    # Method B
    mb_max_v, mb_max_theta = get_errors(mb.get("voltages"), nr.get("voltages"), d_idx)
    mb_status = "PASS" if mb_max_v < tol_pass and mb['converged'] else "FAIL"
    if mb.get("error"):
        mb_status = "ERR"
    elif not mb['converged']:
        mb_status = "DIV"
    mb_v_str = f"{mb_max_v:.2e}" if mb_max_v < 100 else "—"
    mb_t_str = f"{mb_max_theta:.4f}" if mb_max_theta < 100 else "—"
    print(f"  {'Methode B':<12} {'Yes' if mb['converged'] else 'No':<5} "
          f"{mb['iterations']:<6} {mb['time_ms']:<10.2f} {mb_v_str:<12} {mb_t_str:<12} {mb_status:<6}")

    print(f"{'='*100}")


def plot_comparison(results, omega, omega_q, save_path=None):
    """Erstellt 2x2 Vergleichsplot."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")

    colors = {"nr": "tab:red", "method_a": "tab:blue", "method_b": "tab:green"}
    labels = {"nr": "NR (pandapower)", "method_a": "Methode A", "method_b": "Methode B"}

    # (0,0) PV voltage error vs iteration
    ax00 = axes[0, 0]
    for key in ["method_a", "method_b"]:
        data = results[key]
        errors = data.get("pv_errors", [])
        if errors:
            iters = list(range(1, len(errors) + 1))
            ax00.plot(
                iters, errors,
                color=colors[key], marker="o", markersize=4,
                linestyle="-", linewidth=1.5, alpha=0.8,
                label=labels[key],
            )

    ax00.axhline(y=1e-6, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="tol_pv=1e-6")
    ax00.set_xlabel("Iteration", fontsize=11)
    ax00.set_ylabel("max ||V_PV| - V_spec|| [p.u.]", fontsize=11)
    ax00.set_title("(a) PV-Spannungsfehler vs. Iteration", fontsize=12)
    ax00.grid(True, which="both", alpha=0.3)
    ax00.set_ylim(bottom=1e-8, top=1e0)
    ax00.legend(fontsize=9)

    # (0,1) Voltage change vs iteration
    ax01 = axes[0, 1]
    for key in ["method_a", "method_b"]:
        data = results[key]
        v_changes = data.get("v_changes", [])
        if v_changes:
            iters = list(range(1, len(v_changes) + 1))
            ax01.semilogy(
                iters, v_changes,
                color=colors[key], marker="o", markersize=2,
                linestyle="-", linewidth=1.0, alpha=0.8,
                label=labels[key],
            )

    ax01.axhline(y=1e-8, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax01.set_xlabel("Iteration", fontsize=11)
    ax01.set_ylabel("max ||V_new| - |V_old|| [p.u.]", fontsize=11)
    ax01.set_title("(b) Netzwerk-Konvergenz (alle Busse)", fontsize=12)
    ax01.grid(True, which="both", alpha=0.3)
    ax01.set_ylim(bottom=1e-12, top=1e1)
    ax01.legend(fontsize=9)

    # (1,0) Timing comparison
    ax10 = axes[1, 0]
    solvers = ["NR", "Methode A", "Methode B"]
    times = [
        results["nr"]["time_ms"],
        results["method_a"]["time_ms"],
        results["method_b"]["time_ms"],
    ]
    bar_colors = [colors["nr"], colors["method_a"], colors["method_b"]]

    bars = ax10.bar(solvers, times, color=bar_colors, alpha=0.7, edgecolor="black")
    ax10.set_ylabel("Rechenzeit [ms]", fontsize=11)
    ax10.set_title("(c) Rechenzeit Vergleich", fontsize=12)
    ax10.grid(True, axis="y", alpha=0.3)

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax10.annotate(f"{t:.2f}ms",
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points",
                      ha="center", va="bottom", fontsize=9)

    # (1,1) Final voltage profile (bar chart)
    ax11 = axes[1, 1]
    nr_v = results["nr"].get("voltages")
    d_idx = results.get("d_idx")
    if nr_v is not None and d_idx is not None:
        nr_v_d = nr_v[d_idx]
        n_bus = len(nr_v_d)
        x = np.arange(n_bus)
        width = 0.25

        mag_nr = np.abs(nr_v_d)
        ma_v = results["method_a"].get("voltages")
        mb_v = results["method_b"].get("voltages")

        valid_ma = ma_v is not None and len(ma_v) == n_bus
        valid_mb = mb_v is not None and len(mb_v) == n_bus

        mag_ma = np.abs(ma_v) if valid_ma else np.zeros(n_bus)
        mag_mb = np.abs(mb_v) if valid_mb else np.zeros(n_bus)

        ax11.bar(x - width, mag_nr, width, label="NR", color=colors["nr"], alpha=0.7)
        if valid_ma:
            ax11.bar(x, mag_ma, width, label="Methode A", color=colors["method_a"], alpha=0.7)
        if valid_mb:
            ax11.bar(x + width, mag_mb, width, label="Methode B", color=colors["method_b"], alpha=0.7)

        ax11.set_xlabel("Bus Index (d-block)", fontsize=11)
        ax11.set_ylabel("|V| [p.u.]", fontsize=11)
        ax11.set_title("(d) Spannungsprofil Vergleich", fontsize=12)
        ax11.grid(True, axis="y", alpha=0.3)
        ax11.legend(fontsize=9)
        ax11.set_ylim(0.9, 1.1)

    fig.suptitle(
        f"Vergleich: NR vs Methode A vs Methode B — {results['name']} | "
        f"omega={omega}, omega_q={omega_q}",
        fontsize=14, y=1.02,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Vergleich: NR, Methode A und Methode B auf einem Netz"
    )
    parser.add_argument(
        "--network", type=str, default="4bus_1pv",
        help="Network name to test (default: 4bus_1pv)"
    )
    parser.add_argument(
        "--suite", type=str, default="quick",
        help=f"Suite to load network from (default: quick). Valid: {VALID_SUITES}"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available networks in suite and exit"
    )
    parser.add_argument(
        "--omega", type=float, default=1.0,
        help="Relaxation factor omega (default: 1.0)"
    )
    parser.add_argument(
        "--omega-q", type=float, default=1.0,
        help="Q-relaxation factor for Method B (default: 1.0)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save plot to file path"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4,
        help="Tolerance for PASS/FAIL (default: 1e-4)"
    )

    args = parser.parse_args()

    # Validate suite
    if args.suite not in VALID_SUITES:
        print(f"FEHLER: Unbekannte Suite '{args.suite}'")
        print(f"Valid suites: {VALID_SUITES}")
        return 1

    # Load networks
    networks = get_suite_networks(args.suite)

    # List networks if requested
    if args.list:
        print(f"\nVerfügbare Netzwerke in Suite '{args.suite}':")
        print("-" * 50)
        for i, name in enumerate(sorted(networks.keys())):
            print(f"  {i+1:2d}. {name}")
        print("-" * 50)
        print(f"  Gesamt: {len(networks)} Netzwerke")
        return 0

    # Find network
    if args.network not in networks:
        print(f"FEHLER: Netzwerk '{args.network}' nicht gefunden in Suite '{args.suite}'")
        print(f"Verfügbare Netzwerke: {list(networks.keys())}")
        return 1

    net_info = networks[args.network]
    net_constructor = net_info["constructor"]

    print(f"\n{'='*80}")
    print(f"  Vergleich: NR vs Methode A vs Methode B")
    print(f"  Netzwerk: {args.network} (Suite: {args.suite})")
    print(f"  Parameter: omega={args.omega}, omega_q={args.omega_q}")
    print(f"{'='*80}")

    # Run all solvers
    results = run_all_solvers(
        net_constructor,
        args.network,
        omega=args.omega,
        omega_q=args.omega_q,
        verbose=True
    )

    # Print comparison table
    print_comparison_table(results, tol_pass=args.tol)

    # Plot
    if not args.save or args.save != "":
        plot_comparison(results, args.omega, args.omega_q, save_path=args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
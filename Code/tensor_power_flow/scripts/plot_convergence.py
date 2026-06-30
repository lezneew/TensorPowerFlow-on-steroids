# tensor_power_flow/scripts/plot_convergence.py
"""
Konvergenz-Plot: PV |V|-Fehler über Outer-Iterationen für alle Testnetze
=========================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.generators.radial_network import (
    TEST_NETWORKS,
    get_radial_only_networks,
)


def collect_convergence_data(networks: dict, omega: float = 1.0):
    """
    Führt Methode A für alle Netze aus und sammelt die Fehlerhistorie.

    Returns
    -------
    data : list[dict] mit keys:
        - name, eta, n_pv, converged
        - pv_v_errors: list[float] (PV |V|-Fehler pro Outer-Iteration)
        - inner_tols: list[float] (Inner-FPI Toleranz pro Outer)
    """
    data = []

    solver = TPFDensePVMethodA(
        tol=1e-8,
        max_iter_inner=50,
        max_iter_outer=50,
        tol_pv=1e-6,
        omega=omega,
        enforce_q_lims=False,
    )

    for name, info in networks.items():
        print(f"  Rechne: {name}...", end=" ")

        try:
            net = info["constructor"]()
            network = build_network_from_pandapower(net, include_pv=True)

            if network.n_pv == 0:
                print("keine PV → übersprungen")
                continue

            # η berechnen
            Z_B = np.linalg.inv(network.Y_dd)
            scaling = np.conj(network.s_nom)
            M = Z_B * scaling.reshape(1, -1)
            eta = np.max(np.sum(np.abs(M), axis=0))

            # Solve
            result = solver.solve(network)

            if solver.pv_info and solver.pv_info.pv_v_error_history:
                record = {
                    "name": name,
                    "eta": eta,
                    "n_pv": network.n_pv,
                    "n_bus": network.n_bus_phases,
                    "converged": result.converged,
                    "outer_iter": solver.pv_info.outer_iterations,
                    "pv_v_errors": solver.pv_info.pv_v_error_history,
                    "inner_tols": solver.pv_info.v_change_history,
                    "inner_per_outer": solver.pv_info.inner_iterations_per_outer,
                }
                data.append(record)
                status = "✓" if result.converged else "✗"
                print(f"{status} ({record['outer_iter']} outer)")
            else:
                print("keine Historie verfügbar")

        except Exception as e:
            print(f"FEHLER: {e}")

    return data


def plot_convergence(data: list[dict], save_path: str = None):
    """
    Erstellt den Konvergenz-Plot.

    Subplot 1: PV |V|-Fehler vs. Outer-Iteration (alle Netze)
    Subplot 2: Kumulative Inner-Iterationen vs. PV |V|-Fehler
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Farbschema ──
    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(data), 1)) for i in range(len(data))]

    # Marker für konvergiert/divergiert
    marker_conv = "o"
    marker_div = "x"

    # ══════════════════════════════════════════════════════════════
    #  Plot 1: PV |V|-Fehler vs. Outer-Iteration
    # ══════════════════════════════════════════════════════════════
    ax1 = axes[0]

    for i, rec in enumerate(data):
        errors = rec["pv_v_errors"]
        iters = list(range(1, len(errors) + 1))
        marker = marker_conv if rec["converged"] else marker_div
        linestyle = "-" if rec["converged"] else "--"

        label = (f"{rec['name']} "
                 f"(η={rec['eta']:.2f}, PV={rec['n_pv']})")

        ax1.semilogy(
            iters, errors,
            color=colors[i],
            marker=marker,
            markersize=4,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
            alpha=0.85,
        )

    # Toleranzlinie
    ax1.axhline(y=1e-6, color="green", linestyle=":", linewidth=1.5,
                label="tol_pv = 1e-6")
    ax1.axhline(y=1e-4, color="orange", linestyle=":", linewidth=1.0,
                label="1e-4 (PASS-Kriterium)")

    ax1.set_xlabel("Outer-Iteration ℓ", fontsize=12)
    ax1.set_ylabel("max |V_PV| - |V_spec|| [p.u.]", fontsize=12)
    ax1.set_title("Methode A: PV-Spannungsfehler vs. Outer-Iteration", fontsize=13)
    ax1.legend(fontsize=8, loc="upper right", ncol=1)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(left=0.5)
    ax1.set_ylim(bottom=1e-8, top=1e1)

    # ══════════════════════════════════════════════════════════════
    #  Plot 2: PV |V|-Fehler vs. kumulative Inner-Iterationen
    # ══════════════════════════════════════════════════════════════
    ax2 = axes[1]

    for i, rec in enumerate(data):
        errors = rec["pv_v_errors"]
        inner_per_outer = rec["inner_per_outer"]

        # Kumulative innere Iterationen
        cum_inner = np.cumsum(inner_per_outer[:len(errors)])

        marker = marker_conv if rec["converged"] else marker_div
        linestyle = "-" if rec["converged"] else "--"

        label = f"{rec['name']} ({rec['n_bus']} Busse)"

        ax2.semilogy(
            cum_inner, errors,
            color=colors[i],
            marker=marker,
            markersize=4,
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
            alpha=0.85,
        )

    ax2.axhline(y=1e-6, color="green", linestyle=":", linewidth=1.5,
                label="tol_pv = 1e-6")

    ax2.set_xlabel("Kumulative Inner-Iterationen (Gesamt-FPI-Schritte)", fontsize=12)
    ax2.set_ylabel("max ||V_PV| - |V_spec|| [p.u.]", fontsize=12)
    ax2.set_title("Methode A: PV-Fehler vs. Rechenaufwand", fontsize=13)
    ax2.legend(fontsize=8, loc="upper right", ncol=1)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(bottom=1e-8, top=1e1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")

    plt.show()


def plot_convergence_single(data: list[dict], save_path: str = None):
    """
    Einzelner, übersichtlicher Plot: PV |V|-Fehler vs. Outer-Iteration.
    Farbcodiert nach Konvergenz-Status.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Separate Listen für konvergiert / divergiert
    conv_data = [d for d in data if d["converged"]]
    div_data = [d for d in data if not d["converged"]]

    cmap_conv = plt.cm.Greens
    cmap_div = plt.cm.Reds

    # ── Konvergierte Netze (grün-Töne) ──
    for i, rec in enumerate(conv_data):
        errors = rec["pv_v_errors"]
        iters = list(range(1, len(errors) + 1))
        color = cmap_conv(0.4 + 0.5 * i / max(len(conv_data), 1))

        ax.semilogy(
            iters, errors,
            color=color,
            marker="o",
            markersize=3,
            linestyle="-",
            linewidth=1.8,
            label=f"✓ {rec['name']} (η={rec['eta']:.2f})",
            alpha=0.9,
        )

    # ── Divergierte Netze (rot-Töne) ──
    for i, rec in enumerate(div_data):
        errors = rec["pv_v_errors"]
        iters = list(range(1, len(errors) + 1))
        color = cmap_div(0.4 + 0.5 * i / max(len(div_data), 1))

        ax.semilogy(
            iters, errors,
            color=color,
            marker="x",
            markersize=5,
            linestyle="--",
            linewidth=1.5,
            label=f"✗ {rec['name']} (η={rec['eta']:.2f})",
            alpha=0.8,
        )

    # ── Referenzlinien ──
    ax.axhline(y=1e-6, color="blue", linestyle=":", linewidth=2.0,
               label="tol_pv = 1e-6 (Konvergenzkriterium)")
    ax.axhline(y=1e-4, color="darkorange", linestyle="-.", linewidth=1.5,
               label="1e-4 (PASS-Schwelle)")

    # ── Formatierung ──
    ax.set_xlabel("Äußere Iteration ℓ", fontsize=13)
    ax.set_ylabel("max ||V_PV,k| - V_spec,k|| [p.u.]", fontsize=13)
    ax.set_title(
        "Konvergenzverhalten Methode A (Äußere Q-Schleife)\n"
        "Alle Testnetze, ω = 1.0",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)
    ax.set_xlim(left=0.5)
    ax.set_ylim(bottom=1e-8, top=2e0)
    ax.set_xticks(range(0, 55, 5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")

    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Konvergenz-Plot Methode A")
    parser.add_argument("--suite", choices=["quick", "radial", "full"],
                        default="radial")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--save", type=str, default=None,
                        help="Pfad zum Speichern des Plots (z.B. convergence.png)")
    parser.add_argument("--single", action="store_true",
                        help="Nur einen einzelnen Plot statt Doppelplot")
    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  KONVERGENZ-PLOT: Methode A — Alle Testnetze                  ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    # Netze laden
    if args.suite == "quick":
        from tpf.generators.radial_network import get_quick_test_networks
        networks = get_quick_test_networks()
    elif args.suite == "radial":
        networks = get_radial_only_networks()
    else:
        networks = TEST_NETWORKS

    print(f"\n  Suite: '{args.suite}', ω = {args.omega}, {len(networks)} Netze\n")

    # Daten sammeln
    data = collect_convergence_data(networks, omega=args.omega)

    if not data:
        print("\n  ✗ Keine Daten gesammelt — Abbruch")
        return

    # Statistik
    n_conv = sum(1 for d in data if d["converged"])
    n_div = sum(1 for d in data if not d["converged"])
    print(f"\n  Ergebnis: {n_conv} konvergiert, {n_div} divergiert")

    # Plot
    save_path = args.save or f"convergence_method_a_omega{args.omega}.png"

    if args.single:
        plot_convergence_single(data, save_path=save_path)
    else:
        plot_convergence(data, save_path=save_path)


if __name__ == "__main__":
    main()
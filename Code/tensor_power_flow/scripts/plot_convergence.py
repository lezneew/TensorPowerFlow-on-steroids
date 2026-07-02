# tensor_power_flow/scripts/plot_convergence.py
"""
Konvergenz-Plot: PV |V|-Fehler und Gesamtnetz-Fehler über Iterationen
=======================================================================

Plot 1: PV |V|-Fehler vs. Outer-Iteration (nur PV-Knoten Sollwertabweichung)
Plot 2: Gesamtnetz max(||V_new|-|V_old||) vs. kumulative Inner-Iteration
         → zeigt Konvergenzverhalten ALLER Busse (PQ + PV)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
        - name, eta, n_pv, n_bus, converged
        - pv_v_errors: list[float] (PV |V|-Fehler pro Outer-Iteration)
        - inner_tols: list[float] (Inner-FPI Toleranz pro Outer)
        - inner_per_outer: list[int]
        - inner_v_change_all: list[float] (Gesamtnetz-Fehler pro Inner-Iter)
        - outer_start_indices: list[int] (wo Outer-Iter in der flachen Liste starten)
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
                pv_info = solver.pv_info
                record = {
                    "name": name,
                    "eta": eta,
                    "n_pv": network.n_pv,
                    "n_bus": network.n_bus_phases,
                    "converged": result.converged,
                    "outer_iter": pv_info.outer_iterations,
                    "pv_v_errors": pv_info.pv_v_error_history,
                    "inner_tols": pv_info.v_change_history,
                    "inner_per_outer": pv_info.inner_iterations_per_outer,
                    # NEU: Gesamtnetz-Fehler (alle Busse)
                    "inner_v_change_all": pv_info.inner_v_change_all,
                    "outer_start_indices": pv_info.outer_start_indices,
                }
                data.append(record)
                status = "✓" if result.converged else "✗"
                print(f"{status} ({record['outer_iter']} outer, "
                      f"{pv_info.inner_iterations_total} inner total)")
            else:
                print("keine Historie verfügbar")

        except Exception as e:
            print(f"FEHLER: {e}")

    return data


def plot_convergence(data: list[dict], save_path: str = None):
    """
    Erstellt den Konvergenz-Plot.

    Subplot 1: PV |V|-Fehler vs. Outer-Iteration (PV-Knoten Abweichung)
    Subplot 2: Gesamtnetz max(|ΔV|) vs. kumulative Inner-Iteration (ALLE Busse)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Farbschema ──
    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(data), 1)) for i in range(len(data))]

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

    ax1.axhline(y=1e-6, color="green", linestyle=":", linewidth=1.5,
                label="tol_pv = 1e-6")
    ax1.axhline(y=1e-4, color="orange", linestyle=":", linewidth=1.0,
                label="1e-4 (PASS-Kriterium)")

    ax1.set_xlabel("Äußere Iteration ℓ", fontsize=12)
    ax1.set_ylabel("max ||V_PV| - V_spec|| [p.u.]", fontsize=12)
    ax1.set_title("PV-Spannungsfehler vs. Outer-Iteration", fontsize=13)
    ax1.legend(fontsize=8, loc="upper right", ncol=1)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(left=0.5)
    ax1.set_ylim(bottom=1e-8, top=1e1)

    # ══════════════════════════════════════════════════════════════
    #  Plot 2: Gesamtnetz-Fehler vs. kumulative Inner-Iteration
    #           max(||V_new| - |V_old||) über ALLE Busse (PQ + PV)
    # ══════════════════════════════════════════════════════════════
    ax2 = axes[1]

    for i, rec in enumerate(data):
        v_changes = rec["inner_v_change_all"]
        outer_starts = rec["outer_start_indices"]

        if not v_changes:
            continue

        # X-Achse: kumulative Inner-Iteration (1-basiert)
        x = list(range(1, len(v_changes) + 1))

        marker = marker_conv if rec["converged"] else marker_div
        linestyle = "-" if rec["converged"] else "--"

        label = f"{rec['name']} ({rec['n_bus']} Busse, PV={rec['n_pv']})"

        ax2.semilogy(
            x, v_changes,
            color=colors[i],
            marker=marker,
            markersize=2,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
            alpha=0.8,
        )

        # Markiere Outer-Iteration Grenzen mit vertikalen Linien
        for idx_start in outer_starts[1:]:  # Erste überspringen (ist 0)
            if idx_start < len(v_changes):
                ax2.axvline(
                    x=idx_start + 1,
                    color=colors[i],
                    linestyle=":",
                    linewidth=0.5,
                    alpha=0.3,
                )

    ax2.axhline(y=1e-8, color="green", linestyle=":", linewidth=1.5,
                label="tol_inner = 1e-8")
    ax2.axhline(y=1e-6, color="blue", linestyle="-.", linewidth=1.0,
                label="1e-6")

    ax2.set_xlabel("Kumulative Inner-Iteration (Gesamt-FPI-Schritte)", fontsize=12)
    ax2.set_ylabel("max ||V_new| - |V_old|| [p.u.]\n(alle Busse: PQ + PV)", fontsize=12)
    ax2.set_title(
        "Gesamtnetz-Konvergenz: Spannungsänderung aller Busse\n"
        "(vertikale Linien = Q-Update / Outer-Iteration)",
        fontsize=12,
    )
    ax2.legend(fontsize=8, loc="upper right", ncol=1)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(bottom=1e-12, top=1e1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")

    plt.show()


def plot_convergence_single(data: list[dict], save_path: str = None):
    """
    Einzelner Plot: PV |V|-Fehler vs. Outer-Iteration.
    Farbcodiert nach Konvergenz-Status.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    conv_data = [d for d in data if d["converged"]]
    div_data = [d for d in data if not d["converged"]]

    cmap_conv = plt.cm.Greens
    cmap_div = plt.cm.Reds

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

    for i, rec in enumerate(div_data):
        errors = rec["pv_v_errors"]
        iters = list(range(1, len(errors) + 1))
        color = cmap_div(0.4 + 0.5 * i / max(len(div_data), 1))

        ax.loglog(
            iters, errors,
            color=color,
            marker="x",
            markersize=5,
            linestyle="--",
            linewidth=1.5,
            label=f"✗ {rec['name']} (η={rec['eta']:.2f})",
            alpha=0.8,
        )

    ax.axhline(y=1e-6, color="blue", linestyle=":", linewidth=2.0,
               label="tol_pv = 1e-6 (Konvergenzkriterium)")
    ax.axhline(y=1e-4, color="darkorange", linestyle="-.", linewidth=1.5,
               label="1e-4 (PASS-Schwelle)")

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


def plot_network_error_only(data: list[dict], save_path: str = None):
    """
    Dedizierter Plot NUR für die Gesamtnetz-Konvergenz (alle Busse).

    Zeigt max(||V_new| - |V_old||) pro Inner-Iteration für JEDES Netz.
    Vertikale Markierungen zeigen Q-Updates (Outer-Iterationsgrenzen).
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(data), 1)) for i in range(len(data))]

    for i, rec in enumerate(data):
        v_changes = rec["inner_v_change_all"]
        outer_starts = rec["outer_start_indices"]

        if not v_changes:
            continue

        x = list(range(1, len(v_changes) + 1))

        marker = "o" if rec["converged"] else "x"
        linestyle = "-" if rec["converged"] else "--"
        alpha = 0.85 if rec["converged"] else 0.6

        label = (f"{rec['name']} "
                 f"(n={rec['n_bus']}, PV={rec['n_pv']}, η={rec['eta']:.2f})")

        # ax.semilogy(
        #     x, v_changes,
        #     color=colors[i],
        #     marker=marker,
        #     markersize=2,
        #     linestyle=linestyle,
        #     linewidth=1.3,
        #     label=label,
        #     alpha=alpha,
        # )

        # Markiere Outer-Grenzen
        for j, idx_start in enumerate(outer_starts[1:], start=1):
            if idx_start < len(v_changes):
                ax.axvline(
                    x=idx_start + 1,
                    color=colors[i],
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.4,
                )

    # Referenzlinien
    ax.axhline(y=1e-8, color="green", linestyle=":", linewidth=2.0,
               label="tol_inner = 1e-8")
    ax.axhline(y=1e-6, color="blue", linestyle="-.", linewidth=1.5,
               label="1e-6")
    ax.axhline(y=1e-4, color="darkorange", linestyle="-.", linewidth=1.0,
               label="1e-4")

    ax.set_xlabel("Kumulative Inner-Iteration (Gesamt-FPI-Schritte über alle Outer)", fontsize=12)
    ax.set_ylabel("max ||V_new| - |V_old|| über alle Busse (PQ + PV) [p.u.]", fontsize=12)
    ax.set_title(
        "Gesamtnetz-Konvergenz: Spannungsänderung pro FPI-Schritt\n"
        "(Spitzen = Q-Update bricht lokale Konvergenz → erneute FPI)",
        fontsize=13,
    )

    ax.legend(fontsize=8, loc="upper right", ncol=1, framealpha=0.9)
    ax.grid(True)
    ax.grid(True)
    ax.set_ylim(bottom=1e-12, top=1e0)
    ax.set_xlim(left=0.5)
    ax.loglog()
    # Annotation: Erklärung der Spitzen
    ax.annotate(
        "Q-Update\n(Outer-Iter)",
        xy=(0.75, 0.85),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        ha="center",
        style="italic",
    )

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
                        default="full")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--save", type=str, default=None,
                        help="Pfad zum Speichern des Plots (z.B. convergence.png)")
    parser.add_argument("--plot", choices=["both", "pv_only", "net_only"],
                        default="both",
                        help="both: Doppelplot, pv_only: nur PV-Fehler, "
                             "net_only: nur Gesamtnetz-Fehler")
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

    if args.plot == "both":
        plot_convergence(data, save_path=save_path)
    elif args.plot == "pv_only":
        plot_convergence_single(data, save_path=save_path)
    elif args.plot == "net_only":
        plot_network_error_only(data, save_path=save_path)


if __name__ == "__main__":
    main()
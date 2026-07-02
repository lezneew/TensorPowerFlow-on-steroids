# tensor_power_flow/scripts/validate_tpf_pq_baseline.py
"""
Baseline-Performance: TPF (Dense) vs. NR für PQ-only Netze
=============================================================
Repliziert die Benchmarks aus Salazar et al. (2024), Fig. 5(a).
Keine PV-Knoten — reine Skalierungstests.

Erzeugt Plot: Rechenzeit vs. Netzgröße (log-log), TPF vs. NR.
"""

import numpy as np
import sys, os, time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_dense import TPFDenseSolver
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.network_generator_salazar import (
    SALAZAR_PQ_NETWORKS,
    get_salazar_pq_size_sweep,
    get_salazar_pq_variations,
    get_salazar_pq_networks,
)


def benchmark_single_network(net, name, n_repeats=5):
    """
    Benchmarkt TPF und NR für ein einzelnes PQ-Netz.
    Mehrfach-Ausführung für stabile Zeitmessung.
    """
    record = {
        "name": name,
        "n_bus": 0,
        "nr_converged": False,
        "nr_iter": -1,
        "nr_time_ms": 0.0,
        "tpf_converged": False,
        "tpf_iter": -1,
        "tpf_time_ms": 0.0,
        "max_v_error": np.inf,
        "speedup": 0.0,
        "eta": np.inf,
        "error": None,
    }

    # ── NR-Referenz (Aufwärmlauf + Messung) ──
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
    try:
        # Aufwärmlauf
        nr_result = nr_solver.solve_from_net(net)
        if not nr_result.converged:
            record["error"] = "NR divergiert"
            return record

        # Zeitmessung (Median über n_repeats)
        nr_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            pp.runpp(net, algorithm="nr", tolerance_mva=1e-8, max_iteration=100)
            nr_times.append((time.perf_counter() - t0) * 1000)

        record["nr_converged"] = True
        record["nr_iter"] = nr_result.iterations
        record["nr_time_ms"] = float(np.median(nr_times))

    except Exception as e:
        record["error"] = f"NR: {e}"
        return record

    # ── TPF ──
    try:
        network = build_network_from_pandapower(net, include_pv=False)
        record["n_bus"] = network.n_bus_phases

        # η berechnen
        Z_B = np.linalg.inv(network.Y_dd)
        scaling = np.conj(network.s_nom)
        M = Z_B * scaling.reshape(1, -1)
        record["eta"] = float(np.max(np.sum(np.abs(M), axis=0)))

        tpf_solver = TPFDenseSolver(tol=1e-8, max_iter=200)

        # Aufwärmlauf
        tpf_result = tpf_solver.solve(network)

        # Zeitmessung
        tpf_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            tpf_result = tpf_solver.solve(network)
            tpf_times.append((time.perf_counter() - t0) * 1000)

        record["tpf_converged"] = tpf_result.converged
        record["tpf_iter"] = tpf_result.iterations
        record["tpf_time_ms"] = float(np.median(tpf_times))

        # Vergleich
        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        pq_idx = np.where(bus_types == 1)[0]

        v_tpf = tpf_result.voltages.flatten()
        v_nr = nr_result.voltages[pq_idx]
        record["max_v_error"] = float(np.max(np.abs(np.abs(v_tpf) - np.abs(v_nr))))

        if record["tpf_time_ms"] > 0:
            record["speedup"] = record["nr_time_ms"] / record["tpf_time_ms"]

    except Exception as e:
        record["error"] = f"TPF: {e}"

    return record


def print_results_table(records: list):
    """Ergebnistabelle."""
    print(f"\n{'═'*120}")
    print(f"  ERGEBNISTABELLE: TPF (Dense, PQ-only) vs. NR — Baseline-Performance")
    print(f"{'═'*120}")

    hdr = (f"  {'Netz':<22} {'n_bus':<7} {'η':<9} "
           f"{'NR It':<7} {'NR ms':<9} "
           f"{'TPF It':<8} {'TPF ms':<9} "
           f"{'Speedup':<9} {'max ΔV':<12} {'Status'}")
    print(hdr)
    print(f"  {'─'*118}")

    for r in records:
        if r.get("error"):
            print(f"  {r['name']:<22} — {r['error']}")
            continue

        eta_str = f"{r['eta']:.4f}" if r['eta'] < 100 else f"{r['eta']:.1f}"
        max_v_str = f"{r['max_v_error']:.2e}" if r['max_v_error'] < 100 else "—"
        spd_str = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "—"

        status = "✓" if r['tpf_converged'] and r['max_v_error'] < 1e-5 else "✗"

        print(f"  {r['name']:<22} {r['n_bus']:<7} {eta_str:<9} "
              f"{r['nr_iter']:<7} {r['nr_time_ms']:<9.3f} "
              f"{r['tpf_iter']:<8} {r['tpf_time_ms']:<9.3f} "
              f"{spd_str:<9} {max_v_str:<12} {status}")

    # Statistik
    valid = [r for r in records if r.get("tpf_converged") and not r.get("error")]
    if valid:
        speedups = [r["speedup"] for r in valid if r["speedup"] > 0]
        print(f"\n  {'─'*60}")
        print(f"  Speedup TPF/NR:  min={min(speedups):.2f}x  max={max(speedups):.2f}x  "
              f"median={np.median(speedups):.2f}x")
        print(f"  Alle konvergiert: {len(valid)}/{len(records)}")


def plot_results(records: list, save_path: str = None):
    """Plot: Rechenzeit vs. Netzgröße (log-log)."""
    import matplotlib.pyplot as plt

    valid = [r for r in records
             if r.get("tpf_converged") and r.get("nr_converged")
             and not r.get("error")
             and r.get("n_bus", 0) > 0]

    if not valid:
        print("  ⚠ Keine Daten zum Plotten.")
        return

    valid = sorted(valid, key=lambda r: r["n_bus"])

    n_bus = np.array([r["n_bus"] for r in valid])
    nr_t = np.array([r["nr_time_ms"] for r in valid])
    tpf_t = np.array([r["tpf_time_ms"] for r in valid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Plot 1: Rechenzeit vs. Netzgröße ──
    ax1.loglog(n_bus, nr_t, "rs-", markersize=7, linewidth=1.5,
               label="Newton-Raphson (pandapower)", alpha=0.8)
    ax1.loglog(n_bus, tpf_t, "bo-", markersize=7, linewidth=1.5,
               label="TPF Dense (PQ-only)", alpha=0.8)

    ax1.set_xlabel("Netzgröße (Anzahl PQ-Knoten)", fontsize=12)
    ax1.set_ylabel("Rechenzeit [ms]", fontsize=12)
    ax1.set_title("TPF vs. NR: Rechenzeit vs. Netzgröße\n(PQ-only, Salazar-Netze)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)

    # ── Plot 2: Speedup vs. Netzgröße ──
    speedups = nr_t / tpf_t
    ax2.semilogx(n_bus, speedups, "g^-", markersize=8, linewidth=1.5, alpha=0.8)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1.0, label="NR = TPF")

    ax2.set_xlabel("Netzgröße (Anzahl PQ-Knoten)", fontsize=12)
    ax2.set_ylabel("Speedup (NR_time / TPF_time)", fontsize=12)
    ax2.set_title("Speedup TPF über NR vs. Netzgröße", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    # Annotationen
    for i, r in enumerate(valid):
        if r["n_bus"] in [9, 100, 500, 1000, 2000, 5000]:
            ax2.annotate(f"{speedups[i]:.1f}x",
                         (n_bus[i], speedups[i]),
                         textcoords="offset points", xytext=(5, 5),
                         fontsize=8, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline: TPF vs NR (PQ-only)")
    parser.add_argument("--suite", choices=["size", "variations", "all"],
                        default="size")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Wiederholungen pro Netz für stabile Zeitmessung")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  BASELINE: TPF Dense vs. NR — PQ-only Netze (Salazar Paper)     ║")
    print("║  Keine PV-Knoten — reine Skalierungstests                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    if args.suite == "size":
        networks = get_salazar_pq_size_sweep()
    elif args.suite == "variations":
        networks = get_salazar_pq_variations()
    else:
        networks = get_salazar_pq_networks()

    print(f"\n  Suite: '{args.suite}' — {len(networks)} Netze, {args.repeats} Wiederholungen\n")

    records = []
    for name, info in networks.items():
        print(f"  Benchmarke: {name}...", end=" ")
        try:
            net = info["constructor"]()
            record = benchmark_single_network(net, name, n_repeats=args.repeats)
            records.append(record)

            if record.get("error"):
                print(f"FEHLER: {record['error']}")
            else:
                print(f"✓ n={record['n_bus']}, "
                      f"NR={record['nr_time_ms']:.2f}ms, "
                      f"TPF={record['tpf_time_ms']:.2f}ms, "
                      f"Speedup={record['speedup']:.1f}x")
        except Exception as e:
            print(f"FEHLER: {e}")
            records.append({"name": name, "error": str(e)})

    print_results_table(records)

    if not args.no_plot:
        save_path = args.save or "baseline_tpf_vs_nr_pq_only.png"
        plot_results(records, save_path=save_path)


if __name__ == "__main__":
    main()
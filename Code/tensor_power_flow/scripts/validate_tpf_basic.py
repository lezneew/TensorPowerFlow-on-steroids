"""
Validierungsskript: TPF Dense vs. pandapower Newton-Raphson
===========================================================
Ziel: Sicherstellen, dass der TPF korrekte Ergebnisse liefert.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import sys
import os

# Projektpfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tpf.builders.from_pandapower import (
    build_network_from_pandapower,
    get_pq_indices_from_net,
)
from tpf.solvers.tpf_dense import TPFDenseSolver
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.validation.comparator import compare_results


def run_validation(net: pp.pandapowerNet, net_name: str) -> None:
    """Führt die Validierung für ein gegebenes Netz durch."""
    print(f"\n{'='*60}")
    print(f"  Validierung: {net_name}")
    print(f"{'='*60}")

    # --- Schritt 1: Referenzlösung (pandapower NR) ---
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
    nr_result = nr_solver.solve_from_net(net)

    print(f"\n[NR] Konvergiert: {nr_result.converged}")
    print(f"[NR] Iterationen: {nr_result.iterations}")
    print(f"[NR] Zeit: {nr_result.elapsed_time_s*1000:.2f} ms")

    # --- Schritt 2: TPF-Netz aufbauen ---
    try:
        network = build_network_from_pandapower(net)
    except Exception as e:
        print(f"[FEHLER] Builder: {e}")
        return

    print(f"\n[TPF] Netzgröße: {network.n_buses} PQ-Knoten")
    print(f"[TPF] Y_dd Shape: {network.Y_dd.shape}")
    print(f"[TPF] s_nom (erste 5): {network.s_nom[:5]}")

    # --- Schritt 3: TPF lösen ---
    tpf_solver = TPFDenseSolver(tol=1e-8, max_iter=200)
    tpf_result = tpf_solver.solve(network)

    print(f"\n[TPF] Konvergiert: {tpf_result.converged}")
    print(f"[TPF] Iterationen: {tpf_result.iterations}")
    print(f"[TPF] Max Mismatch: {tpf_result.max_mismatch:.2e}")
    print(f"[TPF] Zeit: {tpf_result.elapsed_time_s*1000:.2f} ms")

    # --- Schritt 4: Ergebnisse vergleichen ---
    pq_idx = get_pq_indices_from_net(net)
    comparison = compare_results(tpf_result, nr_result, pq_idx)

    print(f"\n--- Vergleich TPF vs. NR ---")
    print(f"  Max |V| Fehler:    {comparison.max_voltage_error_pu:.2e} p.u.")
    print(f"  Mean |V| Fehler:   {comparison.mean_voltage_error_pu:.2e} p.u.")
    print(f"  Max Winkelfehler:  {comparison.max_angle_error_deg:.4f}°")
    print(f"  Iterations-Ratio:  {comparison.iteration_ratio:.2f}")
    print(f"  Beide konvergiert: {comparison.both_converged}")

    # --- Schritt 5: Pass/Fail ---
    PASS = comparison.max_voltage_error_pu < 1e-5 and comparison.both_converged
    status = "PASS" if PASS else "FAIL"
    print(f"\n  Ergebnis: {status}")

    # --- Schritt 6: Detail-Ausgabe bei Fehler ---
    if not PASS:
        print("\n  [DEBUG] Erste 5 Spannungen:")
        v_tpf = np.abs(tpf_result.voltages.flatten()[:5])
        v_nr = np.abs(nr_result.voltages[pq_idx[:5]])
        print(f"    TPF: {v_tpf}")
        print(f"    NR:  {v_nr}")


def debug_network_types(net, name):
    """Zeigt die PPC-Bustypen für Debugging."""
    pp.runpp(net)
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    type_map = {1: "PQ", 2: "PV", 3: "Slack"}

    print(f"\n[DEBUG] {name} — Bus-Typen im PPC:")
    print(f"  {'Bus':<5} {'Type':<8} {'Pd (MW)':<10} {'Qd (MVAr)':<10} {'Vm (p.u.)'}")
    for i in range(len(bus_types)):
        t = type_map.get(bus_types[i], "?")
        pd = ppc["bus"][i, 2]
        qd = ppc["bus"][i, 3]
        vm = ppc["bus"][i, 7]
        print(f"  {i:<5} {t:<8} {pd:<10.2f} {qd:<10.2f} {vm:.4f}")

    n_slack = np.sum(bus_types == 3)
    n_pv = np.sum(bus_types == 2)
    n_pq = np.sum(bus_types == 1)
    print(f"  → Slack: {n_slack}, PV: {n_pv}, PQ: {n_pq}")


def main():
    print("=" * 60)
    print("  TPF VALIDIERUNG – Phase 2: Standard-TPF korrekt?")
    print("=" * 60)

    # --- Test 1: Kleines Netz (case4gs — 4 Busse, 1 Slack + 3 PQ) ---
    # Hinweis: case4gs hat NUR PQ-Knoten (kein PV) → ideal für Test
    print("\n\n>>> Test 1: case4gs (4 Busse)")
    try:
        net = pn.case4gs()
        debug_network_types(net, 'case4gs')
        run_validation(net, "case4gs")
    except Exception as e:
        print(f"  FEHLER: {e}")

    # --- Test 2: IEEE 33 Bus (radiales Verteilnetz) ---
    print("\n\n>>> Test 2: case33bw (33 Busse, radial)")
    try:
        net = pn.case33bw()
        debug_network_types(net, 'case33bw')
        run_validation(net, "case33bw (IEEE 33)")
    except Exception as e:
        print(f"  FEHLER: {e}")

    # --- Test 3: case_ieee30 (30 Busse — hat PV-Knoten!) ---
    # Hier testen wir, ob der Builder PV-Knoten korrekt ausschließt
    print("\n\n>>> Test 3: case_ieee30 (30 Busse, mit PV-Knoten)")
    try:
        net = pn.case_ieee30()
        debug_network_types(net, 'case_ieee30')
        # PV-Knoten entfernen für reinen PQ-Test
        # (Generatoren außer Slack in statische Last umwandeln)
        run_validation(net, "case_ieee30 (nur PQ-Teil)")
    except Exception as e:
        print(f"  FEHLER: {e}")

if __name__ == "__main__":
    main()
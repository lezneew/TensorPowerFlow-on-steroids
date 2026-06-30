# tensor_power_flow/scripts/validate_pv_method_a.py
"""
Validierung: TPF Methode A (Äußere Q-Schleife) vs. pandapower NR
===================================================================
Erweiterte Ausgabe: Zeiten, Kontraktionsfaktor η, Ergebnistabelle
"""
import numpy as np
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.radial_network import (
    TEST_NETWORKS,
    get_quick_test_networks,
    get_radial_only_networks,
)


# ══════════════════════════════════════════════════════════════════════
#  η-Berechnung
# ══════════════════════════════════════════════════════════════════════

def compute_eta(Y_dd, s_nom, v_min_pu):
    """
    Kontraktionsfaktor η = ||Z_B · diag(s*)||₁ / v_min²

    η < 1 → FPI konvergiert garantiert (Banach)
    """
    n = Y_dd.shape[0]
    if n == 0 or v_min_pu < 1e-6:
        return np.inf
    try:
        Z_B = np.linalg.inv(Y_dd)
    except np.linalg.LinAlgError:
        return np.inf

    scaling = np.conj(s_nom)
    M = Z_B * scaling.reshape(1, -1)
    matrix_1_norm = np.max(np.sum(np.abs(M), axis=0))
    eta = matrix_1_norm / (v_min_pu ** 2)
    return eta


# ══════════════════════════════════════════════════════════════════════
#  Einzelnetz-Validierung
# ══════════════════════════════════════════════════════════════════════

def validate_single_network(net, name, tol_pass=1e-4):
    """
    Validiert Methode A gegen NR für ein einzelnes Netz.

    Returns
    -------
    dict mit allen relevanten Metriken (oder None bei Fehler)
    """
    record = {
        "name": name,
        "n_bus": 0,
        "n_pv": 0,
        "n_pq": 0,
        "eta": np.inf,
        "nr_converged": False,
        "nr_iter": -1,
        "nr_time_ms": 0.0,
        "tpf_converged": False,
        "tpf_outer_iter": 0,
        "tpf_inner_iter_total": 0,
        "tpf_time_ms": 0.0,
        "max_v_error": np.inf,
        "mean_v_error": np.inf,
        "max_angle_error_deg": np.inf,
        "max_pv_v_error": np.inf,
        "speedup": 0.0,
        "passed": False,
        "error": None,
    }

    print(f"\n{'═'*70}")
    print(f"  {name}")
    print(f"{'═'*70}")

    # ── 1. NR-Referenz ──
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
    try:
        nr_result = nr_solver.solve_from_net(net)
    except Exception as e:
        record["error"] = f"NR: {e}"
        print(f"  ✗ NR-Fehler: {e}")
        return record

    if not nr_result.converged:
        record["error"] = "NR divergiert"
        print(f"  ✗ NR divergiert — übersprungen")
        return record

    record["nr_converged"] = True
    record["nr_iter"] = nr_result.iterations
    record["nr_time_ms"] = nr_result.elapsed_time_s * 1000

    print(f"  [NR] Konvergiert: {nr_result.iterations} Iter., "
          f"{record['nr_time_ms']:.2f} ms")

    # ── 2. Netz für TPF aufbauen (MIT PV-Knoten) ──
    try:
        network = build_network_from_pandapower(net, include_pv=True)
    except Exception as e:
        record["error"] = f"Builder: {e}"
        print(f"  ✗ Builder-Fehler: {e}")
        return record

    n_pv = network.n_pv
    n_pq = len(network.pq_indices)
    record["n_bus"] = network.n_bus_phases
    record["n_pv"] = n_pv
    record["n_pq"] = n_pq

    print(f"  [Netz] d-Block: {network.n_bus_phases} Knoten "
          f"(PQ={n_pq}, PV={n_pv})")

    if n_pv == 0:
        record["error"] = "Keine PV-Knoten"
        print(f"  ⚠ Keine PV-Knoten — übersprungen")
        record["passed"] = True
        return record

    # ── 3. Kontraktionsfaktor η ──
    # v_min aus NR-Ergebnis (für den d-Block)
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    pv_idx_ppc = np.where(bus_types == 2)[0]
    pq_idx_ppc = np.where(bus_types == 1)[0]
    d_idx = np.sort(np.concatenate([pq_idx_ppc, pv_idx_ppc]))
    v_min_d = float(np.min(np.abs(nr_result.voltages[d_idx])))

    eta = compute_eta(network.Y_dd, network.s_nom, v_min_d)
    record["eta"] = eta
    eta_status = "✓" if eta < 1.0 else "⚠" if eta < 2.0 else "✗"
    print(f"  [η] Kontraktionsfaktor: {eta:.4f} {eta_status}")

    # ── 4. Methode A lösen ──
    solver_a = TPFDensePVMethodA(
        tol=1e-8,
        max_iter_inner=50,
        max_iter_outer=50,
        tol_pv=1e-6,
        omega=0.5,
        enforce_q_lims=False,
    )

    try:
        result_a = solver_a.solve(network)
    except Exception as e:
        record["error"] = f"Methode A: {e}"
        print(f"  ✗ Methode A Fehler: {e}")
        return record

    record["tpf_converged"] = result_a.converged
    record["tpf_time_ms"] = result_a.elapsed_time_s * 1000
    record["tpf_inner_iter_total"] = result_a.iterations

    if solver_a.pv_info:
        info = solver_a.pv_info
        record["tpf_outer_iter"] = info.outer_iterations
        record["max_pv_v_error"] = info.pv_v_error_final

        print(f"  [TPF-A] Konvergiert: {result_a.converged}")
        print(f"  [TPF-A] Outer: {info.outer_iterations}, "
              f"Inner gesamt: {result_a.iterations}")
        print(f"  [TPF-A] Inner/Outer: {info.inner_iterations_per_outer}")
        print(f"  [TPF-A] PV |V|-Fehler: {info.pv_v_error_final:.2e}")
        print(f"  [TPF-A] Zeit: {record['tpf_time_ms']:.2f} ms")

    # ── 5. Speedup ──
    if record["tpf_time_ms"] > 0:
        record["speedup"] = record["nr_time_ms"] / record["tpf_time_ms"]
    print(f"  [Speedup] NR/TPF-A: {record['speedup']:.2f}x")

    # ── 6. Vergleich mit NR ──
    v_tpf = result_a.voltages.flatten()
    v_nr = nr_result.voltages[d_idx]

    # Spannungsbetragsfehler
    mag_err = np.abs(np.abs(v_tpf) - np.abs(v_nr))
    max_mag_err = float(np.max(mag_err))
    mean_mag_err = float(np.mean(mag_err))

    # Winkelfehler
    angle_err = np.abs(np.angle(v_tpf, deg=True) - np.angle(v_nr, deg=True))
    max_angle_err = float(np.max(angle_err))

    record["max_v_error"] = max_mag_err
    record["mean_v_error"] = mean_mag_err
    record["max_angle_error_deg"] = max_angle_err

    print(f"\n  ── Vergleich vs. NR ──")
    print(f"  Max |V| Fehler:    {max_mag_err:.2e} p.u.")
    print(f"  Mean |V| Fehler:   {mean_mag_err:.2e} p.u.")
    print(f"  Max Winkel-Fehler: {max_angle_err:.4f}°")

    # ── 7. PV-Knoten Details ──
    if solver_a.pv_info and network.pv_indices is not None:
        pv_local = network.pv_indices
        print(f"\n  ── PV-Knoten ──")
        print(f"  {'Idx':<5} {'|V| TPF':<10} {'|V| NR':<10} "
              f"{'|V| Soll':<10} {'ΔV':<10}")
        print(f"  {'─'*45}")

        for i, pv_i in enumerate(pv_local):
            v_tpf_i = np.abs(v_tpf[pv_i])
            v_nr_i = np.abs(v_nr[pv_i])
            v_spec_i = network.pv_v_setpoint[i]
            dv = abs(v_tpf_i - v_nr_i)
            print(f"  {pv_i:<5} {v_tpf_i:<10.6f} {v_nr_i:<10.6f} "
                  f"{v_spec_i:<10.4f} {dv:<10.2e}")

    # ── 8. PASS/FAIL ──
    passed = max_mag_err < tol_pass and result_a.converged
    record["passed"] = passed
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n  Ergebnis: {status}")

    return record


# ══════════════════════════════════════════════════════════════════════
#  Ergebnistabelle
# ══════════════════════════════════════════════════════════════════════

def print_summary_table(records: list[dict]):
    """Druckt die finale Ergebnistabelle über alle Netze."""

    # Filtere valide Records
    valid = [r for r in records if r.get("nr_converged") or r.get("error")]

    print(f"\n\n{'═'*140}")
    print(f"  ERGEBNISTABELLE: Methode A — Alle Netze")
    print(f"{'═'*140}")

    # Header
    hdr = (f"  {'Netz':<22} {'n_d':<5} {'PV':<4} {'PQ':<5} "
           f"{'η':<8} {'η ok':<5} "
           f"{'NR It':<6} {'NR ms':<8} "
           f"{'A Out':<6} {'A Inn':<6} {'A ms':<8} "
           f"{'Speed':<7} "
           f"{'max ΔV':<10} {'mean ΔV':<10} {'max Δθ°':<8} "
           f"{'PV ΔV':<10} {'Status'}")
    print(hdr)
    print(f"  {'─'*138}")

    for r in records:
        namelong = r["name"]
        name =namelong[:]

        if r.get("error") and not r.get("nr_converged"):
            print(f"  {name:<10} — {r.get('error', '?')}")
            continue

        if r["n_pv"] == 0 and r.get("passed"):
            print(f"  {name:<22} (keine PV-Knoten — übersprungen)")
            continue

        # η formatieren
        eta = r["eta"]
        eta_str = f"{eta:.4f}" if eta < 100 else f"{eta:.1f}"
        eta_ok = "✓" if eta < 1.0 else "⚠" if eta < 2.0 else "✗"

        # Fehler formatieren
        max_v = r["max_v_error"]
        mean_v = r["mean_v_error"]
        max_a = r["max_angle_error_deg"]
        pv_v = r["max_pv_v_error"]

        max_v_str = f"{max_v:.2e}" if max_v < 100 else "—"
        mean_v_str = f"{mean_v:.2e}" if mean_v < 100 else "—"
        max_a_str = f"{max_a:.4f}" if max_a < 100 else "—"
        pv_v_str = f"{pv_v:.2e}" if pv_v < 100 else "—"

        # Speedup
        spd = r["speedup"]
        spd_str = f"{spd:.2f}x" if spd > 0 else "—"

        # Status
        if r["passed"]:
            status = "✓ PASS"
        elif not r["tpf_converged"]:
            status = "✗ DIV"
        else:
            status = "✗ FAIL"

        print(f"  {name:<22} {r['n_bus']:<5} {r['n_pv']:<4} {r['n_pq']:<5} "
              f"{eta_str:<8} {eta_ok:<5} "
              f"{r['nr_iter']:<6} {r['nr_time_ms']:<8.2f} "
              f"{r['tpf_outer_iter']:<6} {r['tpf_inner_iter_total']:<6} "
              f"{r['tpf_time_ms']:<8.2f} "
              f"{spd_str:<7} "
              f"{max_v_str:<10} {mean_v_str:<10} {max_a_str:<8} "
              f"{pv_v_str:<10} {status}")

    # ── Statistiken ──
    passed_records = [r for r in records if r["passed"]]
    failed_records = [r for r in records if not r["passed"] and r.get("nr_converged")]
    converged_tpf = [r for r in records if r.get("tpf_converged")]

    print(f"\n  {'─'*100}")
    print(f"  STATISTIKEN:")
    print(f"  {'─'*100}")
    print(f"  Getestet:              {len(records)}")
    print(f"  NR konvergiert:        {sum(1 for r in records if r.get('nr_converged'))}")
    print(f"  TPF-A konvergiert:     {len(converged_tpf)}")
    print(f"  PASS:                  {len(passed_records)}")
    print(f"  FAIL:                  {len(failed_records)}")

    if converged_tpf:
        etas = [r["eta"] for r in converged_tpf if r["eta"] < np.inf]
        v_errors = [r["max_v_error"] for r in converged_tpf if r["max_v_error"] < np.inf]
        speedups = [r["speedup"] for r in converged_tpf if r["speedup"] > 0]
        outer_iters = [r["tpf_outer_iter"] for r in converged_tpf if r["tpf_outer_iter"] > 0]

        if etas:
            print(f"\n  η:          min={min(etas):.4f}  max={max(etas):.4f}  "
                  f"median={np.median(etas):.4f}")
        if v_errors:
            print(f"  max|ΔV|:    min={min(v_errors):.2e}  max={max(v_errors):.2e}  "
                  f"median={np.median(v_errors):.2e}")
        if speedups:
            print(f"  Speedup:    min={min(speedups):.2f}x  max={max(speedups):.2f}x  "
                  f"median={np.median(speedups):.2f}x")
        if outer_iters:
            print(f"  Outer Iter: min={min(outer_iters)}  max={max(outer_iters)}  "
                  f"median={np.median(outer_iters):.0f}")

    # ── η vs. Konvergenz ──
    eta_conv = [r["eta"] for r in converged_tpf if r["eta"] < np.inf]
    eta_div = [r["eta"] for r in records
               if not r.get("tpf_converged") and r.get("nr_converged")
               and r["eta"] < np.inf]

    if eta_conv or eta_div:
        print(f"\n  η-Analyse:")
        if eta_conv:
            print(f"    Konvergiert:   η ∈ [{min(eta_conv):.4f}, {max(eta_conv):.4f}]")
        if eta_div:
            print(f"    Divergiert:    η ∈ [{min(eta_div):.4f}, {max(eta_div):.4f}]")

    print(f"\n{'═'*140}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validierung Methode A")
    parser.add_argument(
        "--suite", choices=["quick", "radial", "full"], default="radial",
        help="Testumfang: quick (4 Netze), radial (ohne IEEE vermascht), full (alles)"
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  VALIDIERUNG: TPF Methode A (Äußere Q-Schleife) vs. NR          ║")
    print("║  Metriken: η, Iterationen, Zeiten, Speedup, |V|-Fehler          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    if args.suite == "quick":
        networks = get_quick_test_networks()
    elif args.suite == "radial":
        networks = get_radial_only_networks()
    else:
        networks = TEST_NETWORKS

    print(f"\n  Suite: '{args.suite}' — {len(networks)} Netze\n")

    records = []
    for name, info in networks.items():
        net = info["constructor"]()
        record = validate_single_network(net, f"{name}")
        records.append(record)

    # ── Finale Ergebnistabelle ──
    print_summary_table(records)

    # ── Gesamtergebnis ──
    n_pass = sum(1 for r in records if r["passed"])
    n_total = len(records)
    print(f"\n  Gesamtergebnis: {n_pass}/{n_total} bestanden")

    if n_pass == n_total:
        print(f"\n  ✓ METHODE A FUNKTIONIERT KORREKT FÜR ALLE TESTNETZE!")
    else:
        n_fail = sum(1 for r in records if not r["passed"] and r.get("nr_converged"))
        print(f"\n  ⚠ {n_fail} Tests fehlgeschlagen — Debugging/ω-Anpassung erforderlich")


if __name__ == "__main__":
    main()
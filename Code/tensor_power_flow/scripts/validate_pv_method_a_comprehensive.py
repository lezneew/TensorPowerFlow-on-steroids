# tensor_power_flow/scripts/validate_pv_method_a_comprehensive.py
"""
Umfassende Validierung: TPF Methode A mit Spektralradius + Konvergenz-Plot
==========================================================================

Validiert Methode A gegen NR für alle TEST_NETWORKS und plottet:
- Links:  PV |V|-Fehler vs. Outer-Iteration
- Rechts: Gesamtnetz max(|ΔV|) vs. kumulative Inner-Iteration (ALLE Busse)

NEU: Berechnet ρ(I + ω·H·D⁻¹) numerisch für jedes Netz.

Aufruf:
    python scripts/validate_pv_method_a_comprehensive.py
    python scripts/validate_pv_method_a_comprehensive.py --suite full --omega 1.0
    python scripts/validate_pv_method_a_comprehensive.py --no-plot
"""

import numpy as np
import sys
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.radial_network import (
    TEST_NETWORKS,
    get_quick_test_networks,
    get_radial_only_networks,
    get_full_test_suite,
    get_comprehensive_networks,
)
from tpf.generators.network_generator_salazar import (
    SALAZAR_TEST_NETWORKS,
    get_salazar_pv_networks,
    get_salazar_scaling_networks,
    SALAZAR_SCALING_NETWORKS,
    get_salazar_low_vm_networks,
    SALAZAR_LOW_VM_NETWORKS,
    get_salazar_low_rx05_networks,
    SALAZAR_LOW_RX05_NETWORKS,
    get_salazar_low_rx10_networks,
    SALAZAR_LOW_RX10_NETWORKS,
    create_salazar_network,
)
from tpf.generators.ieee_pegase_networks import (
    IEE_PEGASE_NETWORKS,
    get_ieee_networks,
    get_pegase_networks,
    get_rte_networks,
    get_large_networks,
    get_all_standard_networks,
)

try:
    from convergence_analysis import (
        compute_spectral_radius_diagonal,
        compute_spectral_radius_corrected,
        compute_empirical_contraction,
        compute_all_convergence_metrics,
    )
except ImportError:
    compute_spectral_radius_diagonal = None
    compute_spectral_radius_corrected = None
    compute_empirical_contraction = None
    compute_all_convergence_metrics = None

# ══════════════════════════════════════════════════════════════════════
#  η-Berechnung
# ══════════════════════════════════════════════════════════════════════

def compute_eta(Y_dd, s_nom, v_min_pu):
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
    return matrix_1_norm / (v_min_pu ** 2)


# ══════════════════════════════════════════════════════════════════════
#  Spektralradius ρ(J_G) numerisch berechnen
# ══════════════════════════════════════════════════════════════════════

def compute_spectral_radius(network, omega, delta_q=1e-5):
    """
    Berechnet den Spektralradius der Iterationsmatrix der äußeren Q-Schleife
    via Finite-Differenzen für die Sensitivitätsmatrix H.

    ρ(I - ω·A_pv_inv @ H)

    wobei:
        A_pv_inv = inv(2·X_pp + ε·I)    (FULL matrix inverse, gekoppelt)
        H_kj = ∂|V_k|²/∂Q_j            (Sensitivitätsmatrix, numerisch)

    HINWEIS: Die lineare Analyse ergibt typischerweise ρ > 1 (≈2), weil
    H ≈ -2·X_pp im linearen Regime. Dennoch konvergiert der Solver oft,
    da die innere FPI nichtlinear ist und zusätzliche Dämpfung bietet.

    Parameters
    ----------
    network : NetworkData (mit PV-Knoten)
    omega : float (Relaxationsfaktor)
    delta_q : float (Perturbation für Finite-Differenzen)

    Returns
    -------
    rho : float (Spektralradius)
    """
    if not network.has_pv or network.n_pv == 0:
        return 0.0

    pv_idx = network.pv_indices
    n_pv = len(pv_idx)
    bphi = network.n_bus_phases

    Z_B = np.linalg.inv(network.Y_dd)
    K = -Z_B
    L = (K @ network.Y_ds @ network.v_s).reshape(-1, 1)

    X_pp = np.imag(Z_B[pv_idx, :][:, pv_idx])
    eps = 1e-10
    A_pv = 2.0 * X_pp + eps * np.eye(n_pv)
    A_pv_inv = np.linalg.inv(A_pv)

    s_base = network.s_nom.copy().reshape(-1, 1)

    def solve_inner_fpi(s_work, max_iter=200, tol=1e-10):
        V = np.ones((bphi, 1), dtype=np.complex128)
        S_conj = np.conj(s_work)
        for _ in range(max_iter):
            LAMBDA = S_conj * (1.0 / np.conj(V))
            V_new = K @ LAMBDA + L
            if np.max(np.abs(np.abs(V_new) - np.abs(V))) < tol:
                return np.abs(V_new[pv_idx, 0]) ** 2
            V = V_new
        return np.abs(V[pv_idx, 0]) ** 2

    v2_base = solve_inner_fpi(s_base)

    H = np.zeros((n_pv, n_pv))
    for j in range(n_pv):
        s_pert = s_base.copy()
        s_pert[pv_idx[j], 0] += 1j * delta_q
        v2_pert = solve_inner_fpi(s_pert)
        H[:, j] = (v2_pert - v2_base) / delta_q

    J_G = np.eye(n_pv) - omega * A_pv_inv @ H

    eigenvalues = np.linalg.eigvals(J_G)
    rho = float(np.max(np.abs(eigenvalues)))

    return rho


# ══════════════════════════════════════════════════════════════════════
#  Einzelnetz-Validierung
# ══════════════════════════════════════════════════════════════════════

def validate_network(net_constructor, name, omega=1.0, tol_pass=1e-4, verbose=True, cold_start=False, analysis="full"):
    """
    Validiert Methode A gegen NR. Gibt Record mit Konvergenzhistorie zurück.

    Parameters
    ----------
    analysis : str
        Konvergenz-Analysemodus: "full" (alle), "diagonal" (rho_diag),
        "corrected" (rho_corr), "contraction" (kappa)
    """
    record = {
        "name": name,
        "n_bus": 0, "n_pv": 0, "n_pq": 0,
        "eta": np.inf,
        "rho": np.inf,
        "rho_diag": np.inf,
        "rho_corr": np.inf,
        "contraction": np.inf,
        "nr_converged": False, "nr_iter": -1, "nr_time_ms": 0.0,
        "tpf_converged": False, "tpf_outer_iter": 0,
        "tpf_inner_iter_total": 0, "tpf_time_ms": 0.0,
        "max_v_error": np.inf, "mean_v_error": np.inf,
        "max_angle_error_deg": np.inf, "max_pv_v_error": np.inf,
        "speedup": 0.0, "passed": False, "error": None,
        "analysis_mode": analysis,
        # Konvergenz-Historie für Plots
        "pv_v_errors": [],
        "inner_v_change_all": [],
        "outer_start_indices": [],
        "inner_per_outer": [],
        "pv_ratio": 0.0,
        "rx_ratio": np.nan,
    }

    # 0. Netz erzeugen
    try:
        net = net_constructor()
    except Exception as e:
        record["error"] = f"Constructor: {str(e)[:50]}"
        if verbose:
            print(f"  X {name:<30} FEHLER (Constructor): {e}")
        return record

    # 1. NR-Referenz
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
    try:
        nr_result = nr_solver.solve_from_net(net)
    except Exception as e:
        record["error"] = f"NR: {str(e)[:50]}"
        if verbose:
            print(f"  X {name:<30} FEHLER (NR): {e}")
        return record

    if not nr_result.converged:
        record["error"] = "NR divergiert"
        if verbose:
            print(f"  X {name:<30} NR divergiert")
        return record

    record["nr_converged"] = True
    record["nr_iter"] = nr_result.iterations
    record["nr_time_ms"] = nr_result.elapsed_time_s * 1000

    # 2. Netzwerk aufbauen
    try:
        network = build_network_from_pandapower(net, include_pv=True)
    except Exception as e:
        record["error"] = f"Builder: {str(e)[:50]}"
        if verbose:
            print(f"  X {name:<30} FEHLER (Builder): {e}")
        return record

    n_pv = network.n_pv
    record["n_bus"] = network.n_bus_phases
    record["n_pv"] = n_pv
    record["pv_ratio"] = n_pv / max(record["n_bus"], 1)
    record["n_pq"] = len(network.pq_indices)

    # 2b. R/X ratio from lines
    try:
        r_vals = net.line['r_ohm_per_km'].values
        x_vals = net.line['x_ohm_per_km'].values
        r_total = np.sum(r_vals)
        x_total = np.sum(x_vals)
        record["rx_ratio"] = r_total / x_total if x_total > 1e-12 else np.nan
    except Exception:
        record["rx_ratio"] = np.nan

    if n_pv == 0:
        record["passed"] = True
        if verbose:
            print(f"  - {name:<30} keine PV (übersprungen)")
        return record

    # 3. η
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    pv_idx_ppc = np.where(bus_types == 2)[0]
    pq_idx_ppc = np.where(bus_types == 1)[0]
    d_idx = np.sort(np.concatenate([pq_idx_ppc, pv_idx_ppc]))
    v_min_d = float(np.min(np.abs(nr_result.voltages[d_idx]))) if len(d_idx) > 0 else 1.0
    record["eta"] = compute_eta(network.Y_dd, network.s_nom, v_min_d)

    # 4. Spektralradius ρ(J_G)
    try:
        rho = compute_spectral_radius(network, omega)
        record["rho"] = rho
    except Exception:
        record["rho"] = np.inf

    # 5. Methode A
    solver = TPFDensePVMethodA(
        tol=1e-8, max_iter_inner=20, max_iter_outer=50,
        tol_pv=1e-6, omega=omega, enforce_q_lims=False,
        cold_start=cold_start,
    )

    try:
        result = solver.solve(network)
    except Exception as e:
        record["error"] = f"TPF: {str(e)[:50]}"
        if verbose:
            print(f"  X {name:<30} FEHLER (TPF): {e}")
        return record

    record["tpf_converged"] = result.converged
    record["tpf_time_ms"] = result.elapsed_time_s * 1000
    record["tpf_inner_iter_total"] = result.iterations

    if solver.pv_info:
        pv_info = solver.pv_info
        record["tpf_outer_iter"] = pv_info.outer_iterations
        record["max_pv_v_error"] = pv_info.pv_v_error_final
        record["pv_v_errors"] = pv_info.pv_v_error_history or []
        record["inner_v_change_all"] = pv_info.inner_v_change_all or []
        record["outer_start_indices"] = pv_info.outer_start_indices or []
        record["inner_per_outer"] = pv_info.inner_iterations_per_outer or []

    # 5b. Erweiterte Konvergenz-Analyse (rho_diag, rho_corr, contraction)
    if analysis in ["full", "diagonal", "corrected",
                    "contraction"] and n_pv > 0 and compute_spectral_radius_diagonal is not None:
        pv_error_history = record["pv_v_errors"]
        try:
            if analysis in ["diagonal", "full"]:
                record["rho_diag"] = compute_spectral_radius_diagonal(network, omega)
            if analysis in ["corrected", "full"]:
                record["rho_corr"] = compute_spectral_radius_corrected(network, omega)
            if analysis in ["contraction", "full"]:
                record["contraction"] = compute_empirical_contraction(pv_error_history)
        except Exception:
            pass

    # 6. Vergleich
    v_tpf = result.voltages.flatten()
    v_nr = nr_result.voltages[d_idx]

    if v_tpf.shape[0] != v_nr.shape[0]:
        record["error"] = f"Dim mismatch: TPF={v_tpf.shape[0]} NR={v_nr.shape[0]}"
        if verbose:
            print(f"  X {name:<30} Dimensionsfehler")
        return record

    mag_err = np.abs(np.abs(v_tpf) - np.abs(v_nr))
    record["max_v_error"] = float(np.max(mag_err))
    record["mean_v_error"] = float(np.mean(mag_err))

    angle_err = np.abs(np.angle(v_tpf, deg=True) - np.angle(v_nr, deg=True))
    record["max_angle_error_deg"] = float(np.max(angle_err))

    if record["tpf_time_ms"] > 0:
        record["speedup"] = record["nr_time_ms"] / record["tpf_time_ms"]

    # 7. PASS/FAIL
    record["passed"] = record["max_v_error"] < tol_pass and result.converged

    if verbose:
        status = "PASS" if record["passed"] else "FAIL"
        eta_str = f"{record['eta']:.3f}" if record['eta'] < 100 else f"{record['eta']:.0f}"
        rho_str = f"{record['rho']:.4f}" if record['rho'] < 100 else "—"

        # Show additional metrics if computed
        extra = ""
        if record.get("rho_diag", np.inf) < np.inf:
            extra += f" rho_diag={record['rho_diag']:.3f}"
        if record.get("rho_corr", np.inf) < np.inf:
            extra += f" rho_corr={record['rho_corr']:.3f}"
        if record.get("contraction", np.inf) < np.inf:
            extra += f" kappa={record['contraction']:.3f}"

        print(f"  {status} {name:<30} n={record['n_bus']:<4} PV={n_pv:<3} "
              f"eta={eta_str:<7} rho={rho_str:<7} dV={record['max_v_error']:.2e} "
              f"out={record['tpf_outer_iter']}{extra}")

    return record


# ══════════════════════════════════════════════════════════════════════
#  Batch-Validierung
# ══════════════════════════════════════════════════════════════════════

def run_validation_suite(networks: dict, omega=1.0, tol_pass=1e-4, verbose=True, cold_start=False, analysis="full"):
    records = []
    for name, info in networks.items():
        record = validate_network(
            info["constructor"], name, omega=omega,
            tol_pass=tol_pass, verbose=verbose, cold_start=cold_start,
            analysis=analysis
        )
        records.append(record)
    return records


# ══════════════════════════════════════════════════════════════════════
#  Ergebnistabelle MIT Spektralradius
# ══════════════════════════════════════════════════════════════════════

def print_results_table(records: list, title: str = "", show_analysis: bool = False):
    print(f"\n{'='*200}")
    if title:
        print(f"  {title}")
        print(f"{'='*200}")

    if show_analysis:
        hdr = (f"  {'Netz':<28} {'n_d':<5} {'PV':<3} "
               f"{'eta':<7} {'rho':<7} {'rho_diag':<8} {'rho_corr':<8} {'kappa':<7} "
               f"{'Out':<4} {'Inn':<4} {'TPF ms':<7} "
               f"{'max dV':<10} {'Status'}")
    else:
        hdr = (f"  {'Netz':<30} {'n_d':<5} {'PV':<4} "
               f"{'eta':<8} {'rho(J_G)':<8} "
               f"{'NR It':<6} {'NR ms':<7} "
               f"{'Out':<5} {'Inn':<5} {'TPF ms':<7} "
               f"{'max dV':<10} {'PV dV':<10} {'dTheta':<8} "
               f"{'Status'}")
    print(hdr)
    print(f"  {'-'*198}")

    for r in records:
        if r["n_pv"] == 0:
            continue

        eta = r["eta"]
        eta_str = f"{eta:.4f}" if eta < 100 else f"{eta:.1f}"

        if show_analysis:
            rho = r.get("rho", np.inf)
            rho_str = f"{rho:.3f}" if rho < 100 else "—"

            rho_diag = r.get("rho_diag", np.inf)
            rho_diag_str = f"{rho_diag:.3f}" if rho_diag < 100 and np.isfinite(rho_diag) else "—"

            rho_corr = r.get("rho_corr", np.inf)
            rho_corr_str = f"{rho_corr:.3f}" if rho_corr < 100 and np.isfinite(rho_corr) else "—"

            contraction = r.get("contraction", np.inf)
            contraction_str = f"{contraction:.3f}" if contraction < 100 and np.isfinite(contraction) else "—"

            if r["passed"]:
                status = "PASS"
            elif not r.get("tpf_converged"):
                status = "DIV"
            elif r.get("error"):
                status = "ERR"
            else:
                status = "FAIL"

            max_v = r["max_v_error"]
            max_v_str = f"{max_v:.2e}" if max_v < 100 else "—"

            print(f"  {r['name']:<28} {r['n_bus']:<5} {r['n_pv']:<3} "
                  f"{eta_str:<7} {rho_str:<7} {rho_diag_str:<8} {rho_corr_str:<8} {contraction_str:<7} "
                  f"{r.get('tpf_outer_iter', 0):<4} {r.get('tpf_inner_iter_total', 0):<4} "
                  f"{r.get('tpf_time_ms', 0):<7.1f} "
                  f"{max_v_str:<10} {status}")
        else:
            rho = r.get("rho", np.inf)
            rho_str = f"{rho:.4f}" if rho < 100 else "—"

            max_v = r["max_v_error"]
            max_v_str = f"{max_v:.2e}" if max_v < 100 else "—"

            pv_v = r.get("max_pv_v_error", np.inf)
            pv_v_str = f"{pv_v:.2e}" if pv_v < 100 else "—"

            angle = r.get("max_angle_error_deg", np.inf)
            angle_str = f"{angle:.4f}" if angle < 100 else "—"

            if r["passed"]:
                status = "PASS"
            elif not r.get("tpf_converged"):
                status = "DIV"
            elif r.get("error"):
                status = "ERR"
            else:
                status = "FAIL"

            print(f"  {r['name']:<30} {r['n_bus']:<5} {r['n_pv']:<4} "
                  f"{eta_str:<8} {rho_str:<8} "
                  f"{r.get('nr_iter', -1):<6} {r.get('nr_time_ms', 0):<7.1f} "
                  f"{r.get('tpf_outer_iter', 0):<5} {r.get('tpf_inner_iter_total', 0):<5} "
                  f"{r.get('tpf_time_ms', 0):<7.1f} "
                  f"{max_v_str:<10} {pv_v_str:<10} {angle_str:<8} "
                  f"{status}")


def print_statistics(records: list, show_analysis: bool = False):
    tested = [r for r in records if r["n_pv"] > 0]
    passed = [r for r in tested if r["passed"]]
    converged = [r for r in tested if r.get("tpf_converged")]
    diverged = [r for r in tested if not r.get("tpf_converged") and r.get("nr_converged")]

    print(f"\n{'='*120}")
    print(f"  STATISTIKEN")
    print(f"{'='*120}")
    print(f"  Netze mit PV getestet:    {len(tested)}")
    print(f"  PASS:                     {len(passed)} ({100*len(passed)/max(len(tested),1):.0f}%)")
    print(f"  FAIL (konvergiert, dV):   {len([r for r in tested if r.get('tpf_converged') and not r['passed']])}")
    print(f"  FAIL (divergiert):        {len(diverged)}")

    if converged:
        etas = [r["eta"] for r in converged if r["eta"] < np.inf]
        rhos = [r["rho"] for r in converged if r["rho"] < np.inf]
        v_errors = [r["max_v_error"] for r in converged if r["max_v_error"] < np.inf]
        outer_iters = [r["tpf_outer_iter"] for r in converged if r["tpf_outer_iter"] > 0]

        if etas:
            print(f"\n  eta (konvergierte):")
            print(f"    min={min(etas):.4f}  max={max(etas):.4f}  "
                  f"median={np.median(etas):.4f}")
        if rhos:
            print(f"  rho(J_G) (konvergierte):")
            print(f"    min={min(rhos):.4f}  max={max(rhos):.4f}  "
                  f"median={np.median(rhos):.4f}")
        if v_errors:
            print(f"  max|dV| (konvergierte):")
            print(f"    min={min(v_errors):.2e}  max={max(v_errors):.2e}  "
                  f"median={np.median(v_errors):.2e}")
        if outer_iters:
            print(f"  Outer-Iterationen:")
            print(f"    min={min(outer_iters)}  max={max(outer_iters)}  "
                  f"median={np.median(outer_iters):.0f}")

    # -- rho vs. Konvergenz Analyse --
    print(f"\n  {'-'*80}")
    print(f"  KONVERGENZ-ANALYSE: Vorhersagekraft der verschiedenen Metriken")
    print(f"  {'-'*80}")

    all_with_rho = [r for r in tested if r["rho"] < np.inf and r.get("nr_converged")]

    if all_with_rho:
        def count_correct(metric_key, threshold=1.0):
            return sum(
                1 for r in all_with_rho
                if r.get(metric_key, np.inf) < np.inf
                and ((r[metric_key] < threshold and r["passed"]) or
                     (r[metric_key] >= threshold and not r["passed"]))
            )

        def count_available(metric_key):
            return sum(1 for r in all_with_rho if r.get(metric_key, np.inf) < np.inf)

        # Original ρ
        n_correct_rho = count_correct("rho")
        n_avail_rho = count_available("rho")
        if n_avail_rho > 0:
            print(f"    rho (voll):   {n_correct_rho:2d}/{n_avail_rho} korrekt ({100*n_correct_rho/max(n_avail_rho,1):.0f}%)")

        # ρ_diag
        n_correct_diag = count_correct("rho_diag")
        n_avail_diag = count_available("rho_diag")
        if n_avail_diag > 0:
            print(f"    rho_diag:     {n_correct_diag:2d}/{n_avail_diag} korrekt ({100*n_correct_diag/max(n_avail_diag,1):.0f}%)")

        # ρ_corr
        n_correct_corr = count_correct("rho_corr")
        n_avail_corr = count_available("rho_corr")
        if n_avail_corr > 0:
            print(f"    rho_corr:     {n_correct_corr:2d}/{n_avail_corr} korrekt ({100*n_correct_corr/max(n_avail_corr,1):.0f}%)")

        # κ (contraction)
        n_correct_kappa = count_correct("contraction")
        n_avail_kappa = count_available("contraction")
        if n_avail_kappa > 0:
            print(f"    kappa:        {n_correct_kappa:2d}/{n_avail_kappa} korrekt ({100*n_correct_kappa/max(n_avail_kappa,1):.0f}%)")

        print(f"\n    Regel: Wert < 1.0 bedeutet Konvergenz vorhergesagt")

    print(f"{'='*120}")


# ══════════════════════════════════════════════════════════════════════
#  PLOTTING — Mit Spektralradius in der Legende
# ══════════════════════════════════════════════════════════════════════

def plot_convergence(records: list, omega: float, save_path: str = None):
    """
    2×2 Konvergenz- und Performance-Plot.

    (0,0) PV |V|-Fehler vs. Outer-Iteration
    (0,1) Gesamtnetz max(|ΔV|) vs. kumulative Inner-Iteration
    (1,0) Solver-Zeit vs. Netzgröße (n_bus) — nur konvergierte
    (1,1) Solver-Zeit vs. PV-Ratio (n_pv/n_bus) — nur konvergierte
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plot_data = [r for r in records
                 if r["n_pv"] > 0
                 and (r["pv_v_errors"] or r["inner_v_change_all"])]

    timing_data = [r for r in records
                   if r["n_pv"] > 0
                   and r.get("nr_converged")
                   and r.get("tpf_converged")
                   and r.get("nr_time_ms", 0) > 0
                   and r.get("tpf_time_ms", 0) > 0]

    if not plot_data and not timing_data:
        print("  ! Keine Daten zum Plotten vorhanden.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), layout="constrained")

    cmap = plt.cm.tab20
    n_plots = max(len(plot_data), 1)
    colors = [cmap(i / n_plots) for i in range(n_plots)]

    legend_handles = []
    legend_labels = []

    # ══════════════════════════════════════════════════════════════
    #  (0,0) PV |V|-Fehler vs. Outer-Iteration
    # ══════════════════════════════════════════════════════════════
    ax00 = axes[0, 0]

    for i, rec in enumerate(plot_data):
        errors = rec["pv_v_errors"]
        if not errors:
            continue
        iters = list(range(1, len(errors) + 1))
        marker = "o" if rec["tpf_converged"] else "x"
        linestyle = "-" if rec["tpf_converged"] else "--"
        alpha = 0.85 if rec["tpf_converged"] else 0.6

        rho = rec.get("rho", np.inf)
        rho_str = f"rho={rho:.2f}" if rho < 100 else "rho=--"
        rx = rec.get("rx_ratio", np.nan)
        rx_str = f"R/X={rx:.1f}" if not np.isnan(rx) else "R/X=--"
        label = (f"{rec['name']} "
                 f"(η={rec['eta']:.2f}, {rho_str}, {rx_str}, PV={rec['n_pv']})")

        line, = ax00.loglog(
            iters, errors,
            color=colors[i], marker=marker, markersize=4,
            linestyle=linestyle, linewidth=1.5, alpha=alpha,
        )
        legend_handles.append(line)
        legend_labels.append(label)

    # ax00.axhline(y=1e-6, color="green", linestyle=":", linewidth=2.0)
    # ax00.axhline(y=1e-4, color="darkorange", linestyle="-.", linewidth=1.5)

    ax00.set_xlabel("Aeuszere Iteration l", fontsize=11)
    ax00.set_ylabel("max ||V_PV| - V_spec|| [p.u.]", fontsize=11)
    ax00.set_title("(a) PV-Spannungsfehler vs. Outer-Iteration", fontsize=12)
    ax00.grid(True, which="both", alpha=0.3)
    #ax00.set_xlim(1)
    ax00.set_ylim(bottom=1e-7, top=1e0)

    # ══════════════════════════════════════════════════════════════
    #  (0,1) Gesamtnetz-Fehler vs. kumulative Inner-Iteration
    # ══════════════════════════════════════════════════════════════
    ax01 = axes[0, 1]

    for i, rec in enumerate(plot_data):
        v_changes = rec["inner_v_change_all"]
        outer_starts = rec["outer_start_indices"]
        if not v_changes:
            continue
        x = list(range(1, len(v_changes) + 1))
        marker = "o" if rec["tpf_converged"] else "x"
        linestyle = "-" if rec["tpf_converged"] else "--"
        alpha = 0.8 if rec["tpf_converged"] else 0.5

        ax01.semilogy(
            x, v_changes,
            color=colors[i], marker=marker, markersize=1.5,
            linestyle=linestyle, linewidth=1.2, alpha=alpha,
        )
        for idx_start in outer_starts[1:]:
            if idx_start < len(v_changes):
                ax01.axvline(x=idx_start + 1,
                             color=colors[i], linestyle=":", linewidth=0.4, alpha=0.3)

    # ax01.axhline(y=1e-8, color="green", linestyle=":", linewidth=2.0)
    # ax01.axhline(y=1e-6, color="blue", linestyle="-.", linewidth=1.0)

    ax01.set_xlabel("Kumulative Inner-Iteration", fontsize=11)
    ax01.set_ylabel("max ||V_new| - |V_old|| [p.u.]", fontsize=11)
    ax01.set_title("(b) Gesamtnetz-Konvergenz (alle Busse)", fontsize=12)
    ax01.grid(True, which="both", alpha=0.3)
    ax01.set_ylim(bottom=1e-12, top=1e1)
    ax01.set_xlim(left=0.5)

    # ══════════════════════════════════════════════════════════════
    #  (1,0) Solver-Zeit vs. Netzgröße — NUR konvergierte (Box Plots)
    # ══════════════════════════════════════════════════════════════
    ax10 = axes[1, 0]

    if timing_data:
        from collections import defaultdict
        nr_by_size = defaultdict(list)
        tpf_by_size = defaultdict(list)
        for r in timing_data:
            nr_by_size[r["n_bus"]].append(r["nr_time_ms"])
            tpf_by_size[r["n_bus"]].append(r["tpf_time_ms"])

        sizes = sorted(nr_by_size.keys())

        if sizes:
            box_width = 0.25
            positions_nr = np.arange(len(sizes)) - box_width / 2
            positions_tpf = np.arange(len(sizes)) + box_width / 2

            bp_nr = ax10.boxplot(
                [nr_by_size[s] for s in sizes],
                positions=positions_nr,
                widths=box_width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor="tab:red", alpha=0.6),
                medianprops=dict(color="darkred", linewidth=1.5),
                whiskerprops=dict(color="tab:red", linewidth=1.2),
                capprops=dict(color="tab:red", linewidth=1.2),
                flierprops=dict(marker="s", markerfacecolor="tab:red",
                                markersize=4, alpha=0.6),
            )

            bp_tpf = ax10.boxplot(
                [tpf_by_size[s] for s in sizes],
                positions=positions_tpf,
                widths=box_width * 0.8,
                patch_artist=True,
                boxprops=dict(facecolor="tab:blue", alpha=0.6),
                medianprops=dict(color="darkblue", linewidth=1.5),
                whiskerprops=dict(color="tab:blue", linewidth=1.2),
                capprops=dict(color="tab:blue", linewidth=1.2),
                flierprops=dict(marker="o", markerfacecolor="tab:blue",
                                markersize=4, alpha=0.6),
            )

            ax10.set_xticks(range(len(sizes)))
            ax10.set_xticklabels(sizes)
            ax10.set_xlim(-0.5, len(sizes) - 0.5)
            ax10.set_yscale("log")

    ax10.set_xlabel("Netzgroesse (Anzahl Busse im d-Block)", fontsize=11)
    ax10.set_ylabel("Rechenzeit [ms]", fontsize=11)
    ax10.set_title("(c) Rechenzeit vs. Netzgroesse (nur konvergierte)", fontsize=12)
    ax10.grid(True, which="both", alpha=0.3)

    # ══════════════════════════════════════════════════════════════
    #  (1,1) Solver-Zeit vs. PV-Ratio — NUR konvergierte
    # ══════════════════════════════════════════════════════════════
    ax11 = axes[1, 1]

    if timing_data:
        # PV-Ratio berechnen
        pv_ratios = np.array([r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
                              for r in timing_data])
        nr_times_r = np.array([r["nr_time_ms"] for r in timing_data])
        tpf_times_r = np.array([r["tpf_time_ms"] for r in timing_data])
        n_bus_arr = np.array([r["n_bus"] for r in timing_data])

        # Farbe nach Netzgröße
        size_norm = (n_bus_arr - n_bus_arr.min()) / max(n_bus_arr.max() - n_bus_arr.min(), 1)
        size_cmap = plt.cm.viridis

        # NR Zeiten
        sc_nr = ax11.scatter(
            pv_ratios * 100, nr_times_r,
            c=size_norm, cmap=size_cmap, marker="s", s=60,
            edgecolors="tab:red", linewidths=1.5, zorder=5, alpha=0.8,
            label="NR (pandapower)",
        )

        # TPF Zeiten
        sc_tpf = ax11.scatter(
            pv_ratios * 100, tpf_times_r,
            c=size_norm, cmap=size_cmap, marker="o", s=60,
            edgecolors="tab:blue", linewidths=1.5, zorder=5, alpha=0.8,
            label="TPF Methode A",
        )

        # Annotationen: Netzgröße an TPF-Punkten
        for r in timing_data:
            ratio = r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
            ax11.annotate(
                f"n={r['n_bus']}",
                (ratio * 100, r["tpf_time_ms"]),
                fontsize=5, alpha=0.5,
                textcoords="offset points", xytext=(3, 3),
            )

        # Verbindungslinien nach Netzgröße gruppiert
        from collections import defaultdict
        nr_by_size = defaultdict(list)
        tpf_by_size = defaultdict(list)
        for r in timing_data:
            ratio = r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
            nr_by_size[r["n_bus"]].append((ratio, r["nr_time_ms"]))
            tpf_by_size[r["n_bus"]].append((ratio, r["tpf_time_ms"]))

        # DEBUG: Print data for n=119
        if 119 in tpf_by_size:
            import sys
            print(f"\n  DEBUG n=119: {len(tpf_by_size[119])} points", flush=True)
            for i, (ratio, time_ms) in enumerate(sorted(tpf_by_size[119], key=lambda x: x[0])):
                print(f"    point {i}: pv_ratio={ratio*100:.2f}%, time={time_ms:.1f}ms", flush=True)

        for size in sorted(nr_by_size.keys()):
            nr_points = sorted(nr_by_size[size], key=lambda x: x[0])
            nr_x = [p[0] * 100 for p in nr_points]
            nr_y = [p[1] for p in nr_points]
            if len(nr_x) > 1:
                ax11.plot(nr_x, nr_y, color="tab:red", linestyle="-", linewidth=1.0, alpha=0.5, zorder=2)

            tpf_points = sorted(tpf_by_size[size], key=lambda x: x[0])
            tpf_x = [p[0] * 100 for p in tpf_points]
            tpf_y = [p[1] for p in tpf_points]
            if len(tpf_x) > 1:
                ax11.plot(tpf_x, tpf_y, color="tab:blue", linestyle="-", linewidth=1.0, alpha=0.5, zorder=2)

            if tpf_y:
                max_idx = np.argmax(tpf_y)
                ax11.annotate(
                    f"n={size}",
                    (tpf_x[max_idx], tpf_y[max_idx]),
                    fontsize=7, color="tab:blue", fontweight="bold",
                    textcoords="offset points", xytext=(5, 0),
                )

        ax11.set_yscale("log")

        # Colorbar für Netzgröße
        cbar = plt.colorbar(sc_tpf, ax=ax11, pad=0.02, fraction=0.04)
        cbar.set_label("Netzgroesse (normiert)", fontsize=9)
        # Setze Colorbar-Ticks auf tatsächliche Busgrößen
        tick_vals = np.linspace(0, 1, 5)
        tick_labels = [f"{int(n_bus_arr.min() + t * (n_bus_arr.max() - n_bus_arr.min()))}"
                       for t in tick_vals]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(tick_labels)

    ax11.set_xlabel("PV-Durchdringung: n_PV / n_total [%]", fontsize=11)
    ax11.set_ylabel("Rechenzeit [ms]", fontsize=11)
    ax11.set_title("(d) Rechenzeit vs. PV-Ratio (nur konvergierte)", fontsize=12)
    ax11.grid(True, which="both", alpha=0.3)

    # ══════════════════════════════════════════════════════════════
    #  LEGENDE oben zentriert
    # ══════════════════════════════════════════════════════════════
    n_conv = sum(1 for r in plot_data if r["tpf_converged"])
    n_div = sum(1 for r in plot_data if not r["tpf_converged"])

    timing_legend_handles = [
        Line2D([0], [0], color="tab:red", marker="s", linestyle="--",
               markersize=8, label="NR (pandapower)"),
        Line2D([0], [0], color="tab:blue", marker="o", linestyle="--",
               markersize=8, label="TPF Methode A"),
        Line2D([0], [0], color="green", linestyle=":", linewidth=2,
               label="tol_pv = 1e-6"),
        Line2D([0], [0], color="darkorange", linestyle="-.", linewidth=1.5,
               label="1e-4 (PASS)"),
    ]

    all_handles = legend_handles + timing_legend_handles
    all_labels = legend_labels + [h.get_label() for h in timing_legend_handles]

    fig.legend(
        all_handles, all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(6, len(all_handles)),
        fontsize=7,
        framealpha=0.9,
        edgecolor="gray",
    )

    fig.suptitle(
        f"Methode A: Konvergenz & Performance — {len(plot_data)} Netze "
        f"({n_conv} konv., {n_div} div.), w = {omega}",
        fontsize=14, y=1.06,
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")

    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  Text-Export für Subplot (b): Kumulative Inner-Iteration
# ══════════════════════════════════════════════════════════════════════

def export_inner_iteration_data(records: list, filepath: str, omega: float):
    """
    Exportiert die Daten aus Subplot (b) - kumulative Inner-Iteration.
    Format: iter, max_delta_V, outer_iter
    """
    with open(filepath, 'w') as f:
        f.write("# Inner iteration convergence data (subplot b)\n")
        f.write(f"# omega = {omega}\n")
        f.write("# Columns: iter, max_delta_V, outer_iter\n")
        f.write("# \n")

        for rec in records:
            v_changes = rec.get("inner_v_change_all", [])
            outer_starts = rec.get("outer_start_indices", [])
            if not v_changes:
                continue

            f.write(f"# Network: {rec['name']}\n")
            f.write(f"# n_bus={rec['n_bus']}, n_pv={rec['n_pv']}, "
                    f"eta={rec['eta']:.4f}, rho={rec.get('rho', np.nan):.4f}, "
                    f"rx_ratio={rec.get('rx_ratio', np.nan):.2f}\n")

            outer_iter = 1
            outer_idx = 0
            for i, delta_v in enumerate(v_changes, 1):
                while outer_idx + 1 < len(outer_starts) and i > outer_starts[outer_idx + 1]:
                    outer_iter += 1
                    outer_idx += 1
                f.write(f"{i:6d}  {delta_v:.10e}  {outer_iter:2d}\n")

            f.write("# \n")

    print(f"\n  Exportiert: {filepath}")


# ══════════════════════════════════════════════════════════════════════
#  TXT and CSV Export Functions
# ══════════════════════════════════════════════════════════════════════

def save_results_to_file(records: list, filepath: str, suite: str, omega: float,
                          tol_pass: float, show_analysis: bool = False):
    """
    Saves validation results to a formatted TXT file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, 'w') as f:
        f.write("=" * 200 + "\n")
        f.write("VALIDATION RESULTS: TPF Methode A\n")
        f.write("=" * 200 + "\n")
        f.write(f"Suite:         {suite}\n")
        f.write(f"Omega (w):     {omega}\n")
        f.write(f"Analysis:      {'full' if show_analysis else 'basic'}\n")
        f.write(f"Tolerance:     {tol_pass} (PASS threshold)\n")
        f.write(f"Timestamp:     {timestamp}\n")
        f.write("=" * 200 + "\n\n")

        tested = [r for r in records if r["n_pv"] > 0]
        if not tested:
            f.write("No networks with PV to report.\n")
            return

        if show_analysis:
            hdr = (f"{'Netz':<22} {'n_d':<5} {'PV':<3} {'PQ':<4} {'PV%':<5} {'R/X':<5} "
                   f"{'eta':<8} {'rho(J_G)':<10} {'rho_diag':<9} {'rho_corr':<9} {'kappa':<7} "
                   f"{'Out':<4} {'Inn':<5} {'TPF ms':<8} {'max dV':<11} {'Status':<6}\n")
        else:
            hdr = (f"{'Netz':<22} {'n_d':<5} {'PV':<3} {'PQ':<4} {'PV%':<5} {'R/X':<5} "
                   f"{'eta':<8} {'rho(J_G)':<10} {'NR It':<6} {'NR ms':<7} "
                   f"{'Out':<4} {'Inn':<5} {'TPF ms':<8} {'Speedup':<8} "
                   f"{'max dV':<11} {'mean dV':<11} {'PV dV':<11} {'dTheta(deg)':<10} {'Status':<6}\n")

        f.write(hdr)
        f.write("-" * 200 + "\n")

        for r in tested:
            name = r['name'][:22]
            n_bus = r['n_bus']
            n_pv = r['n_pv']
            n_pq = r.get('n_pq', 0)
            pv_pct = r.get('pv_ratio', 0) * 100
            rx = r.get('rx_ratio', np.nan)
            rx_str = f"{rx:.2f}" if not np.isnan(rx) else "—"

            eta = r["eta"]
            eta_str = f"{eta:.4f}" if eta < 100 else f"{eta:.1f}"

            rho = r.get("rho", np.inf)
            rho_str = f"{rho:.4f}" if rho < 100 else "—"

            nr_iter = r.get('nr_iter', -1)
            nr_ms = r.get('nr_time_ms', 0)
            out_iter = r.get('tpf_outer_iter', 0)
            inn_iter = r.get('tpf_inner_iter_total', 0)
            tpf_ms = r.get('tpf_time_ms', 0)

            speedup = r.get('speedup', 0)
            speedup_str = f"{speedup:.1f}x" if speedup > 0 else "—"

            max_v = r["max_v_error"]
            max_v_str = f"{max_v:.2e}" if max_v < 100 else "—"

            mean_v = r.get("mean_v_error", np.inf)
            mean_v_str = f"{mean_v:.2e}" if mean_v < 100 else "—"

            pv_v = r.get("max_pv_v_error", np.inf)
            pv_v_str = f"{pv_v:.2e}" if pv_v < 100 else "—"

            angle = r.get("max_angle_error_deg", np.inf)
            angle_str = f"{angle:.4f}" if angle < 100 else "—"

            if r["passed"]:
                status = "PASS"
            elif not r.get("tpf_converged"):
                status = "DIV"
            elif r.get("error"):
                status = "ERR"
            else:
                status = "FAIL"

            if show_analysis:
                rho_diag = r.get("rho_diag", np.inf)
                rho_diag_str = f"{rho_diag:.3f}" if rho_diag < 100 and np.isfinite(rho_diag) else "—"

                rho_corr = r.get("rho_corr", np.inf)
                rho_corr_str = f"{rho_corr:.3f}" if rho_corr < 100 and np.isfinite(rho_corr) else "—"

                contraction = r.get("contraction", np.inf)
                contraction_str = f"{contraction:.3f}" if contraction < 100 and np.isfinite(contraction) else "—"

                f.write(f"{name:<22} {n_bus:<5} {n_pv:<3} {n_pq:<4} {pv_pct:<5.1f} {rx_str:<5} "
                        f"{eta_str:<8} {rho_str:<10} {rho_diag_str:<9} {rho_corr_str:<9} {contraction_str:<7} "
                        f"{out_iter:<4} {inn_iter:<5} {tpf_ms:<8.1f} {max_v_str:<11} {status:<6}\n")
            else:
                f.write(f"{name:<22} {n_bus:<5} {n_pv:<3} {n_pq:<4} {pv_pct:<5.1f} {rx_str:<5} "
                        f"{eta_str:<8} {rho_str:<10} {nr_iter:<6} {nr_ms:<7.1f} "
                        f"{out_iter:<4} {inn_iter:<5} {tpf_ms:<8.1f} {speedup_str:<8} "
                        f"{max_v_str:<11} {mean_v_str:<11} {pv_v_str:<11} {angle_str:<10} {status:<6}\n")

        f.write("\n")
        f.write("=" * 200 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 200 + "\n")

        passed = [r for r in tested if r["passed"]]
        converged = [r for r in tested if r.get("tpf_converged")]
        diverged = [r for r in tested if not r.get("tpf_converged") and r.get("nr_converged")]

        f.write(f"  Networks tested with PV:    {len(tested)}\n")
        f.write(f"  PASS:                       {len(passed)} ({100*len(passed)/max(len(tested),1):.0f}%)\n")
        f.write(f"  FAIL (converged, dV):       {len([r for r in tested if r.get('tpf_converged') and not r['passed']])}\n")
        f.write(f"  FAIL (diverged):            {len(diverged)}\n")

        if converged:
            etas = [r["eta"] for r in converged if r["eta"] < np.inf]
            rhos = [r["rho"] for r in converged if r["rho"] < np.inf]
            v_errors = [r["max_v_error"] for r in converged if r["max_v_error"] < np.inf]
            outer_iters = [r["tpf_outer_iter"] for r in converged if r["tpf_outer_iter"] > 0]

            if etas:
                f.write(f"\n  eta (converged):\n")
                f.write(f"    min={min(etas):.4f}  max={max(etas):.4f}  median={np.median(etas):.4f}\n")
            if rhos:
                f.write(f"  rho(J_G) (converged):\n")
                f.write(f"    min={min(rhos):.4f}  max={max(rhos):.4f}  median={np.median(rhos):.4f}\n")
            if v_errors:
                f.write(f"  max|dV| (converged):\n")
                f.write(f"    min={min(v_errors):.2e}  max={max(v_errors):.2e}  median={np.median(v_errors):.2e}\n")
            if outer_iters:
                f.write(f"  Outer iterations:\n")
                f.write(f"    min={min(outer_iters)}  max={max(outer_iters)}  median={np.median(outer_iters):.0f}\n")

        all_with_rho = [r for r in tested if r["rho"] < np.inf and r.get("nr_converged")]
        if show_analysis and all_with_rho:
            f.write(f"\n  {'-' * 80}\n")
            f.write(f"  CONVERGENCE ANALYSIS: Prediction accuracy of metrics\n")
            f.write(f"  {'-' * 80}\n")

            def count_correct(metric_key, threshold=1.0):
                return sum(
                    1 for r in all_with_rho
                    if r.get(metric_key, np.inf) < np.inf
                    and ((r[metric_key] < threshold and r["passed"]) or
                         (r[metric_key] >= threshold and not r["passed"]))
                )

            def count_available(metric_key):
                return sum(1 for r in all_with_rho if r.get(metric_key, np.inf) < np.inf)

            n_correct_rho = count_correct("rho")
            n_avail_rho = count_available("rho")
            if n_avail_rho > 0:
                f.write(f"    rho (full):    {n_correct_rho:2d}/{n_avail_rho} correct ({100*n_correct_rho/max(n_avail_rho,1):.0f}%)\n")

            n_correct_diag = count_correct("rho_diag")
            n_avail_diag = count_available("rho_diag")
            if n_avail_diag > 0:
                f.write(f"    rho_diag:      {n_correct_diag:2d}/{n_avail_diag} correct ({100*n_correct_diag/max(n_avail_diag,1):.0f}%)\n")

            n_correct_corr = count_correct("rho_corr")
            n_avail_corr = count_available("rho_corr")
            if n_avail_corr > 0:
                f.write(f"    rho_corr:      {n_correct_corr:2d}/{n_avail_corr} correct ({100*n_correct_corr/max(n_avail_corr,1):.0f}%)\n")

            n_correct_kappa = count_correct("contraction")
            n_avail_kappa = count_available("contraction")
            if n_avail_kappa > 0:
                f.write(f"    kappa:         {n_correct_kappa:2d}/{n_avail_kappa} correct ({100*n_correct_kappa/max(n_avail_kappa,1):.0f}%)\n")

            f.write(f"\n    Rule: value < 1.0 predicts convergence\n")

        failed = [r for r in tested if not r["passed"]]
        if failed:
            f.write("\nFAILED NETWORKS:\n")
            for r in failed:
                reason = "DIV" if not r.get("tpf_converged") else ("ERR: " + r.get("error", "dV") if r.get("error") else "accuracy")
                f.write(f"  {r['name']:<22} {reason}\n")

        f.write("=" * 200 + "\n")

    print(f"\n  TXT saved: {filepath}")


def save_csv(records: list, filepath: str):
    """
    Saves validation results to a CSV file for programmatic analysis.
    """
    import csv

    fieldnames = [
        "name", "n_bus", "n_pv", "n_pq", "pv_ratio", "rx_ratio",
        "eta", "rho", "rho_diag", "rho_corr", "contraction",
        "nr_converged", "nr_iter", "nr_time_ms",
        "tpf_converged", "tpf_outer_iter", "tpf_inner_iter_total", "tpf_time_ms",
        "max_v_error", "mean_v_error", "max_angle_error_deg", "max_pv_v_error",
        "speedup", "passed", "error"
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            if r["n_pv"] == 0:
                continue

            row = {k: r.get(k, np.nan) for k in fieldnames}
            row["nr_converged"] = bool(row["nr_converged"])
            row["tpf_converged"] = bool(row["tpf_converged"])
            row["passed"] = bool(row["passed"])

            for key in ["eta", "rho", "rho_diag", "rho_corr", "contraction",
                        "max_v_error", "mean_v_error", "max_angle_error_deg",
                        "max_pv_v_error", "rx_ratio", "speedup"]:
                if row[key] is None or not np.isfinite(row[key]):
                    row[key] = np.nan

            writer.writerow(row)

    print(f"  CSV saved: {filepath}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validierung + Spektralradius + Konvergenz-Plot: TPF Methode A"
    )
    parser.add_argument(
        "--suite", choices=["quick", "radial", "salazar", "salazar_scaling", "salazar_low_vm",
                            "salazar_low_rx05", "salazar_low_rx10", "full",
                            "ieee", "pegase", "rte", "large", "standard"],
        default="salazar_scaling",
        help="Testsuite: quick (4 Netze), radial (ohne IEEE vermascht), salazar/salazar_scaling, "
             "salazar_low_vm (niedrige PV-Spannung für NR-scheiternde Netze), "
             "salazar_low_rx05 (R/X=0.5), salazar_low_rx10 (R/X=1.0), "
             "full (alles), ieee (IEEE 9-300), pegase (PEGASE), rte (French), large (>100), standard (alle)"
    )
    parser.add_argument("--omega", type=float, default=1.0,
                        help="Q-Relaxationsfaktor w (default: 1.0)")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="PASS-Schwelle für max|ΔV| (default: 1e-4)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Keinen Plot erzeugen (nur Tabelle)")
    parser.add_argument("--save", type=str, default=None,
                        help="Plot speichern unter diesem Pfad")
    parser.add_argument("--network", type=str, default=None,
                        help="Nur dieses Netzwerk testen (Name muss exakt stimmen)")
    parser.add_argument("--list", action="store_true",
                        help="Liste aller verfügbaren Netzwerke in der Suite anzeigen")
    parser.add_argument("--export", type=str, default=None,
                        help="Exportiere subplot (b) Daten in Textdatei")
    parser.add_argument("--cold-start", action="store_true",
                        help="Nutze cold start fuer innere FPI (jede outer iteration startet mit V=1.0 pu)")
    parser.add_argument("--size", type=int, default=None,
                        help="Filter networks by bus count (e.g., 20 for sz_20_*)")
    parser.add_argument("--analysis", choices=["full", "diagonal", "corrected", "contraction"],
                        default="full",
                        help="Konvergenz-Analyse-Methode: full (alle), diagonal (ρ_diag), "
                             "corrected (ρ_corr), contraction (κ)")
    parser.add_argument("--output-dir", "-d", type=str, default="./tau_benchmark_results",
                        help="Output directory for TXT/CSV results (default: ./tau_benchmark_results)")
    args = parser.parse_args()

    print("+========================================================================+")
    print("|  VALIDIERUNG + SPEKTRALRADIUS + KONVERGENZ-PLOT: TPF Methode A            |")
    print("|  Berechnet rho(J_G) numerisch fuer jedes Netz                            |")
    print("|  Analysemodus: {:<57}  |".format(args.analysis))
    print("+========================================================================+")

    # Netze laden
    if args.suite == "quick":
        networks = get_quick_test_networks()
    elif args.suite == "radial":
        networks = get_radial_only_networks()
    elif args.suite == "salazar":
        networks = get_salazar_pv_networks()
    elif args.suite == "salazar_scaling":
        networks = get_salazar_scaling_networks()
    elif args.suite == "salazar_low_vm":
        networks = get_salazar_low_vm_networks()
    elif args.suite == "salazar_low_rx05":
        networks = get_salazar_low_rx05_networks()
    elif args.suite == "salazar_low_rx10":
        networks = get_salazar_low_rx10_networks()
    elif args.suite == "ieee":
        networks = get_ieee_networks()
    elif args.suite == "pegase":
        networks = get_pegase_networks()
    elif args.suite == "rte":
        networks = get_rte_networks()
    elif args.suite == "large":
        networks = get_large_networks()
    elif args.suite == "standard":
        networks = get_all_standard_networks()
    else:
        networks = get_comprehensive_networks()


    cold_str = " (COLD START)" if args.cold_start else ""
    print(f"\n  Suite: '{args.suite}' - {len(networks)} Netze")
    print(f"  w = {args.omega}, PASS-Schwelle = {args.tol:.0e}{cold_str}\n")

    # Handle --size filter (filter networks by bus count prefix, e.g., sz_20_*)
    if args.size is not None:
        size_prefix = f"sz_{args.size}_"
        networks = {k: v for k, v in networks.items() if k.startswith(size_prefix)}
        print(f"  -> Nach Groesse gefiltert: '{size_prefix}' -> {len(networks)} Netze")

    # Handle --list
    if args.list:
        print(f"\n  Verfügbare Netzwerke in Suite '{args.suite}':")
        for i, name in enumerate(sorted(networks.keys()), 1):
            print(f"    {i:2}. {name}")
        print(f"\n{'='*90}")
        return

    # Handle --network filter
    if args.network:
        if args.network not in networks:
            print(f"\n  FEHLER: Netzwerk '{args.network}' nicht gefunden in Suite '{args.suite}'")
            print(f"  Verwende --list um alle Netzwerke anzuzeigen.")
            print(f"\n{'='*90}")
            return
        networks = {args.network: networks[args.network]}
        print(f"  -> Nur Netzwerk '{args.network}' wird getestet.\n")

    # Validierung
    t_start = time.perf_counter()
    records = run_validation_suite(
        networks, omega=args.omega, tol_pass=args.tol, verbose=True,
        cold_start=args.cold_start, analysis=args.analysis
    )
    t_total = time.perf_counter() - t_start

    # Determine if we should show analysis columns
    show_analysis = args.analysis in ["full", "diagonal", "corrected", "contraction"] or any(
        r.get("rho_diag", np.inf) < np.inf or r.get("rho_corr", np.inf) < np.inf or r.get("contraction", np.inf) < np.inf
        for r in records
    )

    # Tabelle
    cold_str = " (COLD START)" if args.cold_start else ""
    print_results_table(records, title=f"Methode A — Suite '{args.suite}', w={args.omega}{cold_str}", show_analysis=show_analysis)
    print_statistics(records, show_analysis=show_analysis)

    # Gesamtergebnis
    tested = [r for r in records if r["n_pv"] > 0]
    n_pass = sum(1 for r in tested if r["passed"])
    n_total = len(tested)

    print(f"\n  Gesamtzeit: {t_total:.1f} s")
    print(f"  Gesamtergebnis: {n_pass}/{n_total} PASS "
          f"({100*n_pass/max(n_total,1):.0f}%)")

    if n_pass == n_total:
        print(f"\n  [OK] METHODE A VALIDIERT FUER ALLE {n_total} TESTNETZE!")
    else:
        n_div = sum(1 for r in tested if not r.get("tpf_converged")
                    and r.get("nr_converged"))
        n_acc = sum(1 for r in tested if r.get("tpf_converged")
                    and not r["passed"])
        print(f"\n  ! {n_total - n_pass} Tests fehlgeschlagen:")
        if n_div:
            print(f"    - {n_div} divergiert (w anpassen oder eta > 1)")
        if n_acc:
            print(f"    - {n_acc} Genauigkeit unzureichend")

    # Export
    if args.export:
        export_inner_iteration_data(records, args.export, omega=args.omega)

    # Save TXT and CSV results
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(output_dir, f"validation_{args.suite}_w{args.omega}_{timestamp}.txt")
        csv_path = os.path.join(output_dir, f"validation_{args.suite}_w{args.omega}_{timestamp}.csv")

        save_results_to_file(records, txt_path, args.suite, args.omega, args.tol, show_analysis=show_analysis)
        save_csv(records, csv_path)

    # DEBUG: Print timing_data structure before plotting
    timing_data = [r for r in records
                   if r["n_pv"] > 0
                   and r.get("nr_converged")
                   and r.get("tpf_converged")
                   and r.get("nr_time_ms", 0) > 0
                   and r.get("tpf_time_ms", 0) > 0]
    from collections import defaultdict
    tpf_by_size = defaultdict(list)
    for r in timing_data:
        ratio = r.get("pv_ratio", r["n_pv"] / max(r["n_bus"], 1))
        tpf_by_size[r["n_bus"]].append((ratio, r["tpf_time_ms"]))

    # Print detailed summary of all n_bus groups including network names
    import sys
    for size in sorted(tpf_by_size.keys()):
        # Get detailed info including network names
        size_records = [r for r in timing_data if r['n_bus'] == size]
        size_records_sorted = sorted(size_records, key=lambda r: r.get('pv_ratio', 0))
        sys.stderr.write(f"  n={size}: {len(size_records)} records\n")
        for r in size_records_sorted:
            sys.stderr.write(f"    {r['name']}: pv_ratio={r.get('pv_ratio', 0)*100:.2f}%, time={r['tpf_time_ms']:.1f}ms\n")

    # Plot
    if not args.no_plot:
        save_path = args.save or f"convergence_method_a_{args.suite}_omega{args.omega}.png"
        plot_convergence(records, omega=args.omega, save_path=save_path)

    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
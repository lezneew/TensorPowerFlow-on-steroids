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
    create_salazar_network,
)

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
    numerisch via Finite-Differenzen.

    ρ(I + ω·H·D⁻¹)

    wobei:
        D = diag(2·X_kk)           (Thévenin-Approximation)
        H_kj = ∂|V_k|²/∂Q_j       (wahre Sensitivitätsmatrix, numerisch)

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

    # Vorberechnung
    Z_B = np.linalg.inv(network.Y_dd)
    K = -Z_B
    L = (K @ network.Y_ds @ network.v_s).reshape(-1, 1)

    # Thévenin-Reaktanz
    X_kk = np.imag(Z_B[pv_idx, pv_idx])

    # D-Matrix
    D_inv = np.diag(1.0 / (2.0 * np.where(np.abs(X_kk) > 1e-12, X_kk, 1e-12)))

    # ── Basislösung: Innere FPI mit aktuellem s_nom ──
    s_base = network.s_nom.copy().reshape(-1, 1)

    def solve_inner_fpi(s_work, max_iter=200, tol=1e-10):
        """Löst innere FPI und gibt |V|² an PV-Knoten zurück."""
        V = np.ones((bphi, 1), dtype=np.complex128)
        S_conj = np.conj(s_work)
        for _ in range(max_iter):
            LAMBDA = S_conj * (1.0 / np.conj(V))
            V_new = K @ LAMBDA + L
            if np.max(np.abs(np.abs(V_new) - np.abs(V))) < tol:
                return np.abs(V_new[pv_idx, 0]) ** 2
            V = V_new
        return np.abs(V[pv_idx, 0]) ** 2

    # Basislösung
    v2_base = solve_inner_fpi(s_base)

    # ── H-Matrix via Finite-Differenzen ──
    H = np.zeros((n_pv, n_pv))

    for j in range(n_pv):
        s_pert = s_base.copy()
        s_pert[pv_idx[j], 0] += 1j * delta_q
        v2_pert = solve_inner_fpi(s_pert)
        H[:, j] = (v2_pert - v2_base) / delta_q

    # ── Iterationsmatrix: J_G = I + ω·H·D⁻¹ ──
    J_G = np.eye(n_pv) + omega * H @ D_inv

    # Spektralradius
    eigenvalues = np.linalg.eigvals(J_G)
    rho = float(np.max(np.abs(eigenvalues)))

    return rho


# ══════════════════════════════════════════════════════════════════════
#  Einzelnetz-Validierung
# ══════════════════════════════════════════════════════════════════════

def validate_network(net_constructor, name, omega=1.0, tol_pass=1e-4, verbose=True):
    """
    Validiert Methode A gegen NR. Gibt Record mit Konvergenzhistorie zurück.
    """
    record = {
        "name": name,
        "n_bus": 0, "n_pv": 0, "n_pq": 0,
        "eta": np.inf,
        "rho": np.inf,
        "nr_converged": False, "nr_iter": -1, "nr_time_ms": 0.0,
        "tpf_converged": False, "tpf_outer_iter": 0,
        "tpf_inner_iter_total": 0, "tpf_time_ms": 0.0,
        "max_v_error": np.inf, "mean_v_error": np.inf,
        "max_angle_error_deg": np.inf, "max_pv_v_error": np.inf,
        "speedup": 0.0, "passed": False, "error": None,
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
            print(f"  ✗ {name:<30} FEHLER (Constructor): {e}")
        return record

    # 1. NR-Referenz
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
    try:
        nr_result = nr_solver.solve_from_net(net)
    except Exception as e:
        record["error"] = f"NR: {str(e)[:50]}"
        if verbose:
            print(f"  ✗ {name:<30} FEHLER (NR): {e}")
        return record

    if not nr_result.converged:
        record["error"] = "NR divergiert"
        if verbose:
            print(f"  ✗ {name:<30} NR divergiert")
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
            print(f"  ✗ {name:<30} FEHLER (Builder): {e}")
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
            print(f"  ─ {name:<30} keine PV (übersprungen)")
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
        tol=1e-8, max_iter_inner=50, max_iter_outer=100,
        tol_pv=1e-6, omega=omega, enforce_q_lims=False,
    )

    try:
        result = solver.solve(network)
    except Exception as e:
        record["error"] = f"TPF: {str(e)[:50]}"
        if verbose:
            print(f"  ✗ {name:<30} FEHLER (TPF): {e}")
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

    # 6. Vergleich
    v_tpf = result.voltages.flatten()
    v_nr = nr_result.voltages[d_idx]

    if v_tpf.shape[0] != v_nr.shape[0]:
        record["error"] = f"Dim mismatch: TPF={v_tpf.shape[0]} NR={v_nr.shape[0]}"
        if verbose:
            print(f"  ✗ {name:<30} Dimensionsfehler")
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
        status = "✓" if record["passed"] else "✗"
        eta_str = f"{record['eta']:.3f}" if record['eta'] < 100 else f"{record['eta']:.0f}"
        rho_str = f"{record['rho']:.4f}" if record['rho'] < 100 else "—"
        print(f"  {status} {name:<30} n={record['n_bus']:<4} PV={n_pv:<3} "
              f"η={eta_str:<7} ρ={rho_str:<7} ΔV={record['max_v_error']:.2e} "
              f"out={record['tpf_outer_iter']}")

    return record


# ══════════════════════════════════════════════════════════════════════
#  Batch-Validierung
# ══════════════════════════════════════════════════════════════════════

def run_validation_suite(networks: dict, omega=1.0, tol_pass=1e-4, verbose=True):
    records = []
    for name, info in networks.items():
        record = validate_network(
            info["constructor"], name, omega=omega,
            tol_pass=tol_pass, verbose=verbose
        )
        records.append(record)
    return records


# ══════════════════════════════════════════════════════════════════════
#  Ergebnistabelle MIT Spektralradius
# ══════════════════════════════════════════════════════════════════════

def print_results_table(records: list, title: str = ""):
    print(f"\n{'═'*160}")
    if title:
        print(f"  {title}")
        print(f"{'═'*160}")

    hdr = (f"  {'Netz':<30} {'n_d':<5} {'PV':<4} "
           f"{'η':<8} {'ρ(J_G)':<8} "
           f"{'NR It':<6} {'NR ms':<7} "
           f"{'Out':<5} {'Inn':<5} {'TPF ms':<7} "
           f"{'max ΔV':<10} {'PV ΔV':<10} {'Δθ°':<8} "
           f"{'Status'}")
    print(hdr)
    print(f"  {'─'*158}")

    for r in records:
        if r["n_pv"] == 0:
            continue

        eta = r["eta"]
        eta_str = f"{eta:.4f}" if eta < 100 else f"{eta:.1f}"

        rho = r.get("rho", np.inf)
        rho_str = f"{rho:.4f}" if rho < 100 else "—"

        max_v = r["max_v_error"]
        max_v_str = f"{max_v:.2e}" if max_v < 100 else "—"

        pv_v = r.get("max_pv_v_error", np.inf)
        pv_v_str = f"{pv_v:.2e}" if pv_v < 100 else "—"

        angle = r.get("max_angle_error_deg", np.inf)
        angle_str = f"{angle:.4f}" if angle < 100 else "—"

        if r["passed"]:
            status = "✓ PASS"
        elif not r.get("tpf_converged"):
            status = "✗ DIV"
        elif r.get("error"):
            status = "✗ ERR"
        else:
            status = "✗ FAIL"

        print(f"  {r['name']:<30} {r['n_bus']:<5} {r['n_pv']:<4} "
              f"{eta_str:<8} {rho_str:<8} "
              f"{r.get('nr_iter', -1):<6} {r.get('nr_time_ms', 0):<7.1f} "
              f"{r.get('tpf_outer_iter', 0):<5} {r.get('tpf_inner_iter_total', 0):<5} "
              f"{r.get('tpf_time_ms', 0):<7.1f} "
              f"{max_v_str:<10} {pv_v_str:<10} {angle_str:<8} "
              f"{status}")


def print_statistics(records: list):
    tested = [r for r in records if r["n_pv"] > 0]
    passed = [r for r in tested if r["passed"]]
    converged = [r for r in tested if r.get("tpf_converged")]
    diverged = [r for r in tested if not r.get("tpf_converged") and r.get("nr_converged")]

    print(f"\n{'═'*90}")
    print(f"  STATISTIKEN")
    print(f"{'═'*90}")
    print(f"  Netze mit PV getestet:    {len(tested)}")
    print(f"  PASS:                     {len(passed)} ({100*len(passed)/max(len(tested),1):.0f}%)")
    print(f"  FAIL (konvergiert, ΔV):   {len([r for r in tested if r.get('tpf_converged') and not r['passed']])}")
    print(f"  FAIL (divergiert):        {len(diverged)}")

    if converged:
        etas = [r["eta"] for r in converged if r["eta"] < np.inf]
        rhos = [r["rho"] for r in converged if r["rho"] < np.inf]
        v_errors = [r["max_v_error"] for r in converged if r["max_v_error"] < np.inf]
        outer_iters = [r["tpf_outer_iter"] for r in converged if r["tpf_outer_iter"] > 0]

        if etas:
            print(f"\n  η (konvergierte):")
            print(f"    min={min(etas):.4f}  max={max(etas):.4f}  "
                  f"median={np.median(etas):.4f}")
        if rhos:
            print(f"  ρ(J_G) (konvergierte):")
            print(f"    min={min(rhos):.4f}  max={max(rhos):.4f}  "
                  f"median={np.median(rhos):.4f}")
        if v_errors:
            print(f"  max|ΔV| (konvergierte):")
            print(f"    min={min(v_errors):.2e}  max={max(v_errors):.2e}  "
                  f"median={np.median(v_errors):.2e}")
        if outer_iters:
            print(f"  Outer-Iterationen:")
            print(f"    min={min(outer_iters)}  max={max(outer_iters)}  "
                  f"median={np.median(outer_iters):.0f}")

    # ── ρ vs. Konvergenz Analyse ──
    print(f"\n  {'─'*60}")
    print(f"  ρ-ANALYSE (Spektralradius der äußeren Schleife):")
    print(f"  {'─'*60}")

    rho_conv = [r["rho"] for r in passed if r["rho"] < np.inf]
    rho_div = [r["rho"] for r in diverged if r["rho"] < np.inf]

    if rho_conv:
        print(f"    Konvergiert (PASS): ρ ∈ [{min(rho_conv):.4f}, {max(rho_conv):.4f}]")
        n_rho_lt1 = sum(1 for r in rho_conv if r < 1.0)
        print(f"      davon ρ < 1: {n_rho_lt1}/{len(rho_conv)}")
    if rho_div:
        print(f"    Divergiert:         ρ ∈ [{min(rho_div):.4f}, {max(rho_div):.4f}]")
        n_rho_gt1 = sum(1 for r in rho_div if r >= 1.0)
        print(f"      davon ρ ≥ 1: {n_rho_gt1}/{len(rho_div)}")

    # Korrelation ρ < 1 ↔ Konvergenz
    all_with_rho = [r for r in tested if r["rho"] < np.inf and r.get("nr_converged")]
    if all_with_rho:
        correct_prediction = sum(
            1 for r in all_with_rho
            if (r["rho"] < 1.0 and r["passed"]) or (r["rho"] >= 1.0 and not r["passed"])
        )
        print(f"\n    Vorhersagekraft von ρ < 1 ↔ Konvergenz:")
        print(f"      Korrekt: {correct_prediction}/{len(all_with_rho)} "
              f"({100*correct_prediction/len(all_with_rho):.0f}%)")

    print(f"{'═'*90}")


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
        print("  ⚠ Keine Daten zum Plotten vorhanden.")
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
        rho_str = f"ρ={rho:.2f}" if rho < 100 else "ρ=—"
        rx = rec.get("rx_ratio", np.nan)
        rx_str = f"R/X={rx:.1f}" if not np.isnan(rx) else "R/X=—"
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

    ax00.set_xlabel("Äußere Iteration ℓ", fontsize=11)
    ax00.set_ylabel("max ||V_PV| - V_spec|| [p.u.]", fontsize=11)
    ax00.set_title("(a) PV-Spannungsfehler vs. Outer-Iteration", fontsize=12)
    ax00.grid(True, which="both", alpha=0.3)
    ax00.set_xlim(1,100)
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
    #  (1,0) Solver-Zeit vs. Netzgröße — NUR konvergierte
    # ══════════════════════════════════════════════════════════════
    ax10 = axes[1, 0]

    if timing_data:
        sorted_by_size = sorted(timing_data, key=lambda r: r["n_bus"])
        n_bus_vals = np.array([r["n_bus"] for r in sorted_by_size])
        nr_times = np.array([r["nr_time_ms"] for r in sorted_by_size])
        tpf_times = np.array([r["tpf_time_ms"] for r in sorted_by_size])

        ax10.scatter(n_bus_vals, nr_times,
                     color="tab:red", marker="s", s=60, zorder=5, alpha=0.8)
        ax10.scatter(n_bus_vals, tpf_times,
                     color="tab:blue", marker="o", s=60, zorder=5, alpha=0.8)

        ax10.plot(n_bus_vals, nr_times,
                  color="tab:red", linestyle="--", linewidth=1.0, alpha=0.5)
        ax10.plot(n_bus_vals, tpf_times,
                  color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.5)

        ax10.set_yscale("log")
        ax10.set_xscale("log")

    ax10.set_xlabel("Netzgröße (Anzahl Busse im d-Block)", fontsize=11)
    ax10.set_ylabel("Rechenzeit [ms]", fontsize=11)
    ax10.set_title("(c) Rechenzeit vs. Netzgröße (nur konvergierte)", fontsize=12)
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

        ax11.set_yscale("log")

        # Colorbar für Netzgröße
        cbar = plt.colorbar(sc_tpf, ax=ax11, pad=0.02, fraction=0.04)
        cbar.set_label("Netzgröße (normiert)", fontsize=9)
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
        f"({n_conv} konv., {n_div} div.), ω = {omega}",
        fontsize=14, y=1.06,
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")

    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validierung + Spektralradius + Konvergenz-Plot: TPF Methode A"
    )
    parser.add_argument(
        "--suite", choices=["quick", "radial", "salazar", "salazar_scaling", "full"], default="salazar_scaling",
        help="Testsuite: quick (4 Netze), radial (ohne IEEE vermascht), full (alles)"
    )
    parser.add_argument("--omega", type=float, default=1.0,
                        help="Q-Relaxationsfaktor ω (default: 1.0)")
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
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║  VALIDIERUNG + SPEKTRALRADIUS + KONVERGENZ-PLOT: TPF Methode A            ║")
    print("║  Berechnet ρ(J_G) numerisch für jedes Netz                                ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")

    # Netze laden
    if args.suite == "quick":
        networks = get_quick_test_networks()
    elif args.suite == "radial":
        networks = get_radial_only_networks()
    elif args.suite == "salazar":
        networks = get_salazar_pv_networks()
    elif args.suite == "salazar_scaling":
        networks = get_salazar_scaling_networks()
    else:
        networks = get_comprehensive_networks()


    print(f"\n  Suite: '{args.suite}' — {len(networks)} Netze")
    print(f"  ω = {args.omega}, PASS-Schwelle = {args.tol:.0e}\n")

    # Handle --list
    if args.list:
        print(f"\n  Verfügbare Netzwerke in Suite '{args.suite}':")
        for i, name in enumerate(sorted(networks.keys()), 1):
            print(f"    {i:2}. {name}")
        print(f"\n{'═'*90}")
        return

    # Handle --network filter
    if args.network:
        if args.network not in networks:
            print(f"\n  FEHLER: Netzwerk '{args.network}' nicht gefunden in Suite '{args.suite}'")
            print(f"  Verwende --list um alle Netzwerke anzuzeigen.")
            print(f"\n{'═'*90}")
            return
        networks = {args.network: networks[args.network]}
        print(f"  → Nur Netzwerk '{args.network}' wird getestet.\n")

    # Validierung
    t_start = time.perf_counter()
    records = run_validation_suite(
        networks, omega=args.omega, tol_pass=args.tol, verbose=True
    )
    t_total = time.perf_counter() - t_start

    # Tabelle
    print_results_table(records, title=f"Methode A — Suite '{args.suite}', ω={args.omega}")
    print_statistics(records)

    # Gesamtergebnis
    tested = [r for r in records if r["n_pv"] > 0]
    n_pass = sum(1 for r in tested if r["passed"])
    n_total = len(tested)

    print(f"\n  Gesamtzeit: {t_total:.1f} s")
    print(f"  Gesamtergebnis: {n_pass}/{n_total} PASS "
          f"({100*n_pass/max(n_total,1):.0f}%)")

    if n_pass == n_total:
        print(f"\n  ✓ METHODE A VALIDIERT FÜR ALLE {n_total} TESTNETZE!")
    else:
        n_div = sum(1 for r in tested if not r.get("tpf_converged")
                    and r.get("nr_converged"))
        n_acc = sum(1 for r in tested if r.get("tpf_converged")
                    and not r["passed"])
        print(f"\n  ⚠ {n_total - n_pass} Tests fehlgeschlagen:")
        if n_div:
            print(f"    - {n_div} divergiert (ω anpassen oder η > 1)")
        if n_acc:
            print(f"    - {n_acc} Genauigkeit unzureichend")

    # Plot
    if not args.no_plot:
        save_path = args.save or f"convergence_method_a_{args.suite}_omega{args.omega}.png"
        plot_convergence(records, omega=args.omega, save_path=save_path)

    print(f"\n{'═'*90}")


if __name__ == "__main__":
    main()
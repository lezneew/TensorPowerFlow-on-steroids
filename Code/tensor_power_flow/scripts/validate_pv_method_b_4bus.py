# tensor_power_flow/scripts/validate_pv_method_b_4bus.py
"""
Validierung Methode B: 4-Bus Netz mit detaillierten Iterations-Plots
=====================================================================

Erzeugt Plots:
    1. |V| an jedem Bus vs. Iteration (mit NR-Referenz)
    2. Winkel an jedem Bus vs. Iteration
    3. Q am PV-Bus vs. Iteration
    4. Konvergenzkurve (semilogy)
    5. Leistungsbilanz (P, Q) an jedem Bus vs. Iteration
    6. Finaler Vergleich Method B vs. NR
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.solvers.tpf_pv_method_b import TPFDensePVMethodB
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.radial_network import create_4bus_1pv


# ══════════════════════════════════════════════════════════════════════
#  Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════

def compute_bus_powers(network, V_history):
    """
    Berechnet P und Q an jedem d-Block-Bus fuer jede Iteration.

    s_bus = -V * conj(Y_dd @ V + Y_ds @ v_s)   [Last-Konvention]

    Returns
    -------
    P_hist : (n_iter, bphi) Wirkleistung
    Q_hist : (n_iter, bphi) Blindleistung
    """
    Y_ds_vs = (network.Y_ds @ network.v_s).reshape(-1, 1)
    P_hist = []
    Q_hist = []

    for V in V_history:
        I = network.Y_dd @ V + Y_ds_vs       # (bphi, tau)
        s = -V * np.conj(I)                   # (bphi, tau)
        P_hist.append(s.real[:, 0])
        Q_hist.append(s.imag[:, 0])

    return np.array(P_hist), np.array(Q_hist)


def get_d_block_info(net, network):
    """Gibt d-Block-Indizes und Bus-Labels zurueck."""
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    pv_idx_ppc = np.where(bus_types == 2)[0]
    pq_idx_ppc = np.where(bus_types == 1)[0]
    d_idx = np.sort(np.concatenate([pq_idx_ppc, pv_idx_ppc]))

    # Labels
    labels = []
    for i, idx in enumerate(d_idx):
        btype = "PV" if idx in pv_idx_ppc else "PQ"
        labels.append(f"Bus {idx} ({btype})")

    return d_idx, labels


# ══════════════════════════════════════════════════════════════════════
#  Hauptvalidierung
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  VALIDIERUNG: Methode B (Eingebettete Q-Korrektur)")
    print("  Testnetz: 4-Bus, 1 PV, 2 PQ")
    print("=" * 70)

    # ── 1. Netz erstellen ──
    net = create_4bus_1pv()
    print("\n  [Netz] 4-Bus: Slack(0) -- PV(1, Vspec=1.03) -- PQ(2) -- PQ(3)")

    # ── 2. NR-Referenz ──
    nr_solver = PandapowerNRSolver(tol=1e-10, max_iter=100)
    nr_result = nr_solver.solve_from_net(net)
    assert nr_result.converged, "NR divergiert!"

    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    pv_idx_ppc = np.where(bus_types == 2)[0]
    pq_idx_ppc = np.where(bus_types == 1)[0]
    d_idx = np.sort(np.concatenate([pq_idx_ppc, pv_idx_ppc]))

    v_nr_all = nr_result.voltages
    v_nr_d = v_nr_all[d_idx]

    print(f"  [NR] Konvergiert in {nr_result.iterations} Iterationen")
    print(f"  [NR] |V| = {np.abs(v_nr_d)}")
    print(f"  [NR] angle = {np.angle(v_nr_d, deg=True)} deg")

    if nr_result.pv_q_pu is not None:
        print(f"  [NR] Q_PV = {nr_result.pv_q_pu} p.u.")

    # ── 3. Netz fuer TPF aufbauen ──
    network = build_network_from_pandapower(net, include_pv=True)
    bphi = network.n_bus_phases
    n_pv = network.n_pv
    pv_local = network.pv_indices

    print(f"\n  [Netz] d-Block: {bphi} Busse (PV={n_pv}, PQ={len(network.pq_indices)})")
    print(f"  [Netz] PV-Indizes (lokal): {pv_local}")
    print(f"  [Netz] V_spec = {network.pv_v_setpoint}")

    # ── 4. Methode B loesen (MIT Historie) ──
    solver_b = TPFDensePVMethodB(
        tol=1e-10,
        max_iter=200,
        tol_pv=1e-8,
        enforce_q_lims=False,
        track_convergence=True,
    )

    result_b = solver_b.solve(network)
    info = solver_b.pv_info

    print(f"\n  [Methode B] Konvergiert: {result_b.converged}")
    print(f"  [Methode B] Iterationen: {info.iterations}")
    print(f"  [Methode B] v_change final: {info.voltage_change_history[-1]:.2e}")
    print(f"  [Methode B] PV |V| Fehler: {info.pv_v_error_final:.2e}")
    print(f"  [Methode B] Zeit: {result_b.elapsed_time_s*1000:.2f} ms")

    # ── 5. Vergleich mit NR ──
    v_tpf = result_b.voltages.flatten()
    mag_err = np.abs(np.abs(v_tpf) - np.abs(v_nr_d))
    angle_err = np.abs(np.angle(v_tpf, deg=True) - np.angle(v_nr_d, deg=True))

    print(f"\n  [Vergleich B vs. NR]")
    print(f"  Max |V| Fehler:    {np.max(mag_err):.2e} p.u.")
    print(f"  Max Winkel-Fehler: {np.max(angle_err):.6f} deg")

    # PV Q-Vergleich
    q_b = info.pv_q_final
    q_nr = nr_result.pv_q_pu if nr_result.pv_q_pu is not None else np.zeros(n_pv)
    print(f"  Q_PV (Methode B): {q_b}")
    print(f"  Q_PV (NR):        {q_nr}")
    print(f"  Delta Q:          {np.abs(q_b - q_nr)}")

    # PASS/FAIL
    passed = np.max(mag_err) < 1e-5 and result_b.converged
    status = "PASS" if passed else "FAIL"
    print(f"\n  Ergebnis: {status}")

    # ── 6. Detaillierte Bus-Tabelle ──
    print(f"\n  {'Bus':<6} {'Typ':<5} {'|V| B':<10} {'|V| NR':<10} "
          f"{'dV':<10} {'angle B':<10} {'angle NR':<10}")
    print(f"  {'-'*65}")
    for i in range(bphi):
        btype = "PV" if i in pv_local else "PQ"
        v_b_mag = np.abs(v_tpf[i])
        v_nr_mag = np.abs(v_nr_d[i])
        a_b = np.angle(v_tpf[i], deg=True)
        a_nr = np.angle(v_nr_d[i], deg=True)
        dv = abs(v_b_mag - v_nr_mag)
        print(f"  {d_idx[i]:<6} {btype:<5} {v_b_mag:<10.6f} {v_nr_mag:<10.6f} "
              f"{dv:<10.2e} {a_b:<10.4f} {a_nr:<10.4f}")

    # ══════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════

    if not info.v_history:
        print("\n  [WARN] Keine Historie gespeichert, Plots uebersprungen.")
        return

    n_iter = len(info.v_history)
    iters = np.arange(1, n_iter + 1)

    # Extrahiere |V| und angle pro Iteration
    V_mag_hist = np.array([np.abs(V[:, 0]) for V in info.v_history])   # (n_iter, bphi)
    V_ang_hist = np.array([np.angle(V[:, 0], deg=True)
                           for V in info.v_history])                     # (n_iter, bphi)
    Q_pv_hist = np.array([q[:, 0] for q in info.q_history])            # (n_iter, n_pv)

    # Leistungen pro Iteration berechnen
    P_hist, Q_hist = compute_bus_powers(network, info.v_history)

    # NR-Referenzwerte
    v_nr_mag = np.abs(v_nr_d)
    v_nr_ang = np.angle(v_nr_d, deg=True)

    # Bus-Labels (needed for iteration table)
    d_labels = []
    for i in range(bphi):
        btype = "PV" if i in pv_local else "PQ"
        d_labels.append(f"Bus {d_idx[i]} ({btype})")

    # ══════════════════════════════════════════════════════════════════════
    #  ITERATIONSTABELLE (Console)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 140)
    print("  ITERATIONSTABELLE: Methode B vs. NR-Referenz")
    print("=" * 140)

    header = f" {'Iter':>4} |" + "".join([
        f" Bus{d_idx[i]:>2} {d_labels[i][-4:-1]:>4} |" for i in range(bphi)
    ])
    header += " Q_PV  | dV_max"
    print(header)
    print("-" * 140)

    for n in range(n_iter):
        row = f" {n+1:>4} |"
        for i in range(bphi):
            vm = V_mag_hist[n, i]
            va = V_ang_hist[n, i]
            p = P_hist[n, i]
            q = Q_hist[n, i]
            row += f" |V|{vm:.4f} {va:>6.2f} P{p:>6.3f} Q{q:>6.3f} |"
        row += f" {Q_pv_hist[n, 0]:>7.4f} | {info.voltage_change_history[n]:.2e}"
        print(row)

    print("-" * 140)
    print("NR REF:   |" + "".join([
        f" |V|{v_nr_mag[i]:.4f} {v_nr_ang[i]:>6.2f} -----.--- -----.--- |"
        for i in range(bphi)
    ]) + f" {q_nr[0]:>7.4f} | ------")
    print("=" * 140)

    # Farben
    colors = plt.cm.tab10(np.linspace(0, 1, max(bphi, 3)))

    # ──────────────────────────────────────────────────────────────
    #  Figure 1: Spannungen und Konvergenz (2x2)
    # ──────────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle(
        "Methode B: Iterationsverlauf -- 4-Bus Netz (1 PV, 2 PQ)",
        fontsize=14, fontweight="bold",
    )

    # (1,1) |V| vs. Iteration
    ax = axes1[0, 0]
    for i in range(bphi):
        ls = "-" if i in pv_local else "--"
        ax.plot(iters, V_mag_hist[:, i], color=colors[i], linewidth=1.5,
                linestyle=ls, label=d_labels[i])
        ax.axhline(v_nr_mag[i], color=colors[i], linewidth=0.8,
                   linestyle=":", alpha=0.7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("|V| [p.u.]")
    ax.set_title("Spannungsbetrag |V| pro Bus")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_iter)
    # Annotation
    ax.annotate("gestrichelt = NR-Referenz", xy=(0.98, 0.02),
                xycoords="axes fraction", ha="right", fontsize=8,
                color="gray", style="italic")

    # (1,2) Winkel vs. Iteration
    ax = axes1[0, 1]
    for i in range(bphi):
        ls = "-" if i in pv_local else "--"
        ax.plot(iters, V_ang_hist[:, i], color=colors[i], linewidth=1.5,
                linestyle=ls, label=d_labels[i])
        ax.axhline(v_nr_ang[i], color=colors[i], linewidth=0.8,
                   linestyle=":", alpha=0.7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Winkel [deg]")
    ax.set_title("Spannungswinkel pro Bus")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_iter)

    # (2,1) Q am PV-Bus vs. Iteration
    ax = axes1[1, 0]
    for j in range(n_pv):
        ax.plot(iters, Q_pv_hist[:, j], color="tab:blue", linewidth=2.0,
                label=f"Q_PV (Bus {d_idx[pv_local[j]]})")
        ax.axhline(q_nr[j], color="tab:red", linewidth=1.5, linestyle="--",
                   label=f"Q_NR = {q_nr[j]:.4f}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Q [p.u.]")
    ax.set_title("Blindleistung am PV-Knoten")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_iter)

    # (2,2) Konvergenz (semilogy)
    ax = axes1[1, 1]
    ax.semilogy(iters, info.voltage_change_history, "k-o", markersize=3,
                linewidth=1.5, label="max ||V_new| - |V_old||")
    ax.axhline(solver_b.tol, color="green", linestyle=":", linewidth=1.5,
               label=f"tol = {solver_b.tol:.0e}")
    ax.axhline(1e-6, color="orange", linestyle="-.", linewidth=1.0,
               label="1e-6")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Konvergenzfehler [p.u.]")
    ax.set_title("Konvergenzverlauf")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(1, n_iter)
    ax.set_ylim(bottom=1e-13, top=10)

    plt.tight_layout()
    fig1.savefig("method_b_4bus_convergence.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot gespeichert: method_b_4bus_convergence.png")

    # ──────────────────────────────────────────────────────────────
    #  Figure 2: Leistungen pro Bus (2x1)
    # ──────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 9))
    fig2.suptitle(
        "Methode B: Leistungsverlauf pro Bus -- 4-Bus Netz",
        fontsize=14, fontweight="bold",
    )

    # P pro Bus
    ax = axes2[0]
    for i in range(bphi):
        ls = "-" if i in pv_local else "--"
        ax.plot(iters, P_hist[:, i], color=colors[i], linewidth=1.5,
                linestyle=ls, label=d_labels[i])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("P [p.u.]")
    ax.set_title("Wirkleistung P an jedem Bus (aus Netzgleichung)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_iter)

    # Q pro Bus
    ax = axes2[1]
    for i in range(bphi):
        ls = "-" if i in pv_local else "--"
        ax.plot(iters, Q_hist[:, i], color=colors[i], linewidth=1.5,
                linestyle=ls, label=d_labels[i])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Q [p.u.]")
    ax.set_title("Blindleistung Q an jedem Bus (aus Netzgleichung)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n_iter)

    plt.tight_layout()
    fig2.savefig("method_b_4bus_powers.png", dpi=150, bbox_inches="tight")
    print(f"  Plot gespeichert: method_b_4bus_powers.png")

    # ──────────────────────────────────────────────────────────────
    #  Figure 3: Finaler Vergleich (Bar-Chart)
    # ──────────────────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle(
        "Methode B vs. NR: Finaler Zustand -- 4-Bus Netz",
        fontsize=13, fontweight="bold",
    )

    x_pos = np.arange(bphi)
    width = 0.35

    # |V| Vergleich
    ax = axes3[0]
    v_b_final = np.abs(v_tpf)
    bars1 = ax.bar(x_pos - width/2, v_b_final, width, label="Methode B",
                   color="steelblue", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x_pos + width/2, v_nr_mag, width, label="NR",
                   color="coral", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Bus {d_idx[i]}" for i in range(bphi)], fontsize=9)
    ax.set_ylabel("|V| [p.u.]")
    ax.set_title("Spannungsbetrag")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Angle Vergleich
    ax = axes3[1]
    a_b_final = np.angle(v_tpf, deg=True)
    ax.bar(x_pos - width/2, a_b_final, width, label="Methode B",
           color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x_pos + width/2, v_nr_ang, width, label="NR",
           color="coral", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Bus {d_idx[i]}" for i in range(bphi)], fontsize=9)
    ax.set_ylabel("Winkel [deg]")
    ax.set_title("Spannungswinkel")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Fehler
    ax = axes3[2]
    ax.bar(x_pos, mag_err, color="darkred", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Bus {d_idx[i]}" for i in range(bphi)], fontsize=9)
    ax.set_ylabel("|V|-Fehler [p.u.]")
    ax.set_title("Abweichung |V_B| - |V_NR|")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(1e-6, color="green", linestyle=":", label="1e-6")
    ax.legend()

    plt.tight_layout()
    fig3.savefig("method_b_4bus_final_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Plot gespeichert: method_b_4bus_final_comparison.png")

    plt.show()
    print(f"\n{'='*70}")
    print(f"  VALIDIERUNG ABGESCHLOSSEN: {status}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
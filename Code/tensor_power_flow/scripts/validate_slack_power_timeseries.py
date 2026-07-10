# tensor_power_flow/scripts/validate_slack_power_timeseries.py
"""
Validierung TPF Methode A vs. pandapower NR im Zeitreihenbetrieb.
Vergleicht:
  - Spannungen (bφ, τ)
  - Slack-Wirk- und Blindleistung (1, τ)
Zusätzlich: Konvergenzrate, Performance, Plot über den ersten Tag.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from tpf.builders.from_pandapower import (
    build_network_from_pandapower, build_s_batch_timeseries
)
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.network_generator_salazar import create_salazar_network
from tpf.generators.profile_generators import (
    generate_pv_profile, generate_load_profile
)


def validate_case(nodes: int, n_pv: int, tau: int,
                  seed: int = 42, omega: float = 1.0,
                  tol_slack: float = 1e-5, tol_v: float = 1e-5):
    print(f"\n{'='*74}")
    print(f"  Netz: {nodes}-Bus, {n_pv} PV | τ = {tau:,}")
    print(f"{'='*74}")

    # 1. Netz + Zeitreihen-Profile
    net = create_salazar_network(nodes=nodes, n_pv=n_pv, seed=seed)
    network = build_network_from_pandapower(net, include_pv=True)

    p_load, q_load = generate_load_profile(
        net, tau, "daily_double_peak", seed=seed + 1
    )
    p_pv = None
    if network.has_pv:
        p_pv = generate_pv_profile(
            network.n_pv, tau, "daily_cosine",
            capacity_factor=0.25,
            p_nom_mw=net.gen["p_mw"].values,
            seed=seed + 2,
        )
    s_batch = build_s_batch_timeseries(network, net, p_load, q_load, p_pv)

    # 2. TPF Zeitreihe
    tpf = TPFDensePVMethodA(tol=1e-11, tol_pv=1e-9, omega=omega,
                            max_iter_inner=100, max_iter_outer=50)
    t0 = time.perf_counter()
    res_tpf = tpf.solve_timeseries(network, s_batch, verbose=False)
    t_tpf = time.perf_counter() - t0

    # 3. NR Zeitreihe
    nr = PandapowerNRSolver(tol=1e-10, max_iter=100)
    t0 = time.perf_counter()
    res_nr = nr.solve_timeseries(net, p_load, q_load, p_pv, verbose=False)
    t_nr = time.perf_counter() - t0

    # 4. Konvergenz-Check (Szenarien beide-konvergent)
    conv_tpf = (
        tpf.pv_info.converged_per_scenario
        if (tpf.pv_info is not None
            and tpf.pv_info.converged_per_scenario is not None)
        else np.full(tau, res_tpf.converged, dtype=bool)
    )
    # NR: aggregiert; per-Szenario nicht getrackt → wir setzen True wenn Gesamt-True
    conv_nr = np.full(tau, res_nr.converged, dtype=bool)
    mask = conv_tpf & conv_nr

    n_both = int(np.sum(mask))
    if n_both == 0:
        print(f"  ✗ Keine gemeinsam konvergierten Szenarien.")
        return None

    # 5. Slack-Leistung
    dp = np.abs(res_tpf.s_slack[0, :].real - res_nr.s_slack[0, :].real)
    dq = np.abs(res_tpf.s_slack[0, :].imag - res_nr.s_slack[0, :].imag)
    p_err_max, p_err_mean = float(dp[mask].max()), float(dp[mask].mean())
    q_err_max, q_err_mean = float(dq[mask].max()), float(dq[mask].mean())

    # 6. Spannungen (d-Block bei NR extrahieren)
    ppc = net._ppc
    bt = ppc["bus"][:, 1].astype(int)
    d_idx = np.sort(np.concatenate([np.where(bt == 1)[0], np.where(bt == 2)[0]]))
    v_tpf = np.abs(res_tpf.voltages)
    v_nr  = np.abs(res_nr.voltages[d_idx, :])
    dv = np.abs(v_tpf - v_nr)
    dv_max = float(dv[:, mask].max())
    dv_mean = float(dv[:, mask].mean())

    # 7. Ausgabe
    print(f"\n  Konvergenz:")
    print(f"    TPF: {int(conv_tpf.sum())}/{tau} ({100*conv_tpf.mean():.1f}%)")
    print(f"    NR:  {int(conv_nr.sum())}/{tau} ({100*conv_nr.mean():.1f}%)")
    print(f"    Vergleichsbasis (beide konv.): {n_both}/{tau}")

    print(f"\n  Slack-Leistung:")
    print(f"    max|ΔP| = {p_err_max:.2e}   mean = {p_err_mean:.2e}")
    print(f"    max|ΔQ| = {q_err_max:.2e}   mean = {q_err_mean:.2e}")

    print(f"\n  Spannungen (alle Busse × τ):")
    print(f"    max|Δ|V|| = {dv_max:.2e}   mean = {dv_mean:.2e}")

    print(f"\n  Performance:")
    print(f"    TPF: {t_tpf*1000:>9.1f} ms  ({t_tpf*1000/tau:.3f} ms/PF)")
    print(f"    NR:  {t_nr*1000:>9.1f} ms  ({t_nr*1000/tau:.3f} ms/PF)")
    print(f"    Speedup: {t_nr/t_tpf:.1f}x")

    passed = (p_err_max < tol_slack and q_err_max < tol_slack and dv_max < tol_v)
    print(f"\n  {'✓ PASS' if passed else '✗ FAIL'} "
          f"(tol_slack={tol_slack:.0e}, tol_v={tol_v:.0e})")

    return {
        "tau": tau, "nodes": nodes, "n_pv": n_pv,
        "p_err_max": p_err_max, "q_err_max": q_err_max,
        "dv_max": dv_max, "t_tpf": t_tpf, "t_nr": t_nr,
        "passed": passed, "n_both": n_both,
        "p_tpf": res_tpf.s_slack[0, :].real,
        "q_tpf": res_tpf.s_slack[0, :].imag,
        "p_nr":  res_nr.s_slack[0, :].real,
        "q_nr":  res_nr.s_slack[0, :].imag,
        "dv_per_t": np.max(dv, axis=0),
    }


def plot_slack_comparison(r: dict, save_path: str = None):
    """Plot: P_slack, Q_slack, ΔP/ΔQ, ΔV — ersten Tag (bis 1440 min) zeigen."""
    n_show = min(r["tau"], 1440)
    t = np.arange(n_show)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")

    # (a) P_slack
    ax = axes[0, 0]
    ax.plot(t, r["p_nr"][:n_show], "r-", lw=2.0, alpha=0.85, label="NR")
    ax.plot(t, r["p_tpf"][:n_show], "b--", lw=1.0, label="TPF")
    ax.axhline(y=0, color="gray", lw=0.6)
    ax.set_xlabel("Zeitschritt [min]"); ax.set_ylabel("P_slack [p.u.]")
    ax.set_title("(a) Wirkleistung Slack")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (b) Q_slack
    ax = axes[0, 1]
    ax.plot(t, r["q_nr"][:n_show], "r-", lw=2.0, alpha=0.85, label="NR")
    ax.plot(t, r["q_tpf"][:n_show], "b--", lw=1.0, label="TPF")
    ax.axhline(y=0, color="gray", lw=0.6)
    ax.set_xlabel("Zeitschritt [min]"); ax.set_ylabel("Q_slack [p.u.]")
    ax.set_title("(b) Blindleistung Slack")
    ax.legend(); ax.grid(True, alpha=0.3)

    # (c) Slack-Fehler
    ax = axes[1, 0]
    ax.semilogy(t, np.abs(r["p_tpf"] - r["p_nr"])[:n_show], "-", label="|ΔP|")
    ax.semilogy(t, np.abs(r["q_tpf"] - r["q_nr"])[:n_show], "-", label="|ΔQ|")
    ax.axhline(y=1e-6, color="gray", ls=":", lw=1.0, label="1e-6")
    ax.set_xlabel("Zeitschritt [min]"); ax.set_ylabel("Fehler [p.u.]")
    ax.set_title("(c) Slack-Fehler TPF vs NR")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    # (d) Spannungsfehler
    ax = axes[1, 1]
    ax.semilogy(t, r["dv_per_t"][:n_show], "-", color="tab:purple")
    ax.axhline(y=1e-6, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("Zeitschritt [min]")
    ax.set_ylabel("max_bus |Δ|V|| [p.u.]")
    ax.set_title("(d) Spannungsfehler TPF vs NR")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"TPF vs. NR Zeitreihe — {r['nodes']}-Bus, {r['n_pv']} PV, "
        f"τ={r['tau']} (angezeigt: erste {n_show} Zeitschritte)",
        fontsize=12,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot gespeichert: {save_path}")
    plt.show()


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  VALIDIERUNG: TPF vs. NR — Spannungen und Slack-Leistung (τ-Reihe)  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    results = []
    # Schnell: Genauigkeitscheck
    results.append(validate_case(nodes=100, n_pv=5, tau=100))
    # Ein Tag @ 1 min — schön für Plot
    results.append(validate_case(nodes=100, n_pv=5, tau=1440))

    # Summary
    print(f"\n{'='*74}")
    print(f"  ZUSAMMENFASSUNG")
    print(f"{'='*74}")
    print(f"  {'τ':>7} {'max|ΔP|':>12} {'max|ΔQ|':>12} {'max|Δ|V||':>12} "
          f"{'Speedup':>10} {'Status':>8}")
    for r in results:
        if r is None: continue
        print(f"  {r['tau']:>7} {r['p_err_max']:>12.2e} {r['q_err_max']:>12.2e} "
              f"{r['dv_max']:>12.2e} {r['t_nr']/r['t_tpf']:>9.1f}x "
              f"{'PASS' if r['passed'] else 'FAIL':>8}")

    # Plot des größeren Falls
    if len(results) >= 2 and results[1] is not None:
        plot_slack_comparison(
            results[1], save_path="validate_slack_power_timeseries.png"
        )


if __name__ == "__main__":
    main()
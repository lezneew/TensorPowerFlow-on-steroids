# tensor_power_flow/scripts/validate_pv_method_a_timeseries.py
"""
Validierung + Benchmark: TPF Methode A für Zeitreihen (τ parallel).
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


def run_case(nodes, n_pv, tau, run_nr_ref=True, omega=1.0, chunk=None):
    print(f"\n{'='*70}")
    print(f"  Netz: {nodes} Bus, {n_pv} PV | τ = {tau:,}")
    print(f"{'='*70}")

    net = create_salazar_network(nodes=nodes, n_pv=n_pv, seed=42)
    network = build_network_from_pandapower(net, include_pv=True)

    p_load, q_load = generate_load_profile(net, tau, "daily_double_peak", seed=1)
    p_pv = generate_pv_profile(
        network.n_pv, tau, "daily_cosine",
        capacity_factor=0.25,
        p_nom_mw=net.gen["p_mw"].values if network.has_pv else 0.0,
        seed=2,
    ) if network.has_pv else None

    s_batch = build_s_batch_timeseries(network, net, p_load, q_load, p_pv)

    # TPF
    solver = TPFDensePVMethodA(
        tol=1e-8, max_iter_inner=50, max_iter_outer=30,
        tol_pv=1e-6, omega=omega, enforce_q_lims=False,
    )
    t0 = time.perf_counter()
    tpf_res = solver.solve_timeseries(network, s_batch, chunk_size=chunk, verbose=True)
    t_tpf = time.perf_counter() - t0

    info = solver.pv_info
    conv_rate = 100 * info.n_converged_scenarios / tau
    print(f"\n  TPF: {t_tpf:.2f} s | {t_tpf*1000/tau:.3f} ms/PF | "
          f"conv {info.n_converged_scenarios}/{tau} ({conv_rate:.1f}%)")

    # NR-Referenz (nur für kleine τ)
    if run_nr_ref and tau <= 5000:
        nr = PandapowerNRSolver(tol=1e-8, max_iter=50)
        t0 = time.perf_counter()
        nr_res = nr.solve_timeseries(net, p_load, q_load, p_pv, verbose=False)
        t_nr = time.perf_counter() - t0

        # Vergleich: TPF's d-Block vs. NR's alle Busse (an d-Indizes)
        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        d_idx = np.sort(np.concatenate([
            np.where(bus_types == 1)[0],
            np.where(bus_types == 2)[0],
        ]))
        v_tpf = tpf_res.voltages
        v_nr_d = nr_res.voltages[d_idx, :]
        max_dv = np.max(np.abs(np.abs(v_tpf) - np.abs(v_nr_d)))

        print(f"  NR:  {t_nr:.2f} s | {t_nr*1000/tau:.3f} ms/PF")
        print(f"  Speedup TPF vs. NR: {t_nr/t_tpf:.1f}×")
        print(f"  max |ΔV|: {max_dv:.2e}")
        return {"t_tpf": t_tpf, "t_nr": t_nr, "max_dv": max_dv,
                "conv_rate": conv_rate, "tau": tau, "nodes": nodes}
    return {"t_tpf": t_tpf, "t_nr": None, "max_dv": None,
            "conv_rate": conv_rate, "tau": tau, "nodes": nodes}


def main():
    # Schnelltest: Korrektheit
    print("=== KORREKTHEIT ===")
    run_case(nodes=100, n_pv=5, tau=100)

    # Skalierung
    print("\n=== SKALIERUNG τ ===")
    results = []
    for tau in [1, 100, 1000, 10_000, 100_000]:
        r = run_case(nodes=100, n_pv=5, tau=tau,
                     run_nr_ref=(tau <= 1000))
        results.append(r)

    # Paper-Reproduktion
    print("\n=== PAPER Tab. I: 100 Bus × 525600 τ ===")
    run_case(nodes=100, n_pv=5, tau=525_600, run_nr_ref=False)


if __name__ == "__main__":
    main()
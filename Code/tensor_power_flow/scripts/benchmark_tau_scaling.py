"""
TPF (Methode A, Batch) vs. Newton-Raphson: tau-Skalierung.

Erfasst pro (Netz, tau, Variante, Wiederholung):
  Zeiten (getrennt t0/tm), Iterationszahlen (max/mean/median/p95/sum),
  Straggler-Metriken, Konvergenzquote, Genauigkeit vs. NR, Betriebspunkt,
  Speicher-/GFLOP-Kennzahlen, Chunk-Konfiguration, X_pp-Diagnostik.

Modi:
  (default)        tau-Sweep
  --chunk-sweep    chunk_size-Sweep bei festem tau
  --analyze DIR    Fit t = t0 + tau*tm, tau*, S_inf aus vorhandenen CSVs
"""
import sys, os, time, json, csv, platform, argparse
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tpf.builders.from_pandapower import (
    build_network_from_pandapower, build_s_batch_timeseries)
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver
from tpf.generators.network_generator_salazar import (
    get_salazar_scaling_networks, get_salazar_low_rx05_networks,
    get_salazar_low_rx10_networks, get_salazar_low_vm_networks)
from tpf.generators.profile_generators import (
    generate_pv_profile, generate_load_profile)

ALL_NETWORKS = {**get_salazar_scaling_networks(),
                **get_salazar_low_rx05_networks(),
                **get_salazar_low_rx10_networks(),
                **get_salazar_low_vm_networks()}

VARIANTS = {
    "base":        (dict(cold_start=False, adaptive_inner=False), dict(warm_mode="flat")),
    "cold":        (dict(cold_start=True,  adaptive_inner=False), dict(warm_mode="flat")),
    "adapt":       (dict(cold_start=False, adaptive_inner=True),  dict(warm_mode="flat")),
    "carry":       (dict(cold_start=False, adaptive_inner=False), dict(warm_mode="carry_last")),
    "warmbatch":   (dict(cold_start=False, adaptive_inner=False), dict(warm_mode="provided")),
    "adapt_carry": (dict(cold_start=False, adaptive_inner=True),  dict(warm_mode="carry_last")),
}

SOLVER_BASE = dict(tol=1e-11, tol_pv=1e-9, omega=1.0,
                   max_iter_inner=100, max_iter_outer=50)


def hw_info() -> dict:
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        ncpu = psutil.cpu_count(logical=False)
    except Exception:
        ram, ncpu = float("nan"), os.cpu_count()
    blas = ""
    try:
        blas = json.dumps([d.get("internal_api", "") + ":" + str(d.get("num_threads", ""))
                           for d in np.__config__.__dict__.get("_built_with", []) or []])
    except Exception:
        pass
    try:
        import threadpoolctl
        blas = json.dumps([{k: d[k] for k in ("internal_api", "num_threads")}
                           for d in threadpoolctl.threadpool_info()])
    except Exception:
        pass
    return dict(host=platform.node(), cpu=platform.processor(),
                py=platform.python_version(), numpy=np.__version__,
                n_cores=ncpu, ram_gb=round(ram, 1), blas=blas,
                omp=os.environ.get("OMP_NUM_THREADS", ""))


def build_case(network_name: str, tau: int, seed: int = 42):
    d = ALL_NETWORKS[network_name]
    net = d["constructor"]()
    network = build_network_from_pandapower(net, include_pv=True)
    p_load, q_load = generate_load_profile(net, tau, "daily_double_peak", seed=seed + 1)
    p_pv = None
    if network.has_pv:
        p_pv = generate_pv_profile(network.n_pv, tau, "daily_cosine",
                                   capacity_factor=0.25,
                                   p_nom_mw=net.gen["p_mw"].values, seed=seed + 2)
    s_batch = build_s_batch_timeseries(network, net, p_load, q_load, p_pv)
    meta = {
        "description": d.get("description", ""),
        "category": d.get("category", ""),
        "pv_ratio": d.get("pv_ratio", 0.0),
        "pv_vm_offset_pu": d.get("pv_vm_offset_pu", 0.0),
    }
    return net, network, s_batch, (p_load, q_load, p_pv), meta


def d_bus_indices(net, network) -> np.ndarray:
    for attr in ("d_indices", "bus_indices_d", "d_bus_indices",
                 "non_slack_indices", "idx_d"):
        if hasattr(network, attr):
            v = getattr(network, attr)
            if v is not None:
                return np.asarray(v).ravel()
    slack = set(np.atleast_1d(net.ext_grid.bus.values).tolist())
    return np.array([b for b in net.bus.index.values if b not in slack])


def accuracy_vs_nr(V_tpf, V_nr_all, idx_d):
    try:
        V_ref = V_nr_all[idx_d, :]
        dv = float(np.max(np.abs(np.abs(V_tpf) - np.abs(V_ref))))
        dth = float(np.max(np.abs(np.angle(V_tpf, deg=True)
                                  - np.angle(V_ref, deg=True))))
        return dv, dth
    except Exception:
        return float("nan"), float("nan")


def q(a, p):
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, p)) if a.size else float("nan")


def run_tpf(network, s_batch, variant: str, chunk_size, repeats: int,
            warmup: bool, V0=None, q0=None):
    skw, ckw = VARIANTS[variant]
    solver = TPFDensePVMethodA(**SOLVER_BASE, **skw)

    call = dict(chunk_size=chunk_size, verbose=False, diagnostics=True,
                collect_scenario_stats=True, **ckw)
    if ckw.get("warm_mode") == "provided":
        call.update(V_init=V0, q_init=q0)

    if warmup:
        solver.solve_timeseries(network, s_batch[:, :min(8, s_batch.shape[1])],
                                chunk_size=None, verbose=False, diagnostics=False)

    times, infos, res = [], [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        res = solver.solve_timeseries(network, s_batch, **call)
        times.append(time.perf_counter() - t0)
        infos.append(solver.pv_info)
    k = int(np.argsort(times)[len(times) // 2])
    return np.array(times), infos[k], res


def run_nr(net, profiles, repeats: int, probe_overhead: bool, max_tau: int | None):
    p_load, q_load, p_pv = profiles
    tau = p_load.shape[1]
    nr = PandapowerNRSolver(tol=1e-10, max_iter=100)

    t_ov = nr.measure_overhead_ms(net) if probe_overhead else float("nan")

    extrapolated = False
    tau_run = tau
    if max_tau is not None and tau > max_tau:
        tau_run, extrapolated = max_tau, True
    pl, ql = p_load[:, :tau_run], q_load[:, :tau_run]
    pv = p_pv[:, :tau_run] if p_pv is not None else None

    times, res = [], None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        res = nr.solve_timeseries(net, pl, ql, pv, verbose=False)
        times.append(time.perf_counter() - t0)
    t = float(np.median(times))
    scale = tau / tau_run
    it = getattr(nr, "last_iters_per_scenario", np.array([-1]))
    cv = getattr(nr, "last_converged_per_scenario", np.array([res.converged]))
    return dict(
        t_nr_s=t * scale,
        t_nr_solve_only_s=getattr(nr, "last_solve_time_only_s", float("nan")) * scale,
        t_nr_overhead_ms=t_ov,
        t_nr_solve_est_ms=(t / tau_run * 1e3 - t_ov) if probe_overhead else float("nan"),
        nr_extrapolated=extrapolated, nr_tau_measured=tau_run,
        nr_iter_mean=float(np.mean(it[it > 0])) if np.any(it > 0) else float("nan"),
        nr_iter_max=int(np.max(it)) if it.size else -1,
        nr_conv_frac=float(np.mean(cv)),
    ), res


def one_config(network_name, tau, variant, args, cache_bytes):
    net, network, s_batch, profiles, meta = build_case(network_name, tau, args.seed)
    bphi = network.n_bus_phases

    V0 = q0 = None
    t_pre_batch_ms = 0.0
    if variant == "warmbatch":
        s_mean = s_batch.mean(axis=1).reshape(-1, 1)
        pre = TPFDensePVMethodA(**SOLVER_BASE)
        t0 = time.perf_counter()
        r0 = pre.solve_timeseries(network, s_mean, chunk_size=1, diagnostics=False)
        t_pre_batch_ms = (time.perf_counter() - t0) * 1e3
        V0 = r0.voltages[:, 0]
        q0 = (pre.pv_info.pv_q_final[:, 0] if network.has_pv else None)

    times, info, res = run_tpf(network, s_batch, variant, args.chunk_size,
                               args.repeats, not args.no_warmup, V0, q0)

    nr_stats, res_nr = (dict(), None)
    dv = dth = float("nan")
    if not args.no_nr:
        nr_stats, res_nr = run_nr(net, profiles, args.nr_repeats,
                                  not args.no_overhead_probe, args.nr_max_tau)
        if not nr_stats["nr_extrapolated"]:
            dv, dth = accuracy_vs_nr(res.voltages, res_nr.voltages,
                                     d_bus_indices(net, network))

    ko = np.asarray(info.outer_iterations_per_scenario, dtype=float)
    ki = np.asarray(info.inner_iterations_per_scenario, dtype=float)
    cv = np.asarray(info.converged_per_scenario, dtype=bool)
    t_tpf = float(np.median(times))
    hot = 6 * bphi * info.chunk_size * 16 + 16 * bphi * bphi

    row = dict(
        network=network_name,
        net_description=meta["description"],
        net_category=meta["category"],
        net_pv_ratio=meta["pv_ratio"],
        net_pv_vm_offset_pu=meta["pv_vm_offset_pu"],
        n_bus=int(len(net.bus)), n_bphi=int(bphi),
        n_pv=int(network.n_pv), pv_share=round(network.n_pv / max(1, len(net.bus)), 4),
        tau=tau, variant=variant, repeats=int(len(times)), seed=args.seed,
        t_tpf_s=t_tpf, t_tpf_min_s=float(times.min()), t_tpf_max_s=float(times.max()),
        t_tpf_std_s=float(times.std()),
        tpf_per_scenario_ms=t_tpf / tau * 1e3,
        t_precompute_ms=info.t_precompute_ms, t_solve_ms=info.t_solve_ms,
        t_prebatch_ms=t_pre_batch_ms,
        t_chunk_median_ms=float(np.median(info.t_chunks_ms)) if info.t_chunks_ms else 0.0,
        k_out_max=int(np.max(ko)), k_out_mean=float(np.mean(ko)),
        k_out_med=float(np.median(ko)), k_out_p95=q(ko, 95), k_out_min=int(np.min(ko)),
        k_in_max=int(np.max(ki)), k_in_mean=float(np.mean(ki)),
        k_in_med=float(np.median(ki)), k_in_p95=q(ki, 95), k_in_sum=int(np.sum(ki)),
        batch_eff_outer=float(np.mean(ko) / max(1.0, np.max(ko))),
        batch_eff_inner=float(np.mean(ki) / max(1.0, np.max(ki))),
        n_inner_calls=info.n_inner_calls, n_gemm=info.n_gemm,
        work_executed_cols=info.work_executed_cols,
        work_useful_cols=info.work_useful_cols,
        work_waste_frac=1.0 - info.work_useful_cols / max(1, info.work_executed_cols),
        active_per_outer=json.dumps(info.active_per_outer),
        n_conv=int(cv.sum()), conv_frac=float(cv.mean()),
        tpf_converged=bool(cv.all()),
        pv_v_err_max=float(info.pv_v_error_final),
        dv_max_vs_nr=dv, dtheta_max_deg_vs_nr=dth,
        vmin_min=float(np.nanmin(info.v_min_per_scenario)),
        vmin_med=float(np.nanmedian(info.v_min_per_scenario)),
        vmax_max=float(np.nanmax(info.v_max_per_scenario)),
        q_pv_absmax=float(np.max(np.abs(info.pv_q_final))) if network.has_pv else 0.0,
        cond_Xpp=info.cond_Xpp, rho_jacobi=info.rho_jacobi,
        min_diag_off=info.min_diag_off,
        chunk_size=info.chunk_size, n_chunks=info.n_chunks,
        hot_bytes=int(hot), hot_over_cache=hot / cache_bytes,
        gflops_eff=info.flops_gemm / max(1e-12, info.t_solve_ms * 1e-3) / 1e9,
        warm_mode=info.warm_mode,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    row.update(nr_stats)
    if "t_nr_s" in row:
        row["nr_per_scenario_ms"] = row["t_nr_s"] / tau * 1e3
        row["speedup"] = row["t_nr_s"] / t_tpf if t_tpf > 0 else float("nan")
        if np.isfinite(row.get("t_nr_solve_est_ms", np.nan)):
            row["speedup_fair"] = (row["t_nr_solve_est_ms"] * tau * 1e-3) / t_tpf

    scen = None
    if args.dump_scenarios:
        step = max(1, tau // args.dump_scenarios)
        sel = np.arange(0, tau, step)
        scen = dict(idx=sel, k_out=ko[sel].astype(int), k_in=ki[sel].astype(int),
                    conv=cv[sel], v_min=info.v_min_per_scenario[sel],
                    v_max=info.v_max_per_scenario[sel],
                    pv_err=info.pv_v_error_per_scenario[sel])
    return row, scen


def append_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    keys = sorted({k for r in rows for k in r})
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  -> {len(rows)} Zeilen -> {path}")


def append_scenarios(path, network_name, tau, variant, scen):
    if scen is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["network", "tau", "variant", "scenario", "k_out",
                        "k_in", "converged", "v_min", "v_max", "pv_err"])
        for j, i in enumerate(scen["idx"]):
            w.writerow([network_name, tau, variant, int(i), int(scen["k_out"][j]),
                        int(scen["k_in"][j]), bool(scen["conv"][j]),
                        scen["v_min"][j], scen["v_max"][j], scen["pv_err"][j]])


def analyze(runs_csv, out_csv):
    import collections
    rows = list(csv.DictReader(open(runs_csv)))
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("tpf_converged", "True") != "True":
            continue
        g[(r["network"], r["variant"])].append(r)

    out = []
    for (netn, var), rs in sorted(g.items()):
        tau = np.array([float(r["tau"]) for r in rs])
        t = np.array([float(r["t_tpf_s"]) for r in rs])
        m = tau >= 10
        if m.sum() < 3:
            continue
        A = np.vstack([np.ones(m.sum()), tau[m]]).T
        (t0, tm), *_ = np.linalg.lstsq(A, t[m], rcond=None)
        pred = A @ np.array([t0, tm])
        r2 = 1 - np.sum((t[m] - pred) ** 2) / np.sum((t[m] - t[m].mean()) ** 2)
        nrp = np.array([float(r.get("nr_per_scenario_ms", "nan")) for r in rs])
        tm_nr = float(np.nanmedian(nrp)) * 1e-3
        s_inf = tm_nr / tm if tm > 0 else float("nan")
        tau_star = t0 / (tm_nr - tm) if tm_nr > tm else float("inf")
        out.append(dict(network=netn, variant=var, n_bus=rs[0]["n_bus"],
                        n_pv=rs[0]["n_pv"], t0_ms=t0 * 1e3, tm_ms=tm * 1e3,
                        tm_nr_ms=tm_nr * 1e3, r2=r2, S_inf=s_inf,
                        tau_star=tau_star, throughput_per_s=1.0 / tm if tm > 0 else 0,
                        t_year_35040_s=t0 + 35040 * tm))
    append_csv(out_csv, out)
    print(f"\n{'network':<16}{'var':<12}{'t0[ms]':>10}{'tm[ms]':>10}"
          f"{'S_inf':>9}{'tau*':>9}{'Jahr[s]':>10}")
    for r in out:
        print(f"{r['network']:<16}{r['variant']:<12}{r['t0_ms']:>10.2f}"
              f"{r['tm_ms']:>10.4f}{r['S_inf']:>9.1f}{r['tau_star']:>9.1f}"
              f"{r['t_year_35040_s']:>10.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", "-n", default="sz_40_r010")
    p.add_argument("--filter", "-f", default=None)
    p.add_argument("--tau", "-t", default="1,10,50,100,500,1000,5000,10000,50000,100000")
    p.add_argument("--variants", "-v", default="base")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--nr-repeats", type=int, default=1)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--no-nr", action="store_true")
    p.add_argument("--no-overhead-probe", action="store_true")
    p.add_argument("--nr-max-tau", type=int, default=5000)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--chunk-sweep", default=None)
    p.add_argument("--dump-scenarios", type=int, default=0)
    p.add_argument("--cache-mb", type=float, default=32.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", "-d", default="tau_benchmark_results")
    p.add_argument("--analyze", default=None)
    a = p.parse_args()

    if a.analyze:
        analyze(a.analyze, a.analyze.replace(".csv", "_fits.csv"))
        return
    if a.nr_max_tau == 0:
        a.nr_max_tau = None

    nets = ([n for n in ALL_NETWORKS if n.startswith(a.filter)] if a.filter
            else [s.strip() for s in a.networks.split(",")])
    taus = [int(s) for s in a.tau.split(",")]
    variants = [s.strip() for s in a.variants.split(",")]
    cache = a.cache_mb * 1e6

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs = os.path.join(a.out_dir, f"runs_{stamp}.csv")
    scen = os.path.join(a.out_dir, f"scenarios_{stamp}.csv")
    with open(os.path.join(a.out_dir, f"env_{stamp}.json"), "w") \
            if os.makedirs(a.out_dir, exist_ok=True) is None else open(os.devnull, "w") as f:
        json.dump(dict(hw=hw_info(), args=vars(a)), f, indent=2)

    if a.chunk_sweep:
        tau = taus[0]
        for netn in nets:
            for cs in [int(s) for s in a.chunk_sweep.split(",")]:
                a.chunk_size = cs
                print(f"[{netn}] tau={tau} chunk={cs}")
                r, sc = one_config(netn, tau, variants[0], a, cache)
                append_csv(runs, [r]); append_scenarios(scen, netn, tau, variants[0], sc)
        analyze(runs, runs.replace(".csv", "_fits.csv"))
        return

    for netn in nets:
        for var in variants:
            for tau in taus:
                print(f"[{netn}] variant={var} tau={tau:,}", flush=True)
                try:
                    r, sc = one_config(netn, tau, var, a, cache)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue
                append_csv(runs, [r])
                append_scenarios(scen, netn, tau, var, sc)
                print(f"  t_TPF={r['t_tpf_s']:.3f}s  {r['tpf_per_scenario_ms']:.3f} ms/Sz  "
                      f"k_out={r['k_out_med']:.1f}/{r['k_out_max']}  "
                      f"eff={r['batch_eff_outer']:.2f}  conv={r['conv_frac']:.3f}  "
                      f"S={r.get('speedup', float('nan')):.1f}")
    analyze(runs, runs.replace(".csv", "_fits.csv"))


if __name__ == "__main__":
    main()

# tensor_power_flow/scripts/run_rx_sweep.py
"""
R/X-Sweep: erzeugt eine CSV mit allen Metriken je (nodes, pv_ratio, rx, mode).

Aufruf:
    python -m scripts.run_rx_sweep --out results/rx_sweep.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from tpf.builders.from_pandapower import build_network_from_pandapower
from tpf.generators.rx_sweep import build_rx_case, rx_grid, Z_REF
from tpf.analysis import rx_diagnostics as dg
from tpf.analysis import rx_scaling as sc
import pandapower as pp
import copy
import inspect
import time
from pathlib import Path

JOB_KEY = ("nodes", "pv_ratio", "rx", "mode", "variant")


def _kw(fn, **kw):
    """Nur Kwargs weitergeben, die die Zielfunktion auch kennt."""
    ok = inspect.signature(fn).parameters
    return {k: v for k, v in kw.items() if k in ok and v is not None}


def _enc(seq) -> str:
    """Liste -> ';'-getrennter String (wie sc.encode_residuals)."""
    return ";".join(f"{float(v):.6e}" for v in (seq or []))


def nr_reference(net) -> dict:
    """Newton-Raphson-Referenz: markiert elektrisch nicht loesbare Faelle."""
    n2 = copy.deepcopy(net)
    try:
        pp.runpp(n2, algorithm="nr", init="flat", max_iteration=50)
    except Exception as e:
        return {"nr_conv": False, "nr_iter": np.nan,
                "nr_v_min": np.nan, "nr_err": type(e).__name__}
    ppc = n2.get("_ppc") or {}
    return {"nr_conv": True,
            "nr_iter": float(ppc.get("iterations", np.nan)),
            "nr_v_min": float(n2.res_bus.vm_pu.min()), "nr_err": ""}


def build_jobs(args, grid) -> list[dict]:
    """Alle Kombinationen; PQ zuerst, damit Tab. rx-inner auch bei Abbruch steht."""
    jobs = []
    for mode in args.modes:
        for n in args.nodes:
            for ratio in args.pv_ratios:
                # ohne PV ist die Variante bedeutungslos -> nur einmal rechnen
                variants = ["coupled"] if ratio == 0.0 else list(args.variants)
                for variant in variants:
                    for rx in grid:
                        jobs.append({"nodes": int(n), "pv_ratio": float(ratio),
                                     "rx": float(rx), "mode": mode,
                                     "variant": variant})
    jobs.sort(key=lambda j: (j["pv_ratio"] > 0, j["nodes"], j["mode"],
                             j["pv_ratio"], j["variant"], j["rx"]))
    return jobs


def _key(d) -> tuple:
    return (int(d["nodes"]), round(float(d["pv_ratio"]), 6),
            round(float(d["rx"]), 6), str(d["mode"]), str(d["variant"]))


def load_previous(path) -> tuple[list[dict], set]:
    """Bereits gerechnete Zeilen laden (Resume)."""
    p = Path(path)
    if not p.exists():
        return [], set()
    df = pd.read_csv(p)
    if "variant" not in df.columns:
        df["variant"] = "coupled"
    keys = set(zip(df["nodes"].astype(int), df["pv_ratio"].round(6),
                   df["rx"].round(6), df["mode"].astype(str),
                   df["variant"].astype(str)))
    return df.to_dict("records"), keys

def run_case(nodes, pv_ratio, rx, mode, variant, args) -> dict:
    decoupled = variant == "decoupled"
    exact_sens = variant == "exact"

    case = build_rx_case(nodes=nodes, pv_ratio=pv_ratio, rx=rx, mode=mode,
                         z_abs=Z_REF, load_factor=args.load_factor,
                         vm_offset_pu=args.vm_offset, seed=2000 + nodes)
    nd = build_network_from_pandapower(case.net, include_pv=True)
    if case.n_pv > 0 and not nd.has_pv:
        raise RuntimeError(
            f"NetworkData erkennt keine PV-Knoten: net.gen={len(case.net.gen)}, "
            f"n_pv_soll={case.n_pv}, ppc_vorhanden={case.net.get('_ppc') is not None}")

    row = {
        "nodes": nodes, "pv_ratio": pv_ratio, "n_pv": case.n_pv,
        "rx": rx, "mode": mode, "variant": variant,
        "load_factor": args.load_factor,
        "r_ohm_km": case.r_ohm_per_km, "x_ohm_km": case.x_ohm_per_km,
        "z_abs_ohm_km": case.z_abs_ohm_per_km,
        "z_rel": case.z_rel, "line_factor": case.line_factor,
        "decoupled": decoupled, "exact_sens": exact_sens, "skipped": "",
    }

    if args.nr_reference:
        row.update(nr_reference(case.net))

    # ---- innere Schleife: PQ-only (PV als PQ mit Q=0) ----
    s_pq = nd.s_nom.astype(np.complex128).ravel().copy()
    if nd.has_pv:
        s_pq[nd.pv_indices] = s_pq[nd.pv_indices].real
    t0 = time.perf_counter()
    ref = dg.fpi_reference(nd.Y_dd, nd.Y_ds, nd.v_s, s_pq,
                           **_kw(dg.fpi_reference, max_iter=args.max_inner_pq))
    row["t_inner_s"] = time.perf_counter() - t0
    row.update({
        "inner_iter_pq": ref["n_iter"],
        "inner_conv_pq": ref["converged"],
        "v_min": ref["v_min"],
        "eta_emp": dg.eta_empirical(ref["err"]),
    })
    row.update(dg.eta_theoretical(nd.Y_dd, s_pq, ref["v_min"]))

    fit = sc.eta_from_residuals(ref["err"], skip_head=2)
    row.update({
        "eta_fit": fit["eta"], "eta_fit_r2": fit["r2"],
        "eta_fit_n": fit["n_fit"], "eta_fit_method": fit["method"],
        "res_hist": sc.encode_residuals(ref["err"]),
    })
    row["kappa_emp"] = sc.kappa(fit["eta"], ref["v_min"], case.z_rel)

    if not nd.has_pv:
        return row

    # ---- PV-PV-Block ----
    row.update(dg.xpp_metrics(nd.Y_dd, nd.pv_indices))

    # Screening: hoffnungslose Faelle nicht 60x500 Iterationen rechnen
    if not bool(ref["converged"]):
        row["skipped"] = "inner_div"
    elif np.isfinite(row.get("eta_2", np.nan)) and row["eta_2"] > args.eta_skip:
        row["skipped"] = "eta_bound"
    elif args.skip_nr_fail and row.get("nr_conv") is False:
        row["skipped"] = "nr_fail"
    if row["skipped"]:
        return row

    t0 = time.perf_counter()
    res = dg.method_a_instrumented(
        nd.Y_dd, nd.Y_ds, nd.v_s, nd.s_nom,
        nd.pv_indices, nd.pv_v_setpoint,
        decoupled=decoupled, exact_sensitivity=exact_sens,
        **_kw(dg.method_a_instrumented,
              max_outer=args.max_outer, max_inner=args.max_inner))
    row["t_outer_s"] = time.perf_counter() - t0

    h = res["hist"]
    se = h.get("sens_error") or []
    row.update({
        "outer_iter": res["outer"],
        "inner_total": res["inner_total"],
        "outer_conv": res["converged"],
        "v_err_final": res["v_err_final"],
        "q_max_final": h["q_max"][-1] if h.get("q_max") else 0.0,
        "sens_error_first": se[0] if se else np.nan,
        "sens_error_median": float(np.median(se)) if se else np.nan,
        "sens_error_max": float(np.max(se)) if se else np.nan,
        "v_err_hist": _enc(h.get("v_err")),
        "q_max_hist": _enc(h.get("q_max")),
        "sens_err_hist": _enc(se),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rx_sweep.csv")
    ap.add_argument("--nodes", type=int, nargs="+", default=[40, 120, 350])
    ap.add_argument("--pv-ratios", type=float, nargs="+",
                    default=[0.0, 0.10, 0.25, 0.50])
    ap.add_argument("--modes", nargs="+", default=["const_z", "const_x"])
    ap.add_argument("--variants", nargs="+",
                    default=["coupled", "exact", "decoupled"])
    ap.add_argument("--load-factor", type=float, default=2.0)
    ap.add_argument("--rx-points", type=int, default=13)
    ap.add_argument("--rx-min", type=float, default=0.1)
    ap.add_argument("--rx-max", type=float, default=10.0)
    ap.add_argument("--vm-offset", type=float, default=0.005)
    # Kosten- und Abbruchsteuerung
    ap.add_argument("--max-inner-pq", type=int, default=500)
    ap.add_argument("--max-inner", type=int, default=200)
    ap.add_argument("--max-outer", type=int, default=60)
    ap.add_argument("--eta-skip", type=float, default=1.5,
                    help="Methode A ueberspringen, wenn eta_2 groesser ist")
    ap.add_argument("--skip-nr-fail", action="store_true",
                    help="Faelle ohne NR-Loesung nicht rechnen")
    ap.add_argument("--no-nr-reference", dest="nr_reference",
                    action="store_false")
    ap.set_defaults(nr_reference=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint", type=int, default=20)
    args = ap.parse_args()

    grid = rx_grid(args.rx_points, args.rx_min, args.rx_max)
    jobs = build_jobs(args, grid)
    rows, done = load_previous(args.out) if args.resume else ([], set())
    todo = [j for j in jobs if _key(j) not in done]
    print(f"{len(jobs)} Jobs, {len(done)} vorhanden, {len(todo)} offen")

    for k, job in enumerate(todo, 1):
        try:
            r = run_case(job["nodes"], job["pv_ratio"], job["rx"],
                         job["mode"], job["variant"], args)
            rows.append(r)
            print(f"[{k:>4}/{len(todo)}] n={job['nodes']:<4} "
                  f"pv={job['pv_ratio']:<5} R/X={job['rx']:<6.3f} "
                  f"{job['mode']:<8} {job['variant']:<9} "
                  f"eta={r.get('eta_emp', float('nan')):.3f} "
                  f"outer={r.get('outer_iter', '-')} "
                  f"conv={r.get('outer_conv', '-')} "
                  f"skip={r.get('skipped') or '-'}")
        except Exception as e:
            print(f"[{k:>4}/{len(todo)}] FAIL {job}: {type(e).__name__}: {e}")
            rows.append({**job, "error": str(e)})
        if args.checkpoint and k % args.checkpoint == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\ngeschrieben: {args.out}  ({len(rows)} Zeilen)")


if __name__ == "__main__":
    main()
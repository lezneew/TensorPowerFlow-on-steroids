#!/usr/bin/env python3
# lambda_sweep.py
"""
Sweep: Einfluss des Lastfaktors auf TPF/Methode A.
Erzeugt CSV-Rohdaten + results_lastfaktor.md (via lambda_report.py).

Aufruf:
    python lambda_sweep.py --out results_lastfaktor
    python lambda_sweep.py --experiments e1,e2,e3 --repeats 5
    python lambda_sweep.py --full            # inkl. e7 (lambda x R/X) und e8 (Batch)
"""
from __future__ import annotations

import argparse, json, platform, sys, time, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import pandapower as pp
    HAVE_PP = True
except Exception as _e:
    HAVE_PP, PP_ERR = False, repr(_e)

# ── optionale Anbindung an die Produktions-Solver (nur Gegenprobe) ────────────
PROD = {"ok": False, "reason": "nicht versucht"}
try:
    from tpf.core.network import NetworkData
    from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
    PROD = {"ok": True, "reason": ""}
except Exception as _e:
    PROD = {"ok": False, "reason": repr(_e)}


# ═════════════════════════════════════════════════════════════════════════════
#  Konfiguration
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    out: Path = Path("results_lastfaktor")
    # Netze
    n_bus_list: tuple = (40, 120, 350)
    pv_share_list: tuple = (0.10, 0.25, 0.50)
    seed: int = 0
    branch_span: int = 3          # 1 = reine Kette
    vn_kv: float = 0.4
    s_base_mva: float = 1.0
    len_km: float = 0.02
    # Leitungsbelag (Anker wie im R/X-Sweep: NAYY 4x50 SE)
    z0_ohm_km: float = 0.6597
    x0_ohm_km: float = 0.6513
    rx_ref: float = 1.0          # realistischer MS-Wert, const_z
    rx_mode: str = "const_z"
    # Betriebspunkt
    cos_phi: float = 0.95
    v_min_target_lam1: float = 0.97
    pv_p_total_ratio: float = 0.30   # Sum(P_pv) / Sum(P_load) bei lambda=1
    inverter_oversize: float = 1.10  # S_r / P_pv  -> Q-Reserve
    dv_setpoint: float = 0.005
    # Sweep
    lam_lo: float = 0.2
    lam_hi: float = 11.0
    n_lam: int = 40
    # Solver
    tol_inner: float = 1e-6
    tol_rate: float = 1e-12
    tol_pv: float = 1e-6
    max_inner: int = 300
    max_outer: int = 60
    omega: float = 1.0
    omega_list: tuple = (1.0, 0.8, 0.6, 0.4)
    # Timing
    repeats: int = 1
    # Grenzsuche
    bisect_tol: float = 0.02
    eps_lin_thr: float = 0.6
    # E7 / E8
    rx_list: tuple = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0)
    tau_batch: int = 2000


CFG = Config()
LAM_GRID: np.ndarray = np.array([])


# ═════════════════════════════════════════════════════════════════════════════
#  Netzaufbau (Y-Matrizen manuell -> unabhängig von pandapower-Internals)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Grid:
    n: int
    rx: float
    parents: np.ndarray
    Y_dd: np.ndarray
    Y_ds: np.ndarray
    Y_sd: np.ndarray
    Y_ss: np.ndarray
    Z_B: np.ndarray
    K: np.ndarray
    L: np.ndarray
    p_prof: np.ndarray
    q_prof: np.ndarray
    load_scale: float
    z_abs_ohm_km: float
    z_rel: float
    r_km: float
    x_km: float
    net: object = None


@dataclass
class Case:
    grid: Grid
    pv_share: float
    pv_idx: np.ndarray
    p_pv: np.ndarray          # p.u., positiv = Einspeisung
    q_avail: np.ndarray       # p.u. Blindleistungsreserve je WR

    @property
    def n(self): return self.grid.n

    @property
    def n_pv(self): return len(self.pv_idx)


def _line_params(cfg: Config, rx: float, mode: str):
    if mode == "const_z":
        r = cfg.z0_ohm_km * rx / np.sqrt(1 + rx ** 2)
        x = cfg.z0_ohm_km / np.sqrt(1 + rx ** 2)
    elif mode == "const_x":
        x = cfg.x0_ohm_km
        r = rx * x
    else:
        raise ValueError(mode)
    return r, x


def fpi(K, L, s, tol, max_iter, V0=None):
    """Innere Fixpunktiteration (tau=1).  Rückgabe: V, k, ok, r_final, historie."""
    V = np.ones(K.shape[0], dtype=complex) if V0 is None else V0.copy()
    Sc = np.conj(s)
    hist, r = [], np.inf
    for k in range(1, max_iter + 1):
        Vn = K @ (Sc / np.conj(V)) + L
        if not np.all(np.isfinite(Vn)):
            return V, k, False, np.inf, hist
        r = float(np.max(np.abs(np.abs(Vn) - np.abs(V))))
        V = Vn
        hist.append(r)
        if r < tol:
            return V, k, True, r, hist
    return V, max_iter, False, r, hist


def build_grid(cfg: Config, n: int, rx: float | None = None,
               mode: str | None = None, seed: int | None = None) -> Grid:
    rx = cfg.rx_ref if rx is None else rx
    mode = cfg.rx_mode if mode is None else mode
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(1000 * seed + n)

    r_km, x_km = _line_params(cfg, rx, mode)
    z_abs = float(np.hypot(r_km, x_km))
    z_ref = _line_params(cfg, cfg.rx_ref, "const_z")
    z_rel = z_abs / float(np.hypot(*z_ref))

    # Topologie: radialer Baum, Elternwahl im Fenster branch_span
    parents = np.zeros(n, dtype=int)
    for i in range(n):
        parents[i] = 0 if i == 0 else int(rng.integers(max(0, i - cfg.branch_span), i + 1))

    Zb = cfg.vn_kv ** 2 / cfg.s_base_mva
    y = 1.0 / ((r_km + 1j * x_km) * cfg.len_km / Zb)
    Y = np.zeros((n + 1, n + 1), dtype=complex)
    for i in range(n):
        a, b = parents[i], i + 1
        Y[a, a] += y; Y[b, b] += y; Y[a, b] -= y; Y[b, a] -= y

    Y_dd = Y[1:, 1:].copy()
    Y_ds = Y[1:, 0:1].copy()
    Y_sd = Y[0:1, 1:].copy()
    Y_ss = Y[0:1, 0:1].copy()
    Z_B = np.linalg.inv(Y_dd)
    K = -Z_B
    L = (K @ Y_ds @ np.array([1.0 + 0j])).ravel()

    tan_phi = np.tan(np.arccos(cfg.cos_phi))
    p_prof = np.ones(n)
    q_prof = tan_phi * p_prof

    # Kalibrierung: v_min(lambda=1, reines PQ) == v_min_target
    def vmin_of(scale):
        s = scale * (p_prof + 1j * q_prof)
        V, _, ok, _, _ = fpi(K, L, s, 1e-10, 400)
        return float(np.min(np.abs(V))) if ok else -1.0

    lo, hi = 0.0, 1e-6
    while vmin_of(hi) > cfg.v_min_target_lam1 and hi < 1e3:
        lo, hi = hi, hi * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if vmin_of(mid) > cfg.v_min_target_lam1:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)

    net = _build_pp_net(cfg, n, parents, r_km, x_km) if HAVE_PP else None

    return Grid(n=n, rx=rx, parents=parents, Y_dd=Y_dd, Y_ds=Y_ds, Y_sd=Y_sd,
                Y_ss=Y_ss, Z_B=Z_B, K=K, L=L, p_prof=p_prof, q_prof=q_prof,
                load_scale=scale, z_abs_ohm_km=z_abs, z_rel=z_rel,
                r_km=r_km, x_km=x_km, net=net)


def _build_pp_net(cfg: Config, n, parents, r_km, x_km):
    net = pp.create_empty_network(sn_mva=cfg.s_base_mva)
    pp.create_bus(net, vn_kv=cfg.vn_kv, name="slack")
    for i in range(n):
        pp.create_bus(net, vn_kv=cfg.vn_kv, name=f"b{i+1}")
    pp.create_ext_grid(net, 0, vm_pu=1.0, va_degree=0.0)
    for i in range(n):
        pp.create_line_from_parameters(
            net, from_bus=int(parents[i]), to_bus=i + 1, length_km=cfg.len_km,
            r_ohm_per_km=r_km, x_ohm_per_km=x_km, c_nf_per_km=0.0,
            g_us_per_km=0.0, max_i_ka=10.0)
    for i in range(n):
        pp.create_load(net, bus=i + 1, p_mw=0.0, q_mvar=0.0)
    return net


def make_case(cfg: Config, grid: Grid, pv_share: float) -> Case:
    n = grid.n
    n_pv = int(round(pv_share * n))
    if n_pv == 0:
        pv_idx = np.array([], dtype=int)
        p_pv = np.array([])
    else:
        pv_idx = np.unique(np.linspace(0, n - 1, n_pv).round().astype(int))
        p_tot = cfg.pv_p_total_ratio * float((grid.load_scale * grid.p_prof).sum())
        p_pv = np.full(len(pv_idx), p_tot / len(pv_idx))
    q_avail = p_pv * np.sqrt(max(cfg.inverter_oversize ** 2 - 1.0, 0.0)) if n_pv else np.array([])

    if grid.net is not None:
        # Generatoren einmalig anlegen / synchronisieren
        if len(grid.net.gen):
            grid.net.gen.drop(grid.net.gen.index, inplace=True)
        for k, b in enumerate(pv_idx):
            pp.create_gen(grid.net, bus=int(b) + 1, p_mw=0.0, vm_pu=1.0,
                          name=f"pv{k}", slack=False)
    return Case(grid=grid, pv_share=pv_share, pv_idx=pv_idx, p_pv=p_pv, q_avail=q_avail)


def s_of_lambda(case: Case, lam: float, scale_mode: str = "load") -> np.ndarray:
    """Lastkonvention: positiv = Verbrauch.  PV-Einspeisung wird subtrahiert."""
    g = case.grid
    lam_load = lam if scale_mode in ("load", "load_pv") else 1.0
    lam_pv = lam if scale_mode in ("load_pv", "pv_only") else 1.0
    s = lam_load * g.load_scale * (g.p_prof + 1j * g.q_prof)
    if case.n_pv:
        s[case.pv_idx] -= lam_pv * case.p_pv
    return s


def base_solution(case: Case, s, tol=None, max_iter=None):
    cfg = CFG
    return fpi(case.grid.K, case.grid.L, s,
               tol or cfg.tol_inner, max_iter or cfg.max_inner)


def setpoints(case: Case, s, mode="calibrated", fixed=1.0):
    """V_spec aus Basislösung (Q=0) + Offset, oder fest."""
    V, _, ok, _, _ = base_solution(case, s, tol=1e-10, max_iter=400)
    if not ok:
        return None, None, False
    vmag = np.abs(V)
    if mode == "calibrated":
        vs = vmag[case.pv_idx] + CFG.dv_setpoint
    else:
        vs = np.full(case.n_pv, fixed)
    return vs, float(vmag.min()), True


# ═════════════════════════════════════════════════════════════════════════════
#  Kennzahlen
# ═════════════════════════════════════════════════════════════════════════════
def geometric_rate(hist, tol_floor):
    r = np.asarray(hist, dtype=float)
    m = np.isfinite(r) & (r > 0)
    r = r[m]
    if r.size < 4:
        return np.nan, np.nan
    idx = np.arange(1, r.size - 1)                    # Transiente + Boden raus
    idx = idx[r[idx] > 10.0 * tol_floor]
    if idx.size < 3:
        idx = np.arange(1, r.size - 1)
    y = np.log(r[idx])
    b, a = np.polyfit(idx, y, 1)
    yh = a + b * idx
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum((y - yh) ** 2) / ss if ss > 0 else np.nan
    return float(np.exp(b)), float(r2)


def eta_bounds(Z_B, s, V):
    d = np.abs(V) ** 2
    M = Z_B * (np.abs(s) / d)[None, :]
    return (float(np.linalg.norm(M, 1)),
            float(np.linalg.norm(M, 2)),
            float(np.linalg.norm(M, np.inf)))


def feasibility_index(Z_B, s, v_slack=1.0):
    """min_k |v_s|^2 / (4 |s_k| |z_kk|)   (>1 => hinreichend lösbar)."""
    zkk = np.abs(np.diag(Z_B))
    den = 4.0 * np.abs(s) * zkk
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = np.where(den > 0, v_slack ** 2 / den, np.inf)
    return float(np.min(idx))


# ═════════════════════════════════════════════════════════════════════════════
#  Instrumentierte Methode A (exakter Nachbau von _solve_with_pv, tau=1)
# ═════════════════════════════════════════════════════════════════════════════
def method_a(case: Case, s, v_spec, *, omega=None, tol_pv=None, tol_inner=None,
             max_outer=None, max_inner=None, adaptive=False, warm=True,
             decoupled=False, q_init=None, measure_rate=True):
    cfg = CFG
    omega = cfg.omega if omega is None else omega
    tol_pv = cfg.tol_pv if tol_pv is None else tol_pv
    tol_inner = cfg.tol_inner if tol_inner is None else tol_inner
    max_outer = cfg.max_outer if max_outer is None else max_outer
    max_inner = cfg.max_inner if max_inner is None else max_inner

    g = case.grid
    pv = case.pv_idx
    npv = len(pv)
    Zpp = g.Z_B[np.ix_(pv, pv)]
    Xpp = np.imag(Zpp)
    x_kk = np.diag(Xpp).copy()
    z_kk = np.abs(np.diag(Zpp))
    rho_kk = np.abs(np.real(np.diag(Zpp))) / np.maximum(np.abs(x_kk), 1e-30)
    A_inv = np.linalg.inv(2.0 * Xpp + 1e-12 * np.eye(npv))

    q = np.zeros(npv) if q_init is None else np.asarray(q_init, float).copy()
    s_work = s.astype(complex).copy()
    p_pv_fixed = s_work[pv].real.copy()
    s_work[pv] = p_pv_fixed + 1j * q

    V = np.ones(g.n, dtype=complex)
    steps, k_in_total, k_out, ok = [], 0, 0, False
    err = np.inf
    t0 = time.perf_counter()

    for ell in range(max_outer):
        if not warm:
            V = np.ones(g.n, dtype=complex)
        err_pre = float(np.max(np.abs(np.abs(V[pv]) - v_spec)))
        tol_eff = max(tol_inner, err_pre) if adaptive else tol_inner
        V, k_in, _, r_in, _ = fpi(g.K, g.L, s_work, tol_eff, max_inner, V0=V)
        k_in_total += k_in
        k_out = ell + 1
        if not np.all(np.isfinite(V)):
            err = np.inf
            steps.append(dict(ell=k_out, k_in=k_in, err=np.inf, v_min=np.nan,
                              q_max=float(np.max(np.abs(q))) if npv else 0.0,
                              dq_max=np.nan, eps_med=np.nan, eps_max=np.nan,
                              eps_cf_med=np.nan, tol_eff=tol_eff))
            break

        vmag = np.abs(V[pv])
        err = float(np.max(np.abs(vmag - v_spec)))
        rec = dict(ell=k_out, k_in=k_in, err=err, v_min=float(np.abs(V).min()),
                   q_max=float(np.max(np.abs(q))) if npv else 0.0,
                   tol_eff=tol_eff)

        if err < tol_pv:
            rec.update(dq_max=0.0, eps_med=np.nan, eps_max=np.nan, eps_cf_med=np.nan)
            steps.append(rec)
            ok = True
            break

        dvsq = v_spec ** 2 - vmag ** 2
        dq = (dvsq / (2.0 * x_kk)) if decoupled else (A_inv @ dvsq)
        dQ = np.abs(omega * dq)
        eps = dQ * z_kk ** 2 / (2.0 * np.abs(x_kk) * vmag ** 2)
        eps_cf = (1.0 + rho_kk ** 2) * np.abs(dvsq) / (4.0 * vmag ** 2)
        rec.update(dq_max=float(np.max(dQ)), eps_med=float(np.median(eps)),
                   eps_max=float(np.max(eps)), eps_cf_med=float(np.median(eps_cf)))
        steps.append(rec)

        q = q - omega * dq
        s_work[pv] = p_pv_fixed + 1j * q

    t_ms = (time.perf_counter() - t0) * 1e3
    out = dict(converged=bool(ok), k_out=int(k_out), k_in=int(k_in_total),
               err_final=float(err), t_ms=float(t_ms),
               q=q.copy(), V=V.copy(), s_work=s_work.copy(), steps=steps)

    df = pd.DataFrame(steps)
    out["eps_med"] = float(np.nanmedian(df["eps_med"])) if len(df) else np.nan
    out["eps_max"] = float(np.nanmax(df["eps_max"])) if len(df) else np.nan
    out["eps_cf_med"] = float(np.nanmedian(df["eps_cf_med"])) if len(df) else np.nan
    out["q_max"] = float(np.max(np.abs(q))) if npv else 0.0
    out["kin_per_kout"] = out["k_in"] / max(out["k_out"], 1)
    out["v_min"] = float(np.abs(V).min()) if np.all(np.isfinite(V)) else np.nan
    out["v_max"] = float(np.abs(V).max()) if np.all(np.isfinite(V)) else np.nan

    if measure_rate and np.all(np.isfinite(V)):
        _, _, _, _, h = fpi(g.K, g.L, s_work, CFG.tol_rate, 500, V0=np.ones(g.n, complex))
        e, r2 = geometric_rate(h, CFG.tol_rate)
        out["eta_emp"] = e
        out["eta_r2"] = r2
        e1, e2, ei = eta_bounds(g.Z_B, s_work, V)
        out["eta1"], out["eta2"], out["etainf"] = e1, e2, ei
    else:
        out.update(eta_emp=np.nan, eta_r2=np.nan, eta1=np.nan, eta2=np.nan, etainf=np.nan)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  Referenzen: pandapower NR und Produktions-Solver
# ═════════════════════════════════════════════════════════════════════════════
def run_nr(case: Case, lam, v_spec, scale_mode="load", repeats=1):
    g = case.grid
    if not HAVE_PP or g.net is None:
        return dict(nr_conv=np.nan, nr_iter=np.nan, nr_vmin=np.nan,
                    nr_vmax=np.nan, nr_qmax=np.nan, t_nr_ms=np.nan)
    net = g.net
    lam_load = lam if scale_mode in ("load", "load_pv") else 1.0
    lam_pv = lam if scale_mode in ("load_pv", "pv_only") else 1.0
    p = lam_load * g.load_scale * g.p_prof * CFG.s_base_mva
    q = lam_load * g.load_scale * g.q_prof * CFG.s_base_mva
    net.load.loc[:, "p_mw"] = p
    net.load.loc[:, "q_mvar"] = q
    if case.n_pv and len(net.gen):
        net.gen.loc[:, "p_mw"] = lam_pv * case.p_pv * CFG.s_base_mva
        net.gen.loc[:, "vm_pu"] = v_spec

    best, conv, it = np.inf, False, -1
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        try:
            pp.runpp(net, algorithm="nr", tolerance_mva=1e-8, max_iteration=100,
                     enforce_q_lims=False, init="flat")
            conv = bool(net.converged)
            it = int(net._ppc.get("iterations", -1))
        except Exception:
            conv, it = False, -1
        best = min(best, (time.perf_counter() - t0) * 1e3)

    if conv:
        vm = net.res_bus["vm_pu"].values[1:]
        qmax = float(np.max(np.abs(net.res_gen["q_mvar"].values / CFG.s_base_mva))) \
            if (case.n_pv and len(net.res_gen)) else 0.0
        return dict(nr_conv=True, nr_iter=it, nr_vmin=float(vm.min()),
                    nr_vmax=float(vm.max()), nr_qmax=qmax, t_nr_ms=best)
    return dict(nr_conv=False, nr_iter=it, nr_vmin=np.nan, nr_vmax=np.nan,
                nr_qmax=np.nan, t_nr_ms=best)


def run_production(case: Case, s, v_spec, adaptive=False, warm=True, repeats=1):
    """Gegenprobe mit TPFDensePVMethodA (optional)."""
    if not PROD["ok"]:
        return {}
    import inspect
    g = case.grid
    kw = dict(Y_dd=g.Y_dd, Y_ds=g.Y_ds, Y_sd=g.Y_sd, Y_ss=g.Y_ss,
              v_s=np.array([1.0 + 0j]), s_nom=s.astype(complex).copy(),
              alpha_p=np.ones(g.n), alpha_i=np.zeros(g.n), alpha_z=np.zeros(g.n),
              pv_indices=case.pv_idx, pv_v_setpoint=np.asarray(v_spec, float),
              pv_q_min=None, pv_q_max=None)
    try:
        params = inspect.signature(NetworkData).parameters
        nd = NetworkData(**{k: v for k, v in kw.items() if k in params})
        for k, v in kw.items():
            if not hasattr(nd, k):
                try: setattr(nd, k, v)
                except Exception: pass
        solver = TPFDensePVMethodA(tol=CFG.tol_inner, max_iter_inner=CFG.max_inner,
                                  max_iter_outer=CFG.max_outer, tol_pv=CFG.tol_pv,
                                  omega=CFG.omega, cold_start=not warm,
                                  adaptive_inner=adaptive, use_decoupled=False)
        best, res = np.inf, None
        for _ in range(max(1, repeats)):
            r = solver.solve_batch(nd, s.reshape(-1, 1))
            best = min(best, r.elapsed_time_s * 1e3)
            res = r
        info = solver.pv_info
        return dict(prod_ok=bool(res.converged), prod_k_out=int(info.outer_iterations),
                    prod_k_in=int(info.inner_iterations_total),
                    prod_err=float(info.pv_v_error_final), prod_t_ms=float(best))
    except Exception as e:
        PROD["ok"] = False
        PROD["reason"] = repr(e)
        return {}


# ═════════════════════════════════════════════════════════════════════════════
#  Hilfsfunktionen: Grenzsuche, Klassifikation
# ═════════════════════════════════════════════════════════════════════════════
def find_lambda_star(pred, lam_lo=0.2, lam_hi=30.0, factor=1.5, tol=None):
    """Kleinstes lambda mit pred(lambda)=False (Monotonie angenommen)."""
    tol = CFG.bisect_tol if tol is None else tol
    if not pred(lam_lo):
        return lam_lo
    ok, lam = lam_lo, lam_lo
    while lam < lam_hi:
        lam = min(lam * factor, lam_hi)
        if not pred(lam):
            lo, hi = ok, lam
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                if pred(mid): lo = mid
                else: hi = mid
            return float(hi)
        ok = lam
        if lam >= lam_hi:
            break
    return float("nan")


def classify(tpf_ok, nr_ok, base_ok, err, kin_per_kout):
    if not base_ok:
        return "infeasible_base"
    if tpf_ok and nr_ok:
        return "converged"
    if tpf_ok and not nr_ok:
        return "tpf_only"
    if (not tpf_ok) and nr_ok:
        return "limit_cycle" if (err < 1e-3 and kin_per_kout < 20) else "divergence"
    return "no_solution"


# ═════════════════════════════════════════════════════════════════════════════
#  Experimente
# ═════════════════════════════════════════════════════════════════════════════
_GRID_CACHE: dict = {}


def get_grid(n, rx=None, mode=None):
    key = (n, rx if rx is not None else CFG.rx_ref, mode or CFG.rx_mode, CFG.seed)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = build_grid(CFG, n, rx=key[1], mode=key[2], seed=CFG.seed)
    return _GRID_CACHE[key]


def e1_inner():
    """H1/H2: innere Schleife isoliert (n_pv = 0)."""
    rows = []
    for n in CFG.n_bus_list:
        g = get_grid(n)
        case = make_case(CFG, g, 0.0)
        fails = 0
        for lam in LAM_GRID:
            s = s_of_lambda(case, lam, "load")
            V, k12, ok12, _, hist = fpi(g.K, g.L, s, CFG.tol_rate, 500)
            eta, r2 = geometric_rate(hist, CFG.tol_rate)
            _, k6, ok6, _, _ = fpi(g.K, g.L, s, CFG.tol_inner, 500)
            e1_, e2_, ei_ = eta_bounds(g.Z_B, s, V if ok12 else np.ones(n, complex))
            _, _, _, _, hnom = fpi(g.K, g.L, s, 1e-3, 1)
            e1n, e2n, ein = eta_bounds(g.Z_B, s, np.ones(n, complex))
            vmin = float(np.abs(V).min()) if ok12 else np.nan
            kpred = (np.log(CFG.tol_rate / max(hist[0], 1e-300)) / np.log(eta)
                     if (ok12 and np.isfinite(eta) and 0 < eta < 1) else np.nan)
            rows.append(dict(n=n, lam=float(lam), z_rel=g.z_rel,
                             load_scale=g.load_scale, p_tot=float(lam * g.load_scale * n),
                             conv=bool(ok12), k_in_12=int(k12), k_in_6=int(k6),
                             k_pred=kpred, eta_emp=eta, eta_r2=r2,
                             eta1=e1_, eta2=e2_, etainf=ei_,
                             eta2_nom=e2n, v_min=vmin,
                             feas_min=feasibility_index(g.Z_B, s)))
            fails = fails + 1 if not ok12 else 0
            if fails >= 3:
                break
        print(f"  [e1] n={n} fertig ({len(rows)} Zeilen)")
    return pd.DataFrame(rows)


def e2_outer(setpoint_modes=("calibrated", "fixed")):
    """H3/H4/H6: äußere Schleife, NR-Referenz, Klassifikation."""
    rows, steps = [], []
    for n in CFG.n_bus_list:
        g = get_grid(n)
        for pvs in CFG.pv_share_list:
            case = make_case(CFG, g, pvs)
            for sp_mode in setpoint_modes:
                fails = 0
                for lam in LAM_GRID:
                    s = s_of_lambda(case, lam, "load")
                    vs, vmin_base, base_ok = setpoints(case, s, sp_mode, 1.0)
                    if not base_ok:
                        rows.append(dict(n=n, pv_share=pvs, n_pv=case.n_pv,
                                         setpoint_mode=sp_mode, scale_mode="load",
                                         lam=float(lam), base_ok=False,
                                         cls="infeasible_base"))
                        fails += 1
                        if fails >= 3: break
                        continue
                    r = method_a(case, s, vs, warm=True, adaptive=False)
                    nr = run_nr(case, lam, vs, "load", repeats=CFG.repeats)
                    prod = run_production(case, s, vs, repeats=1)

                    t_best = r["t_ms"]
                    for _ in range(max(0, CFG.repeats - 1)):
                        t_best = min(t_best, method_a(case, s, vs, measure_rate=False)["t_ms"])

                    dv_nr = np.nan
                    if nr.get("nr_conv") is True and np.isfinite(r["v_min"]):
                        try:
                            vm_nr = g.net.res_bus["vm_pu"].values[1:]
                            dv_nr = float(np.max(np.abs(np.abs(r["V"]) - vm_nr)))
                        except Exception:
                            pass
                    q_av = float(np.min(case.q_avail)) if case.n_pv else np.nan
                    row = dict(
                        n=n, pv_share=pvs, n_pv=case.n_pv, setpoint_mode=sp_mode,
                        scale_mode="load", lam=float(lam), base_ok=True,
                        v_min_base=vmin_base, v_spec_min=float(np.min(vs)),
                        v_spec_max=float(np.max(vs)),
                        tpf_conv=r["converged"], k_out=r["k_out"], k_in=r["k_in"],
                        kin_per_kout=r["kin_per_kout"], err_final=r["err_final"],
                        eps_med=r["eps_med"], eps_max=r["eps_max"],
                        eps_cf_med=r["eps_cf_med"], q_max=r["q_max"],
                        q_avail=q_av, q_util=(r["q_max"] / q_av if q_av and q_av > 0 else np.nan),
                        v_min=r["v_min"], v_max=r["v_max"], eta_emp=r["eta_emp"],
                        eta2=r["eta2"], t_tpf_ms=t_best,
                        feas_min=feasibility_index(g.Z_B, s), dv_max_vs_nr=dv_nr, **nr)
                    row.update(prod)
                    row["cls"] = classify(r["converged"], nr.get("nr_conv"),
                                          True, r["err_final"], r["kin_per_kout"])
                    rows.append(row)
                    for st in r["steps"]:
                        steps.append(dict(n=n, pv_share=pvs, setpoint_mode=sp_mode,
                                          lam=float(lam), **st))
                    fails = fails + 1 if not r["converged"] else 0
                    if fails >= 3:
                        break
                print(f"  [e2] n={n} pv={pvs} sp={sp_mode} fertig")
    return pd.DataFrame(rows), pd.DataFrame(steps)


def e3_limits():
    """lambda* je Kriterium."""
    rows = []
    for n in CFG.n_bus_list:
        g = get_grid(n)
        # innere Schleife / Basis / eta=1
        c0 = make_case(CFG, g, 0.0)

        def p_base(l):
            _, _, ok, _, _ = fpi(g.K, g.L, s_of_lambda(c0, l, "load"), CFG.tol_inner, CFG.max_inner)
            return ok

        def p_eta(l):
            s = s_of_lambda(c0, l, "load")
            V, _, ok, _, h = fpi(g.K, g.L, s, CFG.tol_rate, 500)
            if not ok:
                return False
            e, _ = geometric_rate(h, CFG.tol_rate)
            return bool(np.isfinite(e) and e < 1.0)

        def p_nr0(l):
            return run_nr(c0, l, np.array([]), "load")["nr_conv"] is True

        rows += [dict(n=n, pv_share=0.0, criterion="base_fpi", lam_star=find_lambda_star(p_base)),
                 dict(n=n, pv_share=0.0, criterion="eta_lt_1", lam_star=find_lambda_star(p_eta)),
                 dict(n=n, pv_share=0.0, criterion="nr_pq", lam_star=find_lambda_star(p_nr0))]

        for pvs in CFG.pv_share_list:
            case = make_case(CFG, g, pvs)

            def _prep(l):
                s = s_of_lambda(case, l, "load")
                vs, _, ok = setpoints(case, s, "calibrated")
                return s, vs, ok

            def p_tpf(l):
                s, vs, ok = _prep(l)
                return ok and method_a(case, s, vs, measure_rate=False)["converged"]

            def p_nr(l):
                s, vs, ok = _prep(l)
                return ok and (run_nr(case, l, vs, "load")["nr_conv"] is True)

            def p_eps(l):
                s, vs, ok = _prep(l)
                if not ok: return False
                r = method_a(case, s, vs, measure_rate=False)
                e = r["eps_med"]
                return (not np.isfinite(e)) or e < CFG.eps_lin_thr

            def p_qlim(l):
                s, vs, ok = _prep(l)
                if not ok: return False
                r = method_a(case, s, vs, measure_rate=False)
                qa = float(np.min(case.q_avail)) if case.n_pv else np.inf
                return r["q_max"] <= qa

            rows += [
                dict(n=n, pv_share=pvs, criterion="tpf_methode_a", lam_star=find_lambda_star(p_tpf)),
                dict(n=n, pv_share=pvs, criterion="nr_pv", lam_star=find_lambda_star(p_nr)),
                dict(n=n, pv_share=pvs, criterion="eps_lin_0.6", lam_star=find_lambda_star(p_eps)),
                dict(n=n, pv_share=pvs, criterion="q_reserve", lam_star=find_lambda_star(p_qlim)),
            ]
            print(f"  [e3] n={n} pv={pvs} fertig")
    return pd.DataFrame(rows)


def e4_damping():
    """Verschiebung von lambda* durch Dämpfung omega."""
    rows = []
    for n in CFG.n_bus_list:
        g = get_grid(n)
        for pvs in CFG.pv_share_list:
            case = make_case(CFG, g, pvs)
            for om in CFG.omega_list:
                def pred(l, om=om):
                    s = s_of_lambda(case, l, "load")
                    vs, _, ok = setpoints(case, s, "calibrated")
                    return ok and method_a(case, s, vs, omega=om, measure_rate=False)["converged"]
                rows.append(dict(n=n, pv_share=pvs, omega=om,
                                 lam_star=find_lambda_star(pred)))
            print(f"  [e4] n={n} pv={pvs} fertig")
    return pd.DataFrame(rows)


def e5_optim():
    """H5: Warm Start / adaptive innere Toleranz über lambda."""
    rows = []
    for n in CFG.n_bus_list:
        g = get_grid(n)
        for pvs in CFG.pv_share_list:
            case = make_case(CFG, g, pvs)
            for lam in LAM_GRID[::2]:
                s = s_of_lambda(case, lam, "load")
                vs, _, ok = setpoints(case, s, "calibrated")
                if not ok:
                    continue
                for warm in (False, True):
                    for ad in (False, True):
                        r = method_a(case, s, vs, warm=warm, adaptive=ad, measure_rate=False)
                        t = r["t_ms"]
                        for _ in range(max(0, CFG.repeats - 1)):
                            t = min(t, method_a(case, s, vs, warm=warm, adaptive=ad,
                                                measure_rate=False)["t_ms"])
                        rows.append(dict(n=n, pv_share=pvs, lam=float(lam),
                                         warm=warm, adaptive=ad,
                                         conv=r["converged"], k_out=r["k_out"],
                                         k_in=r["k_in"], t_ms=t, v_min=r["v_min"]))
            print(f"  [e5] n={n} pv={pvs} fertig")
    return pd.DataFrame(rows)


def e6_continuation():
    """Q=0 vs. lambda-Fortsetzung (Q aus vorherigem lambda) = Zeitreihenfall."""
    rows = []
    for n in CFG.n_bus_list:
        g = get_grid(n)
        for pvs in CFG.pv_share_list:
            case = make_case(CFG, g, pvs)
            q_prev = None
            for lam in LAM_GRID:
                s = s_of_lambda(case, lam, "load")
                vs, _, ok = setpoints(case, s, "calibrated")
                if not ok:
                    break
                r0 = method_a(case, s, vs, q_init=None, measure_rate=False)
                rc = method_a(case, s, vs, q_init=q_prev, measure_rate=False)
                for tag, r in (("cold_q0", r0), ("continuation", rc)):
                    rows.append(dict(n=n, pv_share=pvs, lam=float(lam), q_init_mode=tag,
                                     conv=r["converged"], k_out=r["k_out"], k_in=r["k_in"],
                                     eps_med=r["eps_med"], q_max=r["q_max"], t_ms=r["t_ms"]))
                q_prev = rc["q"] if rc["converged"] else r0["q"]
                if not (r0["converged"] or rc["converged"]):
                    break
            print(f"  [e6] n={n} pv={pvs} fertig")
    return pd.DataFrame(rows)


def e7_lambda_rx():
    """Gemeinsamer Prädiktor: 2D-Gitter (lambda, R/X)."""
    rows = []
    n, pvs = 120, 0.25
    for rx in CFG.rx_list:
        g = get_grid(n, rx=rx, mode="const_z")
        case = make_case(CFG, g, pvs)
        for lam in LAM_GRID[::3]:
            s = s_of_lambda(case, lam, "load")
            vs, _, ok = setpoints(case, s, "calibrated")
            if not ok:
                continue
            r = method_a(case, s, vs, measure_rate=False)
            rows.append(dict(n=n, pv_share=pvs, rx=rx, lam=float(lam),
                             conv=r["converged"], k_out=r["k_out"], k_in=r["k_in"],
                             eps_med=r["eps_med"], eps_max=r["eps_max"],
                             q_max=r["q_max"], v_min=r["v_min"]))
        print(f"  [e7] rx={rx} fertig")
    return pd.DataFrame(rows)


def e8_batch():
    """Batch mit gemischten lambda (Masked Iteration) — benötigt Produktions-Solver."""
    if not PROD["ok"]:
        print("  [e8] übersprungen (Produktions-Solver nicht verfügbar)")
        return pd.DataFrame()
    import inspect
    rows = []
    n, pvs = 120, 0.25
    g = get_grid(n)
    case = make_case(CFG, g, pvs)
    rng = np.random.default_rng(7)
    for spread in (0.0, 0.5, 1.5):
        lam_c = 2.0
        lams = np.clip(lam_c + spread * (rng.random(CFG.tau_batch) - 0.5) * 2, 0.2, 6.0)
        s_ref = s_of_lambda(case, lam_c, "load")
        vs, _, ok = setpoints(case, s_ref, "calibrated")
        if not ok:
            continue
        S = np.stack([s_of_lambda(case, float(l), "load") for l in lams], axis=1)
        try:
            params = inspect.signature(NetworkData).parameters
            kw = dict(Y_dd=g.Y_dd, Y_ds=g.Y_ds, Y_sd=g.Y_sd, Y_ss=g.Y_ss,
                      v_s=np.array([1.0 + 0j]), s_nom=s_ref.copy(),
                      alpha_p=np.ones(n), alpha_i=np.zeros(n), alpha_z=np.zeros(n),
                      pv_indices=case.pv_idx, pv_v_setpoint=vs,
                      pv_q_min=None, pv_q_max=None)
            nd = NetworkData(**{k: v for k, v in kw.items() if k in params})
            for k, v in kw.items():
                if not hasattr(nd, k):
                    try: setattr(nd, k, v)
                    except Exception: pass
            solver = TPFDensePVMethodA(tol=CFG.tol_inner, max_iter_inner=CFG.max_inner,
                                      max_iter_outer=CFG.max_outer, tol_pv=CFG.tol_pv,
                                      adaptive_inner=True)
            t0 = time.perf_counter()
            res = solver.solve_timeseries(nd, S, verbose=False)
            t_batch = (time.perf_counter() - t0) * 1e3
            info = solver.pv_info
            rows.append(dict(mode="batched_masked", spread=spread, tau=CFG.tau_batch,
                             t_ms=t_batch, k_in_total=int(info.inner_iterations_total),
                             n_conv=int(info.n_converged_scenarios)))
        except Exception as e:
            print(f"  [e8] batched fehlgeschlagen: {e!r}")

        t0, kin, nc = time.perf_counter(), 0, 0
        for l in lams[:min(200, CFG.tau_batch)]:
            r = method_a(case, s_of_lambda(case, float(l), "load"), vs,
                         adaptive=True, measure_rate=False)
            kin += r["k_in"]; nc += int(r["converged"])
        t_seq = (time.perf_counter() - t0) * 1e3
        m = min(200, CFG.tau_batch)
        rows.append(dict(mode="sequential", spread=spread, tau=m, t_ms=t_seq,
                         k_in_total=kin, n_conv=nc))
        print(f"  [e8] spread={spread} fertig")
    return pd.DataFrame(rows)


def e9_modes():
    """Skalierungsmodus: Last / Last+PV / nur PV (Rückspeisung)."""
    rows = []
    for n in CFG.n_bus_list:
        g = get_grid(n)
        for pvs in CFG.pv_share_list:
            case = make_case(CFG, g, pvs)
            for mode in ("load", "load_pv", "pv_only"):
                for lam in LAM_GRID[::2]:
                    s = s_of_lambda(case, lam, mode)
                    vs, vmb, ok = setpoints(case, s, "calibrated")
                    if not ok:
                        rows.append(dict(n=n, pv_share=pvs, scale_mode=mode,
                                         lam=float(lam), base_ok=False))
                        continue
                    r = method_a(case, s, vs, measure_rate=False)
                    nr = run_nr(case, lam, vs, mode)
                    rows.append(dict(n=n, pv_share=pvs, scale_mode=mode, lam=float(lam),
                                     base_ok=True, v_min_base=vmb,
                                     tpf_conv=r["converged"], k_out=r["k_out"],
                                     k_in=r["k_in"], eps_med=r["eps_med"],
                                     q_max=r["q_max"], v_min=r["v_min"], v_max=r["v_max"],
                                     nr_conv=nr["nr_conv"]))
            print(f"  [e9] n={n} pv={pvs} fertig")
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = {
    "e1": ("e1_inner.csv", e1_inner),
    "e2": (("e2_outer.csv", "e2_outer_steps.csv"), e2_outer),
    "e3": ("e3_limits.csv", e3_limits),
    "e4": ("e4_damping.csv", e4_damping),
    "e5": ("e5_optim.csv", e5_optim),
    "e6": ("e6_continuation.csv", e6_continuation),
    "e7": ("e7_lambda_rx.csv", e7_lambda_rx),
    "e8": ("e8_batch.csv", e8_batch),
    "e9": ("e9_modes.csv", e9_modes),
}
DEFAULT_SET = ["e1", "e2", "e3", "e4", "e5", "e6", "e9"]


def main():
    global CFG, LAM_GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_lastfaktor")
    ap.add_argument("--experiments", default=",".join(DEFAULT_SET))
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-bus", default="")
    ap.add_argument("--pv-share", default="")
    ap.add_argument("--n-lam", type=int, default=CFG.n_lam)
    ap.add_argument("--lam-hi", type=float, default=CFG.lam_hi)
    ap.add_argument("--repeats", type=int, default=CFG.repeats)
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()

    CFG.out = Path(a.out)
    CFG.n_lam, CFG.lam_hi, CFG.repeats = a.n_lam, a.lam_hi, a.repeats
    if a.n_bus:
        CFG.n_bus_list = tuple(int(x) for x in a.n_bus.split(","))
    if a.pv_share:
        CFG.pv_share_list = tuple(float(x) for x in a.pv_share.split(","))
    LAM_GRID = np.linspace(CFG.lam_lo, CFG.lam_hi, CFG.n_lam)

    exps = list(EXPERIMENTS) if a.full else [e.strip() for e in a.experiments.split(",")]
    CFG.out.mkdir(parents=True, exist_ok=True)

    print(f"pandapower: {HAVE_PP} | Produktions-Solver: {PROD['ok']} ({PROD['reason'][:60]})")
    print(f"lambda-Gitter: {LAM_GRID[0]:.2f} .. {LAM_GRID[-1]:.2f} ({len(LAM_GRID)} Punkte)")

    t0 = time.perf_counter()
    for key in exps:
        if key not in EXPERIMENTS:
            print(f"  unbekanntes Experiment: {key}")
            continue
        names, fn = EXPERIMENTS[key]
        print(f"\n=== {key} ===")
        res = fn()
        if isinstance(names, tuple):
            for nm, df in zip(names, res):
                df.to_csv(CFG.out / nm, index=False)
                print(f"  -> {nm} ({len(df)} Zeilen)")
        else:
            res.to_csv(CFG.out / names, index=False)
            print(f"  -> {names} ({len(res)} Zeilen)")

    meta = dict(config={k: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                        for k, v in asdict(CFG).items()},
                lam_grid=[float(x) for x in LAM_GRID],
                experiments=exps,
                production_solver=PROD, pandapower=HAVE_PP,
                python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, runtime_s=time.perf_counter() - t0)
    (CFG.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nGesamtlaufzeit: {meta['runtime_s']:.1f} s")

    if not a.no_report:
        try:
            import lambda_report
            lambda_report.build(CFG.out)
        except Exception as e:
            print(f"Report übersprungen: {e!r}")


if __name__ == "__main__":
    main()
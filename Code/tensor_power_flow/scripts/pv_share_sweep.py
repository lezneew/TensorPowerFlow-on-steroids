# =============================================================================
# pv_share_sweep.py  —  Sweep zum Abschnitt "Einfluss des PV-Anteils"
# =============================================================================
"""
Testet die Hypothesen H1-H6:
  H1  Treiber ist n_pv absolut, nicht der Anteil n_pv/n_bus
  H2  Gekoppelte Korrektur ist strukturinvariant (k_out ~ const trotz cond -> oo)
  H3  Entkoppelte Naeherung versagt genau bei rho_J > 1 (Cluster-, keine Zahl-Eigenschaft)
  H4  Mehr PV wirkt stabilisierend (v_min hoch, Q pro Knoten runter)
  H5  Der stabilisierende Effekt ist teilweise Artefakt der Sollwertkalibrierung (+delta)
  H6  PV wirkt auf die Laufzeit ausschliesslich ueber k_in, nicht pro Iteration

Wichtige Konvention: PV-Knoten haben per Default P_gen = 0 (pv_p_factor=0), damit
die Wirkleistungsbilanz beim Variieren von n_pv exakt unveraendert bleibt
(Stoergroesse "Lastbilanz"). Ueber --pvp bzw. Case.pv_p_factor zuschaltbar.

Aufruf:
    python pv_share_sweep.py --out results_pv_share
    python pv_share_sweep.py --out results_pv_share --quick
    python pv_share_sweep.py --out results_pv_share --only e1 e4
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver

warnings.filterwarnings("ignore")

COSPHI_LOAD = 0.95
V_MIN_TARGET = 0.97          # Betriebspunkt-Normierung bei lam = 1
Z0_REF = 0.6470              # |z| in Ohm/km  (NAYY 4x50 SE)
RHO_REF = 7.73               # R/X           (NAYY 4x50 SE)
LEN_KM = 0.03
SN_MVA = 1.0
VN_KV = 0.4


# -----------------------------------------------------------------------------
# 1) Duck-typed NetworkData (nur die von den Solvern gelesenen Attribute)
# -----------------------------------------------------------------------------
@dataclass
class SimpleNetworkData:
    Y_dd: np.ndarray
    Y_ds: np.ndarray
    Y_sd: np.ndarray
    Y_ss: np.ndarray
    v_s: np.ndarray
    s_nom: np.ndarray
    pv_indices: np.ndarray | None = None
    pv_v_setpoint: np.ndarray | None = None
    pv_q_min: np.ndarray | None = None
    pv_q_max: np.ndarray | None = None

    @property
    def n_bus_phases(self) -> int:
        return self.Y_dd.shape[0]

    @property
    def alpha_p(self) -> np.ndarray:
        return np.ones(self.n_bus_phases)

    @property
    def alpha_i(self) -> np.ndarray:
        return np.zeros(self.n_bus_phases)

    @property
    def alpha_z(self) -> np.ndarray:
        return np.zeros(self.n_bus_phases)

    @property
    def has_pv(self) -> bool:
        return self.pv_indices is not None and len(self.pv_indices) > 0

    @property
    def n_pv(self) -> int:
        return 0 if self.pv_indices is None else int(len(self.pv_indices))

    @property
    def has_slack_blocks(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# 2) Fallbeschreibung
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    n_bus: int = 40
    seed: int = 0
    n_feeders: int = 4
    n_pv: int = 0
    placement: str = "random"      # random|clustered|spread|leaves|feeders
    lam: float = 1.0
    rho: float = RHO_REF           # R/X, Modus const_z
    z0: float = Z0_REF
    length_km: float = LEN_KM
    sp_mode: str = "delta"         # delta | abs
    delta: float = 0.005
    v_abs: float = 1.00
    tau: int = 1
    q_lim_pu: float | None = None  # symmetrische Q-Grenze (Injektion, p.u.)
    variant: str = "coupled"       # coupled | decoupled
    pv_p_factor: float = 0.0       # P_pv = factor * P_load des Knotens

    @property
    def share(self) -> float:
        return self.n_pv / max(1, self.n_bus - 1)


# -----------------------------------------------------------------------------
# 3) Topologie, Impedanzen, PV-Platzierung
# -----------------------------------------------------------------------------
def line_rx(z0: float, rho: float) -> tuple[float, float]:
    """Modus const_z: |z| fest, nur der Impedanzwinkel wird gedreht."""
    x = z0 / np.sqrt(1.0 + rho ** 2)
    return rho * x, x


def build_topology(n_bus: int, seed: int, n_feeders: int, p_branch: float = 0.12):
    rng = np.random.default_rng(1000 + seed)
    parents = np.full(n_bus, -1, dtype=int)
    feeder = np.full(n_bus, -1, dtype=int)
    depth = np.zeros(n_bus, dtype=int)
    nf = max(1, min(n_feeders, n_bus - 1))

    tips, members = [], []
    for f in range(nf):
        b = 1 + f
        parents[b], feeder[b], depth[b] = 0, f, 1
        tips.append(b)
        members.append([b])

    for b in range(1 + nf, n_bus):
        f = (b - 1 - nf) % nf
        if rng.random() < p_branch and len(members[f]) > 2:
            par = int(rng.choice(members[f][:-1]))
        else:
            par = tips[f]
        parents[b], feeder[b], depth[b] = par, f, depth[par] + 1
        members[f].append(b)
        if par == tips[f]:
            tips[f] = b
    return parents, feeder, depth


def hop_distance(parents: np.ndarray) -> np.ndarray:
    n = len(parents)
    paths = []
    for b in range(n):
        p, x = [], b
        while x != -1:
            p.append(x)
            x = parents[x]
        paths.append(p[::-1])
    D = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        ai = paths[i]
        for j in range(i + 1, n):
            aj = paths[j]
            k = 0
            m = min(len(ai), len(aj))
            while k < m and ai[k] == aj[k]:
                k += 1
            D[i, j] = D[j, i] = (len(ai) - k) + (len(aj) - k)
    return D


def make_ybus(parents, r_ohm, x_ohm, length_km, n_bus) -> np.ndarray:
    z_base = VN_KV ** 2 / SN_MVA
    z_pu = complex(r_ohm, x_ohm) * length_km / z_base
    y = 1.0 / z_pu
    Y = np.zeros((n_bus, n_bus), dtype=np.complex128)
    for b in range(1, n_bus):
        p = parents[b]
        Y[b, b] += y
        Y[p, p] += y
        Y[b, p] -= y
        Y[p, b] -= y
    return Y


def place_pv(n_bus, n_pv, strategy, seed, parents, feeder, depth, D=None):
    if n_pv <= 0:
        return np.array([], dtype=int)
    cand = np.arange(1, n_bus)
    n_pv = int(min(n_pv, len(cand)))

    if strategy == "random":
        rng = np.random.default_rng(5000 + seed)
        return np.sort(rng.choice(cand, size=n_pv, replace=False))

    if strategy == "clustered":
        picked: list[int] = []
        for f in range(int(feeder.max()) + 1):
            mem = sorted([int(b) for b in cand if feeder[b] == f],
                         key=lambda b: depth[b])
            need = n_pv - len(picked)
            picked += mem[-need:] if need < len(mem) else mem
            if len(picked) >= n_pv:
                break
        return np.sort(np.array(picked[:n_pv], dtype=int))

    if strategy == "spread":
        assert D is not None
        sel = [int(cand[np.argmax(depth[cand])])]
        while len(sel) < n_pv:
            d = D[np.ix_(cand, sel)].min(axis=1).astype(float)
            d[np.isin(cand, sel)] = -1.0
            sel.append(int(cand[int(np.argmax(d))]))
        return np.sort(np.array(sel, dtype=int))

    if strategy == "leaves":
        has_child = np.zeros(n_bus, dtype=bool)
        has_child[parents[parents >= 0]] = True
        leaves = sorted([int(b) for b in cand if not has_child[b]],
                        key=lambda b: -depth[b])
        rest = sorted([int(b) for b in cand if has_child[b]],
                      key=lambda b: -depth[b])
        return np.sort(np.array((leaves + rest)[:n_pv], dtype=int))

    if strategy == "feeders":
        per = [sorted([int(b) for b in cand if feeder[b] == f],
                      key=lambda b: -depth[b])
               for f in range(int(feeder.max()) + 1)]
        out, k = [], 0
        while len(out) < n_pv:
            added = False
            for lst in per:
                if k < len(lst):
                    out.append(lst[k])
                    added = True
                    if len(out) >= n_pv:
                        break
            if not added:
                break
            k += 1
        return np.sort(np.array(out[:n_pv], dtype=int))

    raise ValueError(strategy)


# -----------------------------------------------------------------------------
# 4) X_pp-Kennzahlen (rein numerisch, ohne Solverlauf -> E4)
# -----------------------------------------------------------------------------
def xpp_metrics(X: np.ndarray) -> dict:
    n = X.shape[0]
    if n == 0:
        return dict(cond=np.nan, rho_jacobi=np.nan, min_diag_off=np.nan,
                    offdiag_share=np.nan, coupling_mean=np.nan, xkk_mean=np.nan)
    d = np.abs(np.diag(X))
    row = np.sum(np.abs(X), axis=1)
    off = row - d
    if n == 1:
        return dict(cond=1.0, rho_jacobi=0.0, min_diag_off=np.inf,
                    offdiag_share=0.0, coupling_mean=0.0, xkk_mean=float(d[0]))
    J = np.eye(n) - np.diag(1.0 / np.maximum(d, 1e-15)) @ X
    C = X / np.sqrt(np.outer(np.maximum(d, 1e-15), np.maximum(d, 1e-15)))
    iu = np.triu_indices(n, 1)
    return dict(
        cond=float(np.linalg.cond(X)),
        rho_jacobi=float(np.max(np.abs(np.linalg.eigvals(J)))),
        min_diag_off=float(np.min(d / np.maximum(off, 1e-15))),
        offdiag_share=float(np.mean(off / np.maximum(row, 1e-15))),
        coupling_mean=float(np.mean(np.abs(C[iu]))),
        xkk_mean=float(np.mean(d)),
    )


# -----------------------------------------------------------------------------
# 5) Netzaufbau inklusive Lastkalibrierung und Sollwertdefinition
# -----------------------------------------------------------------------------
CACHE: dict[str, dict] = {"topo": {}, "dist": {}, "scale": {}}


def _base_pattern(case: Case):
    key = (case.n_bus, case.seed, case.n_feeders)
    if key not in CACHE["topo"]:
        parents, feeder, depth = build_topology(case.n_bus, case.seed, case.n_feeders)
        rng = np.random.default_rng(2000 + case.seed)
        p_load = rng.uniform(0.5, 1.5, case.n_bus)
        p_load[0] = 0.0
        CACHE["topo"][key] = (parents, feeder, depth, p_load)
    return CACHE["topo"][key]


def _dist(case: Case, parents):
    key = (case.n_bus, case.seed, case.n_feeders)
    if key not in CACHE["dist"]:
        CACHE["dist"][key] = hop_distance(parents)
    return CACHE["dist"][key]


def _solve_pq(nd: SimpleNetworkData, s_batch, tol=1e-10, max_iter=400):
    sol = TPFDensePVMethodA(tol=tol, max_iter_inner=max_iter)
    res = sol.solve_batch(nd, s_batch)
    return res, sol.pv_info


def _load_scale(case: Case, parents, p_load, Y):
    """Bisektion, sodass v_min(lam=1) = V_MIN_TARGET (betriebspunktnormiert)."""
    key = (case.n_bus, case.seed, case.n_feeders, round(case.z0, 6),
           round(case.length_km, 6))
    if key in CACHE["scale"]:
        return CACHE["scale"][key]

    tanphi = np.tan(np.arccos(COSPHI_LOAD))
    nd0 = SimpleNetworkData(Y[1:, 1:], Y[1:, :1], Y[:1, 1:], Y[:1, :1],
                            np.array([1.0 + 0j]),
                            np.zeros(case.n_bus - 1, dtype=complex))

    def vmin(scale):
        s = scale * p_load[1:] * (1.0 + 1j * tanphi) / SN_MVA
        res, _ = _solve_pq(nd0, s.reshape(-1, 1))
        if not res.converged:
            return -1.0
        return float(np.abs(res.voltages).min())

    lo, hi = 1e-5, 1e-5
    while vmin(hi) > V_MIN_TARGET and hi < 1e4:
        hi *= 2.0
    for _ in range(45):
        mid = np.sqrt(lo * hi)
        v = vmin(mid)
        if v > V_MIN_TARGET:
            lo = mid
        else:
            hi = mid
    CACHE["scale"][key] = lo
    return lo


def build_case(case: Case):
    """Liefert (nd, pp_net, info) fuer einen Fall."""
    parents, feeder, depth, p_load = _base_pattern(case)
    r_ohm, x_ohm = line_rx(case.z0, case.rho)
    Y = make_ybus(parents, r_ohm, x_ohm, case.length_km, case.n_bus)
    scale = _load_scale(case, parents, p_load, Y)

    tanphi = np.tan(np.arccos(COSPHI_LOAD))
    p_l = case.lam * scale * p_load / SN_MVA          # p.u., Verbrauch positiv
    q_l = p_l * tanphi

    D = _dist(case, parents) if case.placement == "spread" else None
    pv_bus = place_pv(case.n_bus, case.n_pv, case.placement, case.seed,
                      parents, feeder, depth, D)
    pv_idx = pv_bus - 1                                # Index im d-Block
    p_pv = np.zeros(case.n_bus)
    if case.n_pv > 0 and case.pv_p_factor != 0.0:
        p_pv[pv_bus] = case.pv_p_factor * p_l[pv_bus]

    Y_dd, Y_ds, Y_sd, Y_ss = Y[1:, 1:], Y[1:, :1], Y[:1, 1:], Y[:1, :1]
    v_s = np.array([1.0 + 0j])
    s_d = (p_l[1:] - p_pv[1:]) + 1j * q_l[1:]

    # Basisloesung (Q_pv = 0) -> Sollwerte, v_min_base, eta_pq
    nd_pq = SimpleNetworkData(Y_dd, Y_ds, Y_sd, Y_ss, v_s, s_d.copy())
    res_pq, info_pq = _solve_pq(nd_pq, s_d.reshape(-1, 1))
    v_base = np.abs(res_pq.voltages[:, 0]) if res_pq.converged else np.full(len(s_d), np.nan)

    if case.n_pv > 0:
        if case.sp_mode == "delta":
            v_spec = v_base[pv_idx] + case.delta
        else:
            v_spec = np.full(case.n_pv, case.v_abs)
    else:
        v_spec = None

    q_lim = None
    if case.q_lim_pu is not None and case.n_pv > 0:
        q_lim = np.full(case.n_pv, float(case.q_lim_pu))

    nd = SimpleNetworkData(
        Y_dd, Y_ds, Y_sd, Y_ss, v_s, s_d.copy(),
        pv_indices=pv_idx if case.n_pv > 0 else None,
        pv_v_setpoint=v_spec,
        pv_q_min=None if q_lim is None else -q_lim,
        pv_q_max=q_lim,
    )

    X_pp = (np.imag(np.linalg.inv(Y_dd))[np.ix_(pv_idx, pv_idx)]
            if case.n_pv > 0 else np.zeros((0, 0)))

    # pandapower-Referenznetz
    net = pp.create_empty_network(sn_mva=SN_MVA)
    for _ in range(case.n_bus):
        pp.create_bus(net, vn_kv=VN_KV)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)
    for b in range(1, case.n_bus):
        pp.create_line_from_parameters(
            net, from_bus=int(parents[b]), to_bus=b, length_km=case.length_km,
            r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=0.0, max_i_ka=10.0)
    for b in range(1, case.n_bus):
        pp.create_load(net, bus=b, p_mw=p_l[b] * SN_MVA, q_mvar=q_l[b] * SN_MVA)
    for i, b in enumerate(pv_bus):
        pp.create_gen(net, bus=int(b), p_mw=p_pv[b] * SN_MVA,
                      vm_pu=float(v_spec[i]), slack=False)

    info = dict(
        parents=parents, feeder=feeder, depth=depth, pv_bus=pv_bus, pv_idx=pv_idx,
        X_pp=X_pp, v_base=v_base, v_min_base=float(np.nanmin(v_base)),
        eta_pq=eta_from_history(info_pq.inner_v_change_all),
        k_in_pq=info_pq.inner_iterations_total, load_scale=scale,
        r_ohm=r_ohm, x_ohm=x_ohm, p_l=p_l, q_l=q_l, p_pv=p_pv,
        depth_pv_mean=float(np.mean(depth[pv_bus])) if case.n_pv else np.nan,
    )
    return nd, net, info


def eta_from_history(hist, nmax=12) -> float:
    h = np.asarray([v for v in hist if np.isfinite(v) and v > 1e-13], dtype=float)
    if len(h) < 4:
        return np.nan
    h = h[-nmax:]
    k = np.arange(len(h))
    sl = np.polyfit(k, np.log(h), 1)[0]
    return float(np.exp(sl))


# -----------------------------------------------------------------------------
# 6) Ein Fall rechnen
# -----------------------------------------------------------------------------
def make_s_batch(nd, info, case, tau, seed=7):
    s0 = nd.s_nom
    if tau == 1:
        return s0.reshape(-1, 1)
    rng = np.random.default_rng(seed)
    prof = 0.6 + 0.8 * rng.random(tau)
    return s0.reshape(-1, 1) * prof.reshape(1, -1)


def run_case(case: Case, repeats: int = 3, with_nr: bool = True,
             max_outer: int = 60, max_inner: int = 200) -> dict:
    nd, net, info = build_case(case)
    X_pp = info["X_pp"]
    m = xpp_metrics(X_pp)

    row = dict(
        n_bus=case.n_bus, seed=case.seed, n_pv=case.n_pv, share=case.share,
        placement=case.placement, variant=case.variant, lam=case.lam,
        rho=case.rho, sp_mode=case.sp_mode, delta=case.delta, tau=case.tau,
        q_lim_pu=(np.nan if case.q_lim_pu is None else case.q_lim_pu),
        pv_p_factor=case.pv_p_factor,
        v_min_base=info["v_min_base"], eta_pq=info["eta_pq"],
        k_in_pq=info["k_in_pq"], depth_pv_mean=info["depth_pv_mean"],
        r_ohm=info["r_ohm"], x_ohm=info["x_ohm"], **m,
    )

    kw = dict(tol=1e-8, max_iter_inner=max_inner, max_iter_outer=max_outer,
              tol_pv=1e-6, omega=1.0, adaptive_inner=True, cold_start=False,
              use_decoupled=(case.variant == "decoupled"),
              enforce_q_lims=(case.q_lim_pu is not None))

    s_batch = make_s_batch(nd, info, case, case.tau)

    # --- Zeitmessung / Batch-Kennzahlen ueber solve_timeseries ---
    best = None
    for _ in range(max(1, repeats)):
        sol = TPFDensePVMethodA(**kw)
        t0 = time.perf_counter()
        res = sol.solve_timeseries(nd, s_batch, warm_mode="flat",
                                   diagnostics=False, verbose=False)
        wall = (time.perf_counter() - t0) * 1e3
        pi = sol.pv_info
        cand = (pi.t_solve_ms, wall, pi, res)
        if best is None or cand[0] < best[0]:
            best = cand
    t_solve_ms, t_wall_ms, pi, res = best

    row.update(
        converged=bool(res.converged),
        conv_share=pi.n_converged_scenarios / max(1, pi.n_scenarios),
        k_out=int(pi.outer_iterations),
        k_in=int(pi.inner_iterations_total),
        k_ratio=(pi.inner_iterations_total / pi.outer_iterations
                 if pi.outer_iterations else np.nan),
        t_pre_ms=pi.t_precompute_ms, t_solve_ms=t_solve_ms, t_wall_ms=t_wall_ms,
        t_per_scen_ms=t_solve_ms / max(1, case.tau),
        gflops=(pi.flops_gemm / (t_solve_ms * 1e-3) / 1e9 if t_solve_ms > 0 else np.nan),
        n_gemm=pi.n_gemm, pv_v_err=float(pi.pv_v_error_final),
        v_min=float(np.nanmin(pi.v_min_per_scenario)) if pi.v_min_per_scenario is not None else np.nan,
        v_max=float(np.nanmax(pi.v_max_per_scenario)) if pi.v_max_per_scenario is not None else np.nan,
    )

    # --- Q-Statistik (Injektionskonvention) und Saettigung ---
    q_inj = np.zeros((0, 1))
    if case.n_pv > 0 and pi.pv_q_final is not None and pi.pv_q_final.size:
        q_inj = -np.asarray(pi.pv_q_final).reshape(case.n_pv, -1)
        row.update(q_max=float(np.max(np.abs(q_inj))),
                   q_med=float(np.median(np.abs(q_inj))),
                   q_sum=float(np.mean(np.sum(np.abs(q_inj), axis=0))))
        if case.q_lim_pu is not None:
            row["sat_share"] = float(np.mean(np.abs(q_inj) >= 0.999 * case.q_lim_pu))
        else:
            row["sat_share"] = np.nan
    else:
        row.update(q_max=np.nan, q_med=np.nan, q_sum=np.nan, sat_share=np.nan)

    # --- tau = 1: Historien, eta, eps_lin ---
    row.update(eta_final=np.nan, eps_lin_meas=np.nan, eps_lin_pred=np.nan,
               hist_err="", hist_kin="")
    if case.tau == 1:
        sol1 = TPFDensePVMethodA(**kw)
        sol1.solve_batch(nd, s_batch)
        p1 = sol1.pv_info
        row["eta_final"] = eta_from_history(
            p1.inner_v_change_all[(p1.outer_start_indices or [0])[-1]:])
        if p1.pv_v_error_history:
            row["hist_err"] = ";".join(f"{v:.6e}" for v in p1.pv_v_error_history)
        if p1.inner_iterations_per_outer:
            row["hist_kin"] = ";".join(str(int(v)) for v in p1.inner_iterations_per_outer)

        if case.n_pv > 0 and q_inj.size:
            dv_act = nd.pv_v_setpoint ** 2 - info["v_base"][info["pv_idx"]] ** 2
            dv_pred = 2.0 * X_pp @ q_inj[:, 0]
            den = max(np.max(np.abs(dv_act)), 1e-15)
            row["eps_lin_meas"] = float(np.max(np.abs(dv_pred - dv_act)) / den)
            row["eps_lin_pred"] = float(
                np.max((1 + case.rho ** 2) * np.abs(dv_act)
                       / (4.0 * nd.pv_v_setpoint ** 2)))

    # --- NR-Referenz ---
    row.update(nr_conv=np.nan, nr_iter=np.nan, t_nr_ms=np.nan,
               err_vm=np.nan, err_va_deg=np.nan, err_q_pu=np.nan)
    if with_nr and case.tau == 1:
        nr = PandapowerNRSolver(tol=1e-10, max_iter=100)
        try:
            rnr = nr.solve_from_net(net)
            row["nr_conv"] = bool(rnr.converged)
            row["nr_iter"] = int(rnr.iterations)
            row["t_nr_ms"] = rnr.elapsed_time_s * 1e3
            if rnr.converged and res.converged:
                v_nr = np.asarray(rnr.voltages).reshape(-1)[1:]
                v_tpf = res.voltages[:, 0]
                row["err_vm"] = float(np.max(np.abs(np.abs(v_nr) - np.abs(v_tpf))))
                row["err_va_deg"] = float(np.max(np.abs(
                    np.rad2deg(np.angle(v_nr) - np.angle(v_tpf)))))
                if case.n_pv > 0 and rnr.pv_q_pu is not None and q_inj.size:
                    bus_of = {int(b): i for i, b in enumerate(rnr.pv_indices)}
                    q_ref = np.array([rnr.pv_q_pu[bus_of[int(b)]]
                                      for b in info["pv_bus"] if int(b) in bus_of])
                    if len(q_ref) == case.n_pv:
                        row["err_q_pu"] = float(np.max(np.abs(q_ref - q_inj[:, 0])))
        except Exception as e:  # pragma: no cover
            row["nr_conv"] = False
            row["nr_err"] = str(e)[:120]
    return row


# -----------------------------------------------------------------------------
# 7) Experimente
# -----------------------------------------------------------------------------
def npv_grid(n_bus: int, quick: bool) -> list[int]:
    """Absolute Stuetzstellen (H1) plus feste Anteile (H1/E2-Kollapstest)."""
    absolute = [1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 40]
    shares = [0.05, 0.10, 0.25, 0.50, 0.75, 0.99]
    g = set(a for a in absolute if a <= n_bus - 1)
    g |= set(max(1, int(round(s * (n_bus - 1)))) for s in shares)
    g.add(n_bus - 1)
    out = sorted(g)
    if quick:
        out = out[::2]
    return out


def exp_e1(quick: bool) -> pd.DataFrame:
    sizes = [40, 200] if quick else [40, 200, 500]
    seeds = range(2) if quick else range(5)
    rows = []
    for n in sizes:
        for npv in npv_grid(n, quick):
            for sd in seeds:
                for var in ("coupled", "decoupled"):
                    c = Case(n_bus=n, seed=sd, n_pv=npv, variant=var)
                    rows.append({**run_case(c, repeats=2,
                                            with_nr=(var == "coupled" and sd == 0)),
                                 "exp": "E1"})
                    print(f"  E1 n={n:4d} n_pv={npv:4d} seed={sd} {var:10s} "
                          f"k_out={rows[-1]['k_out']:3d} conv={rows[-1]['converged']}")
    return pd.DataFrame(rows)


def exp_e1b(quick: bool) -> pd.DataFrame:
    """Kontrolle Stoergroesse Lastbilanz: PV mit / ohne Wirkleistungseinspeisung."""
    rows = []
    for npv in npv_grid(200, quick):
        for f in (0.0, 2.0):
            c = Case(n_bus=200, seed=0, n_pv=npv, pv_p_factor=f)
            rows.append({**run_case(c, repeats=1, with_nr=False), "exp": "E1b"})
    return pd.DataFrame(rows)


def exp_e3(quick: bool) -> tuple[pd.DataFrame, dict]:
    strategies = ["random", "clustered", "spread", "leaves", "feeders"]
    sizes = [200] if quick else [40, 200, 500]
    rows, mats = [], {}
    for n in sizes:
        grid = [4, 8, 12, 20, 40] if not quick else [8, 20]
        for npv in [g for g in grid if g <= n - 1]:
            for st in strategies:
                for var in ("coupled", "decoupled"):
                    c = Case(n_bus=n, seed=0, n_pv=npv, placement=st, variant=var)
                    rows.append({**run_case(c, repeats=1, with_nr=False), "exp": "E3"})
                if n == 200 and npv == 20:
                    _, _, inf = build_case(Case(n_bus=n, seed=0, n_pv=npv, placement=st))
                    mats[f"n{n}_npv{npv}_{st}"] = inf["X_pp"]
            print(f"  E3 n={n} n_pv={npv} done")
    return pd.DataFrame(rows), mats


def exp_e4(quick: bool) -> pd.DataFrame:
    """Rein numerisch: Struktur von X_pp ohne Solverlauf."""
    rows = []
    sizes = [40, 200] if quick else [40, 120, 200, 500, 1000]
    for n in sizes:
        parents, feeder, depth = build_topology(n, 0, 4)
        r, x = line_rx(Z0_REF, RHO_REF)
        Zim = np.imag(np.linalg.inv(make_ybus(parents, r, x, LEN_KM, n)[1:, 1:]))
        D = hop_distance(parents)
        grid = sorted(set([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96,
                           128, 192, 256, 384, n - 1]))
        for npv in [g for g in grid if 1 <= g <= n - 1]:
            for st in ["random", "clustered", "spread", "leaves", "feeders"]:
                for sd in (range(1) if st != "random" else range(5)):
                    idx = place_pv(n, npv, st, sd, parents, feeder, depth, D) - 1
                    m = xpp_metrics(Zim[np.ix_(idx, idx)])
                    rows.append(dict(exp="E4", n_bus=n, n_pv=npv, seed=sd,
                                     placement=st, share=npv / (n - 1), **m))
    return pd.DataFrame(rows)


def exp_e5(quick: bool) -> pd.DataFrame:
    rows = []
    deltas = [0.002, 0.005, 0.02] if quick else [0.002, 0.005, 0.01, 0.02, 0.05]
    grid = npv_grid(200, quick)
    for npv in grid:
        for d in deltas:
            rows.append({**run_case(Case(n_bus=200, seed=0, n_pv=npv,
                                         sp_mode="delta", delta=d),
                                    repeats=1, with_nr=False), "exp": "E5"})
        rows.append({**run_case(Case(n_bus=200, seed=0, n_pv=npv,
                                     sp_mode="abs", v_abs=1.00),
                                repeats=1, with_nr=False), "exp": "E5"})
    return pd.DataFrame(rows)


def exp_e6(quick: bool) -> pd.DataFrame:
    rows = []
    lims = [None, 0.33, 0.10]
    for n in ([200] if quick else [40, 200, 500]):
        for npv in npv_grid(n, quick):
            for q in lims:
                rows.append({**run_case(Case(n_bus=n, seed=0, n_pv=npv, q_lim_pu=q),
                                        repeats=1, with_nr=False), "exp": "E6"})
    return pd.DataFrame(rows)


def exp_e7(quick: bool) -> pd.DataFrame:
    rows = []
    taus = [1, 100, 1000] if quick else [1, 10, 100, 1000, 5000]
    for n in ([40, 200] if quick else [40, 200, 500]):
        for share in [0.0, 0.10, 0.30, 0.50]:
            npv = int(round(share * (n - 1)))
            for tau in taus:
                rows.append({**run_case(Case(n_bus=n, seed=0, n_pv=npv, tau=tau),
                                        repeats=1, with_nr=False), "exp": "E7"})
                print(f"  E7 n={n} n_pv={npv} tau={tau} "
                      f"t/scen={rows[-1]['t_per_scen_ms']:.4f} ms")
    return pd.DataFrame(rows)


def exp_e8(quick: bool) -> pd.DataFrame:
    rows = []
    shares = [0.0, 0.05, 0.10, 0.25, 0.50] if quick else [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.99]
    lams = [0.5, 1, 2, 4, 6, 8] if quick else [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rhos = [0.1, 1.0, 7.73] if quick else [0.1, 0.32, 1.0, 2.15, 4.64, 7.73, 10.0]
    n = 200
    for share in shares:
        npv = int(round(share * (n - 1)))
        for lam in lams:
            rows.append({**run_case(Case(n_bus=n, seed=0, n_pv=npv, lam=lam),
                                    repeats=1, with_nr=False),
                         "exp": "E8", "axis": "lam"})
        for rho in rhos:
            rows.append({**run_case(Case(n_bus=n, seed=0, n_pv=npv, rho=rho),
                                    repeats=1, with_nr=False),
                         "exp": "E8", "axis": "rho"})
        print(f"  E8 share={share} done")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 8) main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_pv_share")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(a.lower() for a in args.only) if args.only else None

    def go(name, fn):
        if want and name not in want:
            return
        print(f"\n=== {name.upper()} ===")
        t0 = time.perf_counter()
        r = fn(args.quick)
        if isinstance(r, tuple):
            df, mats = r
            np.savez_compressed(out / "xpp_matrices.npz", **mats)
        else:
            df = r
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"--> {name}.csv  ({len(df)} Zeilen, {time.perf_counter()-t0:.1f} s)")

    go("e1", exp_e1)
    go("e1b", exp_e1b)
    go("e3", exp_e3)
    go("e4", exp_e4)
    go("e5", exp_e5)
    go("e6", exp_e6)
    go("e7", exp_e7)
    go("e8", exp_e8)

    (out / "meta.json").write_text(json.dumps(dict(
        quick=args.quick, python=platform.python_version(),
        numpy=np.__version__, pandapower=pp.__version__,
        machine=platform.machine(), processor=platform.processor(),
        v_min_target=V_MIN_TARGET, cosphi_load=COSPHI_LOAD,
        z0_ref=Z0_REF, rho_ref=RHO_REF, length_km=LEN_KM,
        tol_pv=1e-6, tol_inner=1e-8, omega=1.0,
        note="PV-Knoten mit P_gen=0 (pv_p_factor=0) -> Lastbilanz unabhaengig von n_pv",
    ), indent=2), encoding="utf-8")
    print("\nfertig.")


if __name__ == "__main__":
    main()
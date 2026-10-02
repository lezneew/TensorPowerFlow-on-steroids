#!/usr/bin/env python3
# tau_sweep.py
"""
Tau-Sweep: Batch-Dimension des Tensor Power Flow mit PV-Knoten, inklusive
Stresstest ueber Lastfaktor und R/X-Verhaeltnis.

Ersetzt die inkonsistenten Vorgaenger-Sweeps aus dem Batch-/Mehrere-Lastfluesse-
Kapitel. Netzsynthese identisch zur Logik aus lambda_sweep.py (radialer Baum,
betriebspunktnormierte Lastkalibrierung v_min(lambda=1)=0.97 p.u.). Der Solver
ist durchgaengig TPFDensePVMethodA in der im Optimierungskapitel begruendeten
Standardkonfiguration (gekoppelte Q-Korrektur, innerer Warm Start, adaptive
innere Toleranz) -- diese wird in diesem Sweep NICHT variiert. Batch-Init
ausschliesslich 'flat'.

Stressgitter (5 distinkte Punkte, keine 3x3-Faktorstruktur):
    axis="base"   level=0  lam=1.0   rho=1.0    (Basis, in beiden Achsen enthalten)
    axis="lambda" level=1  lam=5.0   rho=1.0
    axis="lambda" level=2  lam=9.0   rho=1.0
    axis="rho"    level=1  lam=1.0   rho=3.16
    axis="rho"    level=2  lam=1.0   rho=7.73

Fuer eine vollstaendige 3-Punkte-Kurve je Achse beim Auswerten filtern:
    lambda-Achse: df[(df.stress_axis=="base") | (df.stress_axis=="lambda")]
    rho-Achse:    df[(df.stress_axis=="base") | (df.stress_axis=="rho")]

Output:
    <out>/tau_sweep_main.csv   -- ein Zeile je (n_bus, pv_share, stress, tau)
    <out>/tau_sweep_nr.csv     -- ein Zeile je (n_bus, pv_share, stress),
                                  NR-Referenz unabhaengig von tau
    <out>/meta.json            -- Konfiguration, Overhead-Kalibrierung, Laufzeit

Wichtige Spalten in tau_sweep_main.csv:
    n_bus, pv_share, n_pv, tau, stress_axis, stress_level, lam_ref, rho
    lam_batch_min/max          -- tatsaechlicher Ensemble-Bereich (Jitter-Kontrolle)
    v_min_base, load_scale     -- Arbeitspunkt-Kenngroessen des Basisnetzes
    t_pre_ms_min/median        -- Vorberechnung (Z_B, PV-Sensitivitaet, ...)
    t_solve_ms_min/median      -- reine Iterationszeit (innerer solver.pv_info)
    t_wall_ms_min/median       -- externe Python-Wallclock inkl. Overhead
    t_total_ms_min/median      -- = t_pre + t_solve  ("Gesamtlaufzeit")
    t_per_scen_ms_min/median   -- t_total / tau
    k_out, k_in                -- Batch-Gesamtiterationen (aeussere/innere Schleife)
    converged, conv_share      -- Batch- bzw. Spalten-Konvergenz
    v_min_batch, v_max_batch   -- ueber alle Szenarien des Batches
    q_max_batch, q_med_batch   -- Blindleistung an PV-Knoten (keine Q-Limits!)
    gflops                     -- erreichte Gleitkommaleistung im dichten Produkt
    mem_est_gb, skip_reason    -- Speicher-/Zeit-Cap-Diagnose
    t_nr_total_ms_min/median   -- NR-Referenz hochskaliert auf denselben tau
    speedup_raw, speedup_corrected -- roh bzw. overhead-korrigiert

Aufruf:
    python tau_sweep.py --mode quick    --out results_tau_quick
    python tau_sweep.py --mode thorough --out results_tau_thorough --resume
    python tau_sweep.py --mode thorough --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandapower as pp
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver


# ═════════════════════════════════════════════════════════════════════════════
# 0) Konfiguration
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SweepConfig:
    mode: str = "quick"
    out: Path = field(default_factory=lambda: Path("results_tau"))
    seed: int = 0
    branch_span: int = 3
    vn_kv: float = 0.4
    s_base_mva: float = 1.0
    len_km: float = 0.02
    z0_ohm_km: float = 0.6597          # NAYY 4x50 SE, |z|-Anker (const_z)
    cos_phi: float = 0.95
    v_min_target_lam1: float = 0.97
    pv_p_total_ratio: float = 0.30
    dv_setpoint: float = 0.005
    jitter_lambda: float = 0.15        # additiver Ensemble-Spread [lambda-Einheiten]
    tol_inner: float = 1e-6
    tol_pv: float = 1e-6
    max_inner: int = 300
    max_outer: int = 60
    mem_cap_gb: float = 20.0
    mem_factor: float = 6.0            # Sicherheitsfaktor: geschaetzte gleichzeitig
                                        # lebende (n_bus x tau)-Arrays
    repeats: int = 3
    time_cap_s: float = 300.0
    nr_tau_ref: int = 200
    nr_overhead_reps: int = 50
    checkpoint: int = 1
    max_total_hours: float | None = None


# 5 distinkte Stresspunkte (kein 3x3-Faktorgitter, siehe Modulkopf)
STRESS_CONFIGS: list[dict] = [
    dict(axis="base",   level=0, lam=1.0, rho=1.0),
    dict(axis="lambda", level=1, lam=5.0, rho=1.0),
    dict(axis="lambda", level=2, lam=9.0, rho=1.0),
    dict(axis="rho",    level=1, lam=1.0, rho=3.16),
    dict(axis="rho",    level=2, lam=1.0, rho=7.73),
]


def grid_lists_for_mode(mode: str) -> dict:
    if mode == "quick":
        return dict(
            n_bus=[40, 200, 500, 1000, 1500],
            pv_share=[0.0, 0.50],
            tau=[1, 10, 1_000, 10_000],
        )
    if mode == "thorough":
        return dict(
            n_bus=[20, 40, 75, 120, 200, 350, 500, 750, 1000, 1500],
            pv_share=[0.0, 0.30, 0.50],
            tau=[1, 10, 100, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
        )
    raise ValueError(f"unbekannter Modus: {mode}")


def defaults_for_mode(mode: str) -> dict:
    if mode == "quick":
        return dict(repeats=3, time_cap_s=300.0, nr_tau_ref=200)
    return dict(repeats=5, time_cap_s=1800.0, nr_tau_ref=500)


def make_seed(*parts) -> int:
    """Deterministischer Seed unabhaengig von PYTHONHASHSEED (Teile koennen Strings sein)."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# ═════════════════════════════════════════════════════════════════════════════
# 1) Netzsynthese (identische Logik zu lambda_sweep.py)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Grid:
    n: int
    rho: float
    parents: np.ndarray
    Y_dd: np.ndarray
    Y_ds: np.ndarray
    Y_sd: np.ndarray
    Y_ss: np.ndarray
    K: np.ndarray
    L: np.ndarray
    p_prof: np.ndarray
    q_prof: np.ndarray
    load_scale: float
    r_km: float
    x_km: float
    net: object


def fpi_ref(K, L, s, tol, max_iter, V0=None):
    """Referenz-FPI (tau=1), fuer Kalibrierung und Sollwertbestimmung."""
    V = np.ones(K.shape[0], dtype=complex) if V0 is None else V0.copy()
    Sc = np.conj(s)
    for k in range(1, max_iter + 1):
        Vn = K @ (Sc / np.conj(V)) + L
        if not np.all(np.isfinite(Vn)):
            return V, k, False
        if np.max(np.abs(np.abs(Vn) - np.abs(V))) < tol:
            return Vn, k, True
        V = Vn
    return V, max_iter, False


def line_rx(z0: float, rho: float) -> tuple[float, float]:
    """Modus const_z: |z| fest, nur der Impedanzwinkel wird gedreht."""
    x = z0 / np.sqrt(1.0 + rho ** 2)
    return rho * x, x


_GRID_CACHE: dict = {}


def build_grid(cfg: SweepConfig, n: int, rho: float) -> Grid:
    key = (n, round(rho, 6), cfg.seed)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    rng = np.random.default_rng(1000 * cfg.seed + n)
    r_km, x_km = line_rx(cfg.z0_ohm_km, rho)

    parents = np.zeros(n, dtype=int)
    for i in range(n):
        parents[i] = 0 if i == 0 else int(rng.integers(max(0, i - cfg.branch_span), i + 1))

    z_base = cfg.vn_kv ** 2 / cfg.s_base_mva
    y = 1.0 / ((r_km + 1j * x_km) * cfg.len_km / z_base)
    Y = np.zeros((n + 1, n + 1), dtype=complex)
    for i in range(n):
        a, b = parents[i], i + 1
        Y[a, a] += y; Y[b, b] += y; Y[a, b] -= y; Y[b, a] -= y

    Y_dd, Y_ds, Y_sd, Y_ss = Y[1:, 1:], Y[1:, :1], Y[:1, 1:], Y[:1, :1]
    Z_B = np.linalg.inv(Y_dd)
    K = -Z_B
    L = (K @ Y_ds @ np.array([1.0 + 0j])).ravel()

    tan_phi = np.tan(np.arccos(cfg.cos_phi))
    p_prof = np.ones(n)
    q_prof = tan_phi * p_prof

    def vmin_of(scale: float) -> float:
        s = scale * (p_prof + 1j * q_prof)
        V, _, ok = fpi_ref(K, L, s, 1e-10, 400)
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

    net = _build_pp_net(cfg, n, parents, r_km, x_km)
    g = Grid(n=n, rho=rho, parents=parents, Y_dd=Y_dd, Y_ds=Y_ds, Y_sd=Y_sd,
             Y_ss=Y_ss, K=K, L=L, p_prof=p_prof, q_prof=q_prof,
             load_scale=scale, r_km=r_km, x_km=x_km, net=net)
    _GRID_CACHE[key] = g
    return g


def _build_pp_net(cfg: SweepConfig, n: int, parents: np.ndarray, r_km: float, x_km: float):
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


# ═════════════════════════════════════════════════════════════════════════════
# 2) PV-Fall: Platzierung + Sollwertkalibrierung
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Case:
    grid: Grid
    pv_share: float
    pv_idx: np.ndarray
    p_pv: np.ndarray
    v_spec: np.ndarray | None
    v_min_base: float

    @property
    def n(self) -> int:
        return self.grid.n

    @property
    def n_pv(self) -> int:
        return len(self.pv_idx)


_CASE_CACHE: dict = {}


def make_case(cfg: SweepConfig, grid: Grid, pv_share: float, ref_lambda: float) -> Case:
    """PV-Platzierung + Sollwerte, kalibriert bei ref_lambda. Der Cache ist
    innerhalb eines Sweep-Laufs sicher, da jede (n_bus, pv_share, stress)-
    Kombination genau einmal besucht wird (siehe Modulkopf)."""
    key = (id(grid), round(pv_share, 6), round(ref_lambda, 6))
    if key in _CASE_CACHE:
        return _CASE_CACHE[key]

    n = grid.n
    n_pv = int(round(pv_share * n))            # Konvention wie lambda_sweep.py: Anteil von n
    if n_pv == 0:
        pv_idx, p_pv = np.array([], dtype=int), np.array([])
    else:
        pv_idx = np.unique(np.linspace(0, n - 1, n_pv).round().astype(int))
        p_tot = cfg.pv_p_total_ratio * float((grid.load_scale * grid.p_prof).sum())
        p_pv = np.full(len(pv_idx), p_tot / len(pv_idx))

    if len(grid.net.gen):
        grid.net.gen.drop(grid.net.gen.index, inplace=True)
    for k, b in enumerate(pv_idx):
        pp.create_gen(grid.net, bus=int(b) + 1, p_mw=0.0, vm_pu=1.0,
                      name=f"pv{k}", slack=False)

    s_ref = ref_lambda * grid.load_scale * (grid.p_prof + 1j * grid.q_prof)
    if n_pv:
        s_ref = s_ref.copy()
        s_ref[pv_idx] -= p_pv

    V, _, ok = fpi_ref(grid.K, grid.L, s_ref, 1e-10, 400)
    vmag = np.abs(V) if ok else np.full(n, np.nan)
    v_spec = (vmag[pv_idx] + cfg.dv_setpoint) if n_pv else None

    case = Case(grid=grid, pv_share=pv_share, pv_idx=pv_idx, p_pv=p_pv,
                v_spec=v_spec, v_min_base=float(np.nanmin(vmag)))
    _CASE_CACHE[key] = case
    return case


def s_of_lambda(case: Case, lam: float) -> np.ndarray:
    g = case.grid
    s = lam * g.load_scale * (g.p_prof + 1j * g.q_prof)
    if case.n_pv:
        s = s.copy()
        s[case.pv_idx] -= case.p_pv
    return s


def make_batch(case: Case, target_lambda: float, tau: int,
                jitter: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """S in C^{n x tau}, vektorisiert (wichtig fuer tau bis 1e6).
    tau=1: exakt target_lambda (kein Jitter). tau>1: additiver Jitter in
    lambda-Einheiten um target_lambda (kein multiplikativer Jitter, um bei
    hohen Stress-Lambdas nicht ueber lambda* hinauszulaufen)."""
    g = case.grid
    if tau == 1:
        lam = np.array([target_lambda])
    else:
        rng = np.random.default_rng(seed)
        lam = target_lambda + jitter * (2.0 * rng.random(tau) - 1.0)
        lam = np.maximum(lam, 0.05)
    base = g.load_scale * (g.p_prof + 1j * g.q_prof)      # shape (n,)
    S = np.outer(base, lam)                                # shape (n, tau)
    if case.n_pv:
        S[case.pv_idx, :] -= case.p_pv[:, None]
    return S, lam


# ═════════════════════════════════════════════════════════════════════════════
# 3) Duck-typed NetworkData fuer TPFDensePVMethodA
# ═════════════════════════════════════════════════════════════════════════════
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


def to_network_data(case: Case) -> SimpleNetworkData:
    g = case.grid
    return SimpleNetworkData(
        Y_dd=g.Y_dd, Y_ds=g.Y_ds, Y_sd=g.Y_sd, Y_ss=g.Y_ss,
        v_s=np.array([1.0 + 0j]),
        s_nom=s_of_lambda(case, 1.0),          # Platzhalter, pro Job durch S ersetzt
        pv_indices=case.pv_idx if case.n_pv else None,
        pv_v_setpoint=case.v_spec,
        pv_q_min=None, pv_q_max=None,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4) Speicher-/Zeit-Caps
# ═════════════════════════════════════════════════════════════════════════════
def mem_estimate_gb(n_bus: int, tau: int, mem_factor: float) -> float:
    """Grobe obere Schranke: mem_factor gleichzeitig lebende (n_bus x tau)
    complex128-Arrays (V, S, Zwischenergebnisse). PV-Arrays sind (n_pv x tau)
    mit n_pv << n_bus und daher hier vernachlaessigt."""
    return mem_factor * 16.0 * n_bus * tau / (1024 ** 3)


# ═════════════════════════════════════════════════════════════════════════════
# 5) TPF-Job (mit Repeats, Zeit-Cap, Fehlerbehandlung)
# ═════════════════════════════════════════════════════════════════════════════
def run_tpf_job(nd: SimpleNetworkData, S: np.ndarray, cfg: SweepConfig,
                repeats: int, time_cap_s: float):
    t_pre, t_solve, t_wall = [], [], []
    pv_info_first, res_first = None, None
    timed_out, error = False, ""

    for _ in range(max(1, repeats)):
        try:
            sol = TPFDensePVMethodA(
                tol=cfg.tol_inner, max_iter_inner=cfg.max_inner,
                max_iter_outer=cfg.max_outer, tol_pv=cfg.tol_pv,
                omega=1.0, cold_start=False, adaptive_inner=True,
                use_decoupled=False, enforce_q_lims=False,
            )
            t0 = time.perf_counter()
            res = sol.solve_timeseries(nd, S, warm_mode="flat",
                                       diagnostics=False, verbose=False)
            wall_ms = (time.perf_counter() - t0) * 1e3
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            break

        pi = sol.pv_info
        t_pre.append(pi.t_precompute_ms)
        t_solve.append(pi.t_solve_ms)
        t_wall.append(wall_ms)
        if pv_info_first is None:
            pv_info_first, res_first = pi, res
        if wall_ms / 1000.0 > time_cap_s:
            timed_out = True
            break

    return dict(t_pre=t_pre, t_solve=t_solve, t_wall=t_wall), pv_info_first, res_first, timed_out, error


# ═════════════════════════════════════════════════════════════════════════════
# 6) NR-Referenz (tau-unabhaengig, separates Ensemble)
# ═════════════════════════════════════════════════════════════════════════════
def calibrate_nr_overhead(cfg: SweepConfig) -> dict:
    """Einmalige Messung des reinen Aufrufoverheads (triviales 2-Bus-Netz)."""
    net = pp.create_empty_network(sn_mva=1.0)
    pp.create_bus(net, vn_kv=0.4)
    pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    pp.create_line_from_parameters(net, 0, 1, length_km=0.01,
                                    r_ohm_per_km=0.6, x_ohm_per_km=0.1,
                                    c_nf_per_km=0.0, max_i_ka=10.0)
    pp.create_load(net, 1, p_mw=0.01, q_mvar=0.003)
    times = []
    for _ in range(cfg.nr_overhead_reps):
        t0 = time.perf_counter()
        pp.runpp(net, algorithm="nr", init="flat")
        times.append((time.perf_counter() - t0) * 1e3)
    return dict(nr_overhead_ms_min=float(np.min(times)),
                nr_overhead_ms_median=float(np.median(times)))


def run_nr_reference(cfg: SweepConfig, case: Case, lam_ref: float, n_ref: int) -> dict:
    """Sequentielle NR-Referenz ueber n_ref Szenarien, gleiche Jitter-Konvention
    wie das TPF-Batch-Ensemble. tau-unabhaengig: liefert die Zeit PRO Szenario."""
    net = case.grid.net
    g = case.grid
    seed = make_seed("nr", case.n, round(case.pv_share, 6), round(lam_ref, 6))
    rng = np.random.default_rng(seed)
    lam_samples = lam_ref + cfg.jitter_lambda * (2.0 * rng.random(n_ref) - 1.0)
    lam_samples = np.maximum(lam_samples, 0.05)

    times, n_conv, iters = [], 0, []
    nr_solver = PandapowerNRSolver(tol=1e-8, max_iter=100)
    for lam in lam_samples:
        p_l = lam * g.load_scale * g.p_prof * cfg.s_base_mva
        q_l = lam * g.load_scale * g.q_prof * cfg.s_base_mva
        net.load.loc[:, "p_mw"] = p_l
        net.load.loc[:, "q_mvar"] = q_l
        if case.n_pv:
            net.gen.loc[:, "p_mw"] = case.p_pv * cfg.s_base_mva
            net.gen.loc[:, "vm_pu"] = case.v_spec

        t0 = time.perf_counter()
        ok = False
        try:
            nr_result = nr_solver.solve_from_net(net)
            ok = nr_result.converged
        except Exception:
            ok = False
        times.append((time.perf_counter() - t0) * 1e3)
        n_conv += int(ok)
        if ok:
            iters.append(nr_result.iterations)

    return dict(
        n_samples=n_ref, nr_conv_share=n_conv / max(1, n_ref),
        t_nr_ms_min=float(np.min(times)), t_nr_ms_median=float(np.median(times)),
        nr_iter_median=(float(np.median(iters)) if iters else np.nan),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7) Resume / Checkpoint
# ═════════════════════════════════════════════════════════════════════════════
def key_main(mode, n_bus, pv_share, tau, axis, level) -> tuple:
    return (str(mode), int(n_bus), round(float(pv_share), 6),
            int(tau), str(axis), int(level))


def key_nr(mode, n_bus, pv_share, axis, level) -> tuple:
    return (str(mode), int(n_bus), round(float(pv_share), 6), str(axis), int(level))


def load_resume(out: Path, resume: bool):
    rows_main, rows_nr = [], []
    done_main, done_nr = set(), set()
    if not resume:
        return rows_main, rows_nr, done_main, done_nr

    p_main, p_nr = out / "tau_sweep_main.csv", out / "tau_sweep_nr.csv"
    if p_main.exists():
        df = pd.read_csv(p_main)
        rows_main = df.to_dict("records")
        for r in rows_main:
            done_main.add(key_main(r["mode"], r["n_bus"], r["pv_share"],
                                   r["tau"], r["stress_axis"], r["stress_level"]))
    if p_nr.exists():
        df = pd.read_csv(p_nr)
        rows_nr = df.to_dict("records")
        for r in rows_nr:
            done_nr.add(key_nr(r["mode"], r["n_bus"], r["pv_share"],
                               r["stress_axis"], r["stress_level"]))
    return rows_main, rows_nr, done_main, done_nr


def flush(out: Path, rows_main: list, rows_nr: list):
    out.mkdir(parents=True, exist_ok=True)
    if rows_main:
        pd.DataFrame(rows_main).to_csv(out / "tau_sweep_main.csv", index=False)
    if rows_nr:
        pd.DataFrame(rows_nr).to_csv(out / "tau_sweep_nr.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 8) Dry-Run: Job-/Speicherschaetzung ohne Ausfuehrung
# ═════════════════════════════════════════════════════════════════════════════
def dry_run_summary(cfg: SweepConfig, grids: dict):
    n_bus_list, pv_list, tau_list = grids["n_bus"], grids["pv_share"], grids["tau"]
    stress = STRESS_CONFIGS
    print(f"\n=== Dry-Run: Modus {cfg.mode} ===")
    print(f"n_bus:    {n_bus_list}")
    print(f"pv_share: {pv_list}")
    print(f"tau:      {tau_list}")
    print(f"stress:   {[ (s['axis'], s['level']) for s in stress ]}")
    print(f"repeats={cfg.repeats}  time_cap_s={cfg.time_cap_s}  "
          f"mem_cap_gb={cfg.mem_cap_gb}  mem_factor={cfg.mem_factor}\n")

    total, would_run, would_skip = 0, 0, 0
    rows = []
    for n_bus in n_bus_list:
        for pv_share in pv_list:
            for s in stress:
                skip_from_here = False
                for tau in tau_list:
                    total += 1
                    mem_gb = mem_estimate_gb(n_bus, tau, cfg.mem_factor)
                    skip = skip_from_here or mem_gb > cfg.mem_cap_gb
                    if skip:
                        would_skip += 1
                        skip_from_here = True
                    else:
                        would_run += 1
                    rows.append(dict(n_bus=n_bus, pv_share=pv_share, tau=tau,
                                     axis=s["axis"], level=s["level"],
                                     mem_est_gb=mem_gb, skip=skip))
    df = pd.DataFrame(rows)
    print(f"Job-Slots gesamt (Hauptgitter):     {total}")
    print(f"davon voraussichtlich ausgefuehrt:  {would_run}")
    print(f"davon memory-cap geskippt:          {would_skip}")
    print("\ngroesste noch ausgefuehrte (n_bus, tau, mem_est_gb):")
    ok = df[~df.skip].sort_values("mem_est_gb", ascending=False).head(10)
    print(ok[["n_bus", "tau", "mem_est_gb"]].to_string(index=False))

    nr_blocks = len(n_bus_list) * len(pv_list) * len(stress)
    print(f"\nNR-Referenzbloecke: {nr_blocks}  "
          f"( je {grids.get('nr_tau_ref', cfg.nr_tau_ref)} Einzelsolves "
          f"-> {nr_blocks * grids.get('nr_tau_ref', cfg.nr_tau_ref)} NR-Aufrufe gesamt )")


# ═════════════════════════════════════════════════════════════════════════════
# 9) Hauptschleife
# ═════════════════════════════════════════════════════════════════════════════
def run_sweep(cfg: SweepConfig, grids: dict, resume: bool) -> dict:
    rows_main, rows_nr, done_main, done_nr = load_resume(cfg.out, resume)
    nr_cache: dict = {}
    for r in rows_nr:
        nr_cache[key_nr(r["mode"], r["n_bus"], r["pv_share"],
                        r["stress_axis"], r["stress_level"])] = r

    overhead = calibrate_nr_overhead(cfg)
    print(f"NR-Overhead: min={overhead['nr_overhead_ms_min']:.2f} ms, "
          f"median={overhead['nr_overhead_ms_median']:.2f} ms")

    n_bus_list, pv_list, tau_list = grids["n_bus"], grids["pv_share"], grids["tau"]
    stress_list = STRESS_CONFIGS
    total_blocks = len(n_bus_list) * len(pv_list) * len(stress_list)
    idx = 0
    t_sweep_start = time.perf_counter()

    for n_bus in n_bus_list:
        for pv_share in pv_list:
            for s in stress_list:
                idx += 1
                if cfg.max_total_hours is not None:
                    if (time.perf_counter() - t_sweep_start) / 3600.0 > cfg.max_total_hours:
                        print(f"\nZeitbudget ({cfg.max_total_hours} h) erreicht, "
                              f"breche kontrolliert ab.")
                        flush(cfg.out, rows_main, rows_nr)
                        return overhead

                axis, level, lam_ref, rho = s["axis"], s["level"], s["lam"], s["rho"]
                grid = build_grid(cfg, n_bus, rho)
                case = make_case(cfg, grid, pv_share, lam_ref)
                nd = to_network_data(case)

                k_nr = key_nr(cfg.mode, n_bus, pv_share, axis, level)
                if k_nr not in nr_cache:
                    nr_row = run_nr_reference(cfg, case, lam_ref, grids["nr_tau_ref"])
                    nr_row.update(mode=cfg.mode, n_bus=n_bus, pv_share=pv_share,
                                 n_pv=case.n_pv, stress_axis=axis, stress_level=level,
                                 lam_ref=lam_ref, rho=rho)
                    rows_nr.append(nr_row)
                    nr_cache[k_nr] = nr_row
                nr_row = nr_cache[k_nr]

                print(f"[{idx:>4}/{total_blocks}] n={n_bus:<5} pv={pv_share:<4} "
                      f"stress={axis}({level}) lam={lam_ref:<4} rho={rho:.3f}  "
                      f"nr_conv_share={nr_row.get('nr_conv_share', float('nan')):.2f}")

                skip_from_here = False
                for tau in tau_list:
                    k_main = key_main(cfg.mode, n_bus, pv_share, tau, axis, level)
                    if k_main in done_main:
                        continue

                    row = dict(mode=cfg.mode, n_bus=n_bus, pv_share=pv_share,
                               n_pv=case.n_pv, tau=tau, stress_axis=axis,
                               stress_level=level, lam_ref=lam_ref, rho=rho,
                               v_min_base=case.v_min_base, load_scale=grid.load_scale)

                    mem_gb = mem_estimate_gb(n_bus, tau, cfg.mem_factor)
                    row["mem_est_gb"] = mem_gb
                    if skip_from_here or mem_gb > cfg.mem_cap_gb:
                        row["skip_reason"] = "memory_cap"
                        row["timed_out"] = False
                        rows_main.append(row)
                        skip_from_here = True
                        continue

                    seed_batch = make_seed(n_bus, round(pv_share, 6), tau, axis, level)
                    S, lam_actual = make_batch(case, lam_ref, tau, cfg.jitter_lambda, seed_batch)

                    times, pi, res, timed_out, error = run_tpf_job(
                        nd, S, cfg, repeats=grids["repeats"], time_cap_s=grids["time_cap_s"])

                    row.update(lam_batch_min=float(lam_actual.min()),
                              lam_batch_max=float(lam_actual.max()))

                    if error:
                        row["skip_reason"] = "error"
                        row["error"] = error
                        rows_main.append(row)
                        print(f"    tau={tau:<8} FEHLER: {error}")
                        continue

                    row.update(
                        t_pre_ms_min=float(np.min(times["t_pre"])),
                        t_pre_ms_median=float(np.median(times["t_pre"])),
                        t_solve_ms_min=float(np.min(times["t_solve"])),
                        t_solve_ms_median=float(np.median(times["t_solve"])),
                        t_wall_ms_min=float(np.min(times["t_wall"])),
                        t_wall_ms_median=float(np.median(times["t_wall"])),
                    )
                    row["t_total_ms_min"] = row["t_pre_ms_min"] + row["t_solve_ms_min"]
                    row["t_total_ms_median"] = row["t_pre_ms_median"] + row["t_solve_ms_median"]
                    row["t_per_scen_ms_min"] = row["t_total_ms_min"] / max(tau, 1)
                    row["t_per_scen_ms_median"] = row["t_total_ms_median"] / max(tau, 1)

                    if pi is not None:
                        v_min_arr = getattr(pi, "v_min_per_scenario", None)
                        v_max_arr = getattr(pi, "v_max_per_scenario", None)
                        row.update(
                            k_out=int(pi.outer_iterations),
                            k_in=int(pi.inner_iterations_total),
                            converged=bool(res.converged),
                            conv_share=pi.n_converged_scenarios / max(1, pi.n_scenarios),
                            v_min_batch=(float(np.nanmin(v_min_arr)) if v_min_arr is not None else np.nan),
                            v_max_batch=(float(np.nanmax(v_max_arr)) if v_max_arr is not None else np.nan),
                            gflops=(pi.flops_gemm / (row["t_solve_ms_min"] * 1e-3) / 1e9
                                   if row["t_solve_ms_min"] > 0 else np.nan),
                        )
                        q_final = getattr(pi, "pv_q_final", None)
                        if case.n_pv and q_final is not None and np.size(q_final):
                            q = np.abs(np.asarray(q_final))
                            row.update(q_max_batch=float(np.max(q)), q_med_batch=float(np.median(q)))
                        else:
                            row.update(q_max_batch=np.nan, q_med_batch=np.nan)

                    # --- NR-Referenz auf denselben tau hochskaliert ---
                    t_nr_min = nr_row["t_nr_ms_min"] * tau
                    t_nr_med = nr_row["t_nr_ms_median"] * tau
                    ov = overhead["nr_overhead_ms_median"]
                    t_nr_corr_per_scen = max(nr_row["t_nr_ms_median"] - ov, 1e-6)
                    t_nr_corr_total = t_nr_corr_per_scen * tau
                    row.update(
                        t_nr_total_ms_min=t_nr_min, t_nr_total_ms_median=t_nr_med,
                        speedup_raw=(t_nr_med / row["t_total_ms_median"]
                                    if row["t_total_ms_median"] > 0 else np.nan),
                        speedup_corrected=(t_nr_corr_total / row["t_total_ms_median"]
                                          if row["t_total_ms_median"] > 0 else np.nan),
                    )

                    row["timed_out"] = timed_out
                    row["skip_reason"] = "time_cap" if timed_out else ""
                    if timed_out:
                        skip_from_here = True

                    rows_main.append(row)
                    print(f"    tau={tau:<8} t_total(min)={row['t_total_ms_min']:>10.2f} ms  "
                          f"k_out={row.get('k_out', '-')!s:<4} k_in={row.get('k_in', '-')!s:<5} "
                          f"conv={row.get('converged', '-')!s:<6} "
                          f"speedup(korr)={row.get('speedup_corrected', float('nan')):.2f}")

                if idx % cfg.checkpoint == 0:
                    flush(cfg.out, rows_main, rows_nr)

    flush(cfg.out, rows_main, rows_nr)
    return overhead


# ═════════════════════════════════════════════════════════════════════════════
# 10) CLI / main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["quick", "thorough"], default="quick")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mem-cap-gb", type=float, default=30.0)
    ap.add_argument("--mem-factor", type=float, default=6.0)
    ap.add_argument("--time-cap-s", type=float, default=None,
                    help="ueberschreibt den Modus-Default")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--nr-tau-ref", type=int, default=None)
    ap.add_argument("--jitter-lambda", type=float, default=0.15)
    ap.add_argument("--checkpoint", type=int, default=1,
                    help="CSV nach je N (n_bus,pv_share,stress)-Bloecken sichern")
    ap.add_argument("--max-total-hours", type=float, default=None)
    args = ap.parse_args()

    defaults = defaults_for_mode(args.mode)
    cfg = SweepConfig(
        mode=args.mode,
        out=Path(args.out or f"results_tau_{args.mode}"),
        mem_cap_gb=args.mem_cap_gb, mem_factor=args.mem_factor,
        jitter_lambda=args.jitter_lambda, checkpoint=args.checkpoint,
        repeats=args.repeats if args.repeats is not None else defaults["repeats"],
        time_cap_s=args.time_cap_s if args.time_cap_s is not None else defaults["time_cap_s"],
        nr_tau_ref=args.nr_tau_ref if args.nr_tau_ref is not None else defaults["nr_tau_ref"],
        max_total_hours=args.max_total_hours,
    )

    grids = grid_lists_for_mode(args.mode)
    grids.update(repeats=cfg.repeats, time_cap_s=cfg.time_cap_s, nr_tau_ref=cfg.nr_tau_ref)

    if args.dry_run:
        dry_run_summary(cfg, grids)
        return

    print(f"Modus: {cfg.mode}  ->  Output: {cfg.out}")
    print(f"n_bus={grids['n_bus']}")
    print(f"pv_share={grids['pv_share']}")
    print(f"tau={grids['tau']}")
    print(f"repeats={cfg.repeats}  time_cap_s={cfg.time_cap_s}  "
          f"nr_tau_ref={cfg.nr_tau_ref}  mem_cap_gb={cfg.mem_cap_gb}")

    t0 = time.perf_counter()
    overhead = run_sweep(cfg, grids, resume=args.resume)
    runtime_s = time.perf_counter() - t0

    meta = dict(
        mode=cfg.mode, n_bus=grids["n_bus"], pv_share=grids["pv_share"],
        tau=grids["tau"], stress_configs=STRESS_CONFIGS,
        repeats=cfg.repeats, time_cap_s=cfg.time_cap_s, nr_tau_ref=cfg.nr_tau_ref,
        mem_cap_gb=cfg.mem_cap_gb, mem_factor=cfg.mem_factor,
        jitter_lambda=cfg.jitter_lambda, dv_setpoint=cfg.dv_setpoint,
        v_min_target_lam1=cfg.v_min_target_lam1, cos_phi=cfg.cos_phi,
        z0_ohm_km=cfg.z0_ohm_km, pv_p_total_ratio=cfg.pv_p_total_ratio,
        solver_config=dict(coupled=True, warm_start=True, adaptive_inner=True,
                           batch_init="flat", enforce_q_lims=False),
        nr_overhead=overhead, runtime_s=runtime_s,
        python=sys.version.split()[0], platform=platform.platform(),
        numpy=np.__version__, pandapower=pp.__version__,
    )
    (cfg.out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nfertig. Laufzeit: {runtime_s/60:.1f} min")
    print(f"geschrieben: {cfg.out}/tau_sweep_main.csv, tau_sweep_nr.csv, meta.json")


if __name__ == "__main__":
    main()
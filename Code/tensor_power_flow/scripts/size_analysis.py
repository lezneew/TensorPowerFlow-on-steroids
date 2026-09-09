# =============================================================================
# size_analysis.py  —  Analyze solver performance vs network size with varying PV %
# =============================================================================
"""
Generates CSV and plots comparing TPF solver performance with 0%, 25%, 50%, 75% PV nodes.
Tests network sizes: 20, 40, 75, 120, 200, 350, 500, 750, 1000, 1500 buses.

Usage:
    python size_analysis.py --out size_analysis_results
    python size_analysis.py --out size_analysis_results --quick
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from tpf.solvers.tpf_dense import TPFDenseSolver
from tpf.solvers.tpf_pv_method_a import TPFDensePVMethodA
from tpf.solvers.nr_reference import PandapowerNRSolver

warnings.filterwarnings("ignore")

COSPHI_LOAD = 0.95
V_MIN_TARGET = 0.97
Z0_REF = 0.6470
RHO_REF = 7.73
LEN_KM = 0.03
SN_MVA = 1.0
VN_KV = 0.4


# -----------------------------------------------------------------------------
# Network Data (duck-typed, only needed attributes)
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
# Case definition
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    n_bus: int = 40
    seed: int = 0
    n_feeders: int = 4
    n_pv: int = 0
    pv_percent: float = 0.0
    placement: str = "random"
    lam: float = 1.0
    rho: float = RHO_REF
    z0: float = Z0_REF
    length_km: float = LEN_KM
    tau: int = 1


# -----------------------------------------------------------------------------
# Topology and network building
# -----------------------------------------------------------------------------
def line_rx(z0: float, rho: float) -> tuple[float, float]:
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
# Caching and network construction
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


def _solve_with_pv(nd: SimpleNetworkData, s_batch, tol=1e-8, max_inner=200, max_outer=60):
    sol = TPFDensePVMethodA(tol=tol, max_iter_inner=max_inner,
                            max_iter_outer=max_outer, tol_pv=1e-6,
                            omega=1.0, adaptive_inner=True, cold_start=False)
    res = sol.solve_batch(nd, s_batch)
    return res, sol.pv_info


def _load_scale(case: Case, parents, p_load, Y):
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
    parents, feeder, depth, p_load = _base_pattern(case)
    r_ohm, x_ohm = line_rx(case.z0, case.rho)
    Y = make_ybus(parents, r_ohm, x_ohm, case.length_km, case.n_bus)
    scale = _load_scale(case, parents, p_load, Y)

    tanphi = np.tan(np.arccos(COSPHI_LOAD))
    p_l = case.lam * scale * p_load / SN_MVA
    q_l = p_l * tanphi

    D = _dist(case, parents) if case.placement == "spread" else None
    pv_bus = place_pv(case.n_bus, case.n_pv, case.placement, case.seed,
                      parents, feeder, depth, D)
    pv_idx = pv_bus - 1

    Y_dd, Y_ds, Y_sd, Y_ss = Y[1:, 1:], Y[1:, :1], Y[:1, 1:], Y[:1, :1]
    v_s = np.array([1.0 + 0j])
    s_d = p_l[1:] + 1j * q_l[1:]

    if case.n_pv > 0:
        nd_pq = SimpleNetworkData(Y_dd, Y_ds, Y_sd, Y_ss, v_s, s_d.copy())
        res_pq, _ = _solve_pq(nd_pq, s_d.reshape(-1, 1))
        v_base = np.abs(res_pq.voltages[:, 0]) if res_pq.converged else np.full(len(s_d), np.nan)
        v_spec = v_base[pv_idx] + 0.005
    else:
        v_spec = None

    nd = SimpleNetworkData(
        Y_dd, Y_ds, Y_sd, Y_ss, v_s, s_d.copy(),
        pv_indices=pv_idx if case.n_pv > 0 else None,
        pv_v_setpoint=v_spec,
    )

    return nd


# -----------------------------------------------------------------------------
# Run a single case
# -----------------------------------------------------------------------------
def run_case(case: Case) -> dict:
    nd = build_case(case)
    s_batch = nd.s_nom.reshape(-1, 1)

    if case.n_pv == 0:
        solver = TPFDenseSolver(tol=1e-8, max_iter=400)
        t0 = time.perf_counter()
        res = solver.solve_batch(nd, s_batch)
        t_solve_ms = (time.perf_counter() - t0) * 1e3

        row = dict(
            n_bus=case.n_bus,
            seed=case.seed,
            pv_percent=case.pv_percent,
            n_pv=case.n_pv,
            t_solve_ms=t_solve_ms,
            iterations=res.iterations,
            outer_iterations=1,
            converged=res.converged,
        )
    else:
        kw = dict(tol=1e-8, max_iter_inner=200, max_iter_outer=60,
                  tol_pv=1e-6, omega=1.0, adaptive_inner=True, cold_start=False)
        solver = TPFDensePVMethodA(**kw)
        t0 = time.perf_counter()
        res = solver.solve_batch(nd, s_batch)
        t_solve_ms = (time.perf_counter() - t0) * 1e3
        pi = solver.pv_info

        row = dict(
            n_bus=case.n_bus,
            seed=case.seed,
            pv_percent=case.pv_percent,
            n_pv=case.n_pv,
            t_solve_ms=t_solve_ms,
            iterations=pi.inner_iterations_total,
            outer_iterations=int(pi.outer_iterations),
            converged=res.converged,
        )

    row["t_per_iter_ms"] = row["t_solve_ms"] / row["iterations"] if row["iterations"] > 0 else np.nan

    return row


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="size_analysis_results")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        sizes = [20, 40, 75, 120, 200]
    else:
        sizes = [20, 40, 75, 120, 200, 350, 500, 750, 1000, 1500]

    pv_percents = [0, 25, 50, 75]
    seeds = [0, 1, 2, 4, 5]

    print(f"Running size analysis: {len(sizes)} sizes, {len(pv_percents)} PV%, {len(seeds)} seeds")
    print(f"Total runs: {len(sizes) * len(pv_percents) * len(seeds)}")
    print()

    rows = []
    for n in sizes:
        for pv_pct in pv_percents:
            n_pv = int(round(pv_pct / 100.0 * (n - 1)))
            for seed in seeds:
                case = Case(n_bus=n, seed=seed, n_pv=n_pv, pv_percent=pv_pct)
                print(f"  n={n:4d} pv_pct={pv_pct:3d}% n_pv={n_pv:3d} seed={seed}", end=" ")
                try:
                    row = run_case(case)
                    print(f"t={row['t_solve_ms']:.2f}ms iter={row['iterations']:3d} conv={row['converged']}")
                    rows.append(row)
                except Exception as e:
                    print(f"ERROR: {e}")
                    rows.append(dict(
                        n_bus=n, seed=seed, pv_percent=pv_pct, n_pv=n_pv,
                        t_solve_ms=np.nan, iterations=0, outer_iterations=0, t_per_iter_ms=np.nan,
                        converged=False
                    ))

    df = pd.DataFrame(rows)
    csv_path = out / f"{args.out}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path} ({len(df)} rows)")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(5.9, 8.5))

    colors = {0: "#1f77b4", 25: "#2ca02c", 50: "#ff7f0e", 75: "#d62728"}
    labels = {0: "0%", 25: "25%", 50: "50%", 75: "75%"}

    size_arr = np.array(sorted(df["n_bus"].unique()), dtype=float)

    for ax_idx, y_col in enumerate(["t_solve_ms", "t_per_iter_ms", "iterations"]):
        ax = axes[ax_idx]
        ax.set_xscale("log")
        ax.set_yscale("log")

        box_width = 0.15

        for i, pv_pct in enumerate(pv_percents):
            subset = df[(df["pv_percent"] == pv_pct) & df["converged"]]
            data_by_size = [subset[subset["n_bus"] == n][y_col].values for n in size_arr]

            positions = size_arr + (i - 1.5) * box_width * size_arr * 0.3

            bp = ax.boxplot(data_by_size, positions=positions,
                            widths=box_width * size_arr * 0.25,
                            patch_artist=True,
                            boxprops=dict(facecolor=colors[pv_pct], alpha=0.6),
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(color=colors[pv_pct], linewidth=1.2),
                            capprops=dict(color=colors[pv_pct], linewidth=1.2),
                            flierprops=dict(marker="o", markerfacecolor=colors[pv_pct],
                                           markersize=3, alpha=0.5))

        ax.set_xticks(size_arr)
        ax.set_xticklabels(size_arr.astype(int))
        ax.set_xlabel(r"$n_{\mathrm{bus}}$", fontsize=11)

        if y_col == "t_solve_ms":
            ax.set_ylabel("Total solve time [ms]", fontsize=11)
        elif y_col == "t_per_iter_ms":
            ax.set_ylabel("Time per iteration [ms]", fontsize=11)
        else:
            ax.set_ylabel("Total iterations", fontsize=11)

        ax.grid(True, which="both", alpha=0.3)

    legend_items = [plt.Rectangle((0, 0), 1, 1, fc=colors[p], alpha=0.6) for p in pv_percents]
    legend_labels = [labels[p] for p in pv_percents]
    axes[0].legend(legend_items, legend_labels, loc="upper left", fontsize=9,
                   title="PV %", title_fontsize=9)

    plt.tight_layout()
    plot_path = out / f"{args.out}.pgf"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")

    png_path = out / f"{args.out}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {png_path}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()

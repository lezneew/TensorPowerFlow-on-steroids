#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_batch_figures.py
=====================
Erzeugt die PGF-Abbildungen und Kennzahlen des Abschnitts "Mehrere Lastfluesse".

Datenquellen
------------
1) runs_20260818_145947.csv       aktueller Solverstand (gekoppelte Q-Korrektur,
                                  Warm Start, adaptive Toleranz), n = 40..1000,
                                  PV-Anteile 0/30/50 %.
2) tau_scaling_sz_<n>_r<pv>_*.csv frueherer Solverstand (entkoppelte Q-Korrektur,
                                  ohne Warm Start / adaptive Toleranz), dafuer
                                  PV-Anteile 0..60 % feiner aufgeloest.
                                  Nur fuer Abb. 2 und die Fussnote verwendet.

Ausgabe (--outdir, Standard: figures/)
    batch_scaling_tau.pgf       Abb. 1  ms/Szenario und Beschleunigung ueber tau
    batch_total_time_tau.pgf    Abb. 2  Gesamtlaufzeit ueber tau, 4 Netzgroessen
    batch_pv_effect.pgf         Abb. 3  Plateauzeit ueber PV-Anteil, Proportionalitaet
    batch_key_numbers.txt       alle im Text zitierten Zahlen
    tab_batch_plateau.tex       Zeilen von Tab. 1
    tab_batch_pv.tex            Zeilen von Tab. 2

Aufruf
    python make_batch_figures.py --csv runs_20260818_145947.csv --legacy-dir .
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------
TAU_PLATEAU = 10_000      # Plateau = Median ueber tau >= TAU_PLATEAU
TAU_YEAR = 35_040         # 15-min-Jahreszeitreihe
LEGACY_RE = re.compile(r"tau_scaling_sz_(\d+)_r(\d+)_(\d{8}_\d{6})\.csv$")

COL = {40: "#1b6ca8", 200: "#2e8b57", 500: "#d1621a", 1000: "#8b1a1a"}
LS = {0: "-", 30: "--", 50: ":"}
MK = {0: "o", 30: "s", 50: "^"}

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",     # ggf. "lualatex"
    "pgf.rcfonts": False,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 6.8,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "axes.axisbelow": True,
})


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------
def crossing_tau(tau, speed) -> float:
    """Kleinstes tau mit Beschleunigung = 1, log-log-interpoliert."""
    tau, s = np.asarray(tau, float), np.asarray(speed, float)
    o = np.argsort(tau)
    tau, s = tau[o], s[o]
    ok = np.isfinite(s) & (s > 0)
    tau, s = tau[ok], s[ok]
    if len(tau) == 0:
        return np.nan
    if s[0] >= 1.0:
        return float(tau[0])
    for i in range(1, len(tau)):
        if s[i] >= 1.0 > s[i - 1]:
            x0, x1 = np.log10(tau[i - 1]), np.log10(tau[i])
            y0, y1 = np.log10(s[i - 1]), np.log10(s[i])
            return float(10 ** (x0 - y0 * (x1 - x0) / (y1 - y0)))
    return np.nan


def med_plateau(g: pd.DataFrame, col: str) -> float:
    gp = g[g.tau >= TAU_PLATEAU]
    return float((gp if len(gp) else g)[col].median())


# ----------------------------------------------------------------------
# Datenaufbereitung: aktueller Stand
# ----------------------------------------------------------------------
def load_new(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["pv_pct"] = (df["pv_share"] * 100).round().astype(int)
    df["nr_extrapolated"] = df["nr_extrapolated"].astype(bool)
    df["t_total_ms"] = df["t_precompute_ms"] + df["t_solve_ms"]
    df["pre_share"] = df["t_precompute_ms"] / df["t_total_ms"]
    return df.sort_values(["n_bus", "pv_pct", "tau"]).reset_index(drop=True)


def per_config(df: pd.DataFrame) -> pd.DataFrame:
    t_pre_by_n = df.groupby("n_bus")["t_precompute_ms"].median()
    out = []
    for (n, pv), g in df.groupby(["n_bus", "pv_pct"]):
        g = g.sort_values("tau")
        gm = g[~g.nr_extrapolated]                 # NR tatsaechlich gemessen
        t_inf = med_plateau(g, "tpf_per_scenario_ms")
        t_nr = float(gm.nr_per_scenario_ms.median())
        ovh = float(g.t_nr_overhead_ms.median())
        last = g.iloc[-1]
        out.append(dict(
            n=int(n), pv=int(pv), n_pv=int(g.n_pv.iloc[0]), tau_max=int(last.tau),
            k_in=int(med_plateau(g, "k_in_max")),
            k_in_tau1=int(g.k_in_max.iloc[0]),
            k_out=int(med_plateau(g, "k_out_max")),
            k_out_tau1=int(g.k_out_max.iloc[0]),
            t_inf=t_inf, thr=1e3 / t_inf,
            gflops=med_plateau(g, "gflops_eff"),
            t_pre=float(t_pre_by_n[n]),
            gain=float(g.tpf_per_scenario_ms.iloc[0]) / t_inf,
            pre_share_tau1=float(g.pre_share.iloc[0]),
            pre_share_max=float(g.pre_share.iloc[-1]),
            tau_pre_equal=float(t_pre_by_n[n] / t_inf),
            t_nr=t_nr, t_nr_ovh=ovh,
            su_raw=t_nr / t_inf, su_fair=max(t_nr - ovh, 0.0) / t_inf,
            su_tau1=float(g.speedup.iloc[0]),
            tau_star_raw=crossing_tau(g.tau, g.speedup),
            tau_star_fair=crossing_tau(g.tau, g.speedup_fair),
            cond=float(g.cond_Xpp.iloc[0]), rho=float(g.rho_jacobi.iloc[0]),
            diag_off=float(g.min_diag_off.iloc[0]),
            vmin_tau1=float(g.vmin_min.iloc[0]), vmin_max=float(g.vmin_min.iloc[-1]),
            dv_max=float(gm.dv_max_vs_nr.max()),
            dth_max=float(gm.dtheta_max_deg_vs_nr.max()),
            pv_err=float(g.pv_v_err_max.max()), q_max=float(g.q_pv_absmax.max()),
            conv=float(g.conv_frac.min()),
        ))
    cfg = pd.DataFrame(out).sort_values(["n", "pv"]).reset_index(drop=True)
    cfg["k_per_outer"] = np.where(cfg.k_out > 0, cfg.k_in / cfg.k_out.clip(lower=1),
                                  np.nan)
    return cfg


# ----------------------------------------------------------------------
# Datenaufbereitung: Alt-Sweep
# ----------------------------------------------------------------------
def load_legacy(folder: Path) -> pd.DataFrame:
    frames, dropped = [], []
    for p in sorted(folder.glob("tau_scaling_sz_*_r*.csv")):
        m = LEGACY_RE.search(p.name)
        if not m:
            continue
        d = pd.read_csv(p)
        d["n_bus"] = int(m.group(1))
        d["pv_pct"] = int(m.group(2))
        d["stamp"] = m.group(3)
        d["file"] = p.name
        d["tpf_converged"] = d["tpf_converged"].astype(bool)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    keep = []
    for (n, pv), g in df.groupby(["n_bus", "pv_pct"]):
        best, best_key = None, (-1, -1)
        for fn, gg in g.groupby("file"):
            key = (len(gg), int(gg.tau.max()))
            if key > best_key:
                if best is not None:
                    dropped.append(best.file.iloc[0])
                best, best_key = gg, key
            else:
                dropped.append(fn)
        keep.append(best)
    out = pd.concat(keep, ignore_index=True)
    out.attrs["dropped"] = sorted(set(dropped))
    out["n_pv_est"] = (out.n_bus * out.pv_pct / 100).round().astype(int)
    return out.sort_values(["n_bus", "pv_pct", "tau"]).reset_index(drop=True)


def legacy_summary(leg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, pv), g in leg.groupby(["n_bus", "pv_pct"]):
        g = g.sort_values("tau")
        div = g[~g.tpf_converged]
        rows.append(dict(
            n=int(n), pv=int(pv), n_pv=int(g.n_pv_est.iloc[0]),
            t_scen=med_plateau(g, "tpf_per_scenario"),
            t_nr_scen=med_plateau(g, "nr_per_scenario"),
            t_tot_max=float(g.t_tpf.iloc[-1]), tau_max=int(g.tau.iloc[-1]),
            su_max_tau=float(g.speedup.iloc[-1]),
            tau_div=(int(div.tau.min()) if len(div) else 0),
            stamp=g.stamp.iloc[0], file=g.file.iloc[0],
        ))
    s = pd.DataFrame(rows).sort_values(["n", "pv"]).reset_index(drop=True)
    base = s[s.pv == 0].set_index("n").t_scen
    s["ratio_pq"] = [r.t_scen / base[r.n] if r.n in base.index else np.nan
                     for _, r in s.iterrows()]
    return s


# ----------------------------------------------------------------------
# Abb. 1: Amortisation und Break-even
# ----------------------------------------------------------------------
def fig_tau(df: pd.DataFrame, cfg: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.85),
                                   constrained_layout=True)

    nr = df.loc[~df.nr_extrapolated, "nr_per_scenario_ms"]
    ax1.axhspan(nr.quantile(0.10), nr.quantile(0.90), color="0.87", zorder=0)
    ax1.axhline(nr.median(), color="0.45", lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax1.text(1.3, nr.median() * 1.25, "NR (pandapower)", color="0.35", fontsize=7)

    for _, r in cfg.iterrows():
        g = df[(df.n_bus == r.n) & (df.pv_pct == r.pv)].sort_values("tau")
        ax1.plot(g.tau, g.tpf_per_scenario_ms, color=COL[r.n], ls=LS[r.pv],
                 marker=MK[r.pv], mfc="none", mew=0.7)
        ax2.plot(g.tau, g.speedup, color=COL[r.n], ls=LS[r.pv],
                 marker=MK[r.pv], mfc="none", mew=0.7)

    r = cfg[cfg.n == 1000].iloc[0]
    tt = np.logspace(0, np.log10(r.tau_max), 60)
    ax1.plot(tt, r.t_pre / tt + r.t_inf, color="0.25", lw=0.7, ls=(0, (1, 1.5)))
    ax1.text(2.0e3, r.t_pre / 2.0e3 + r.t_inf * 3,
             r"$t_\mathrm{pre}/\tau+t_\infty$", color="0.25", fontsize=7)

    ax1.set(xscale="log", yscale="log", xlabel=r"$\tau$",
            ylabel=r"$t_\mathrm{Szen}$ in ms")
    ax2.axhline(1.0, color="0.2", lw=0.8)
    ax2.axvspan(1e4, 1.1e5, color="0.92", zorder=0)
    ax2.text(1.4e4, 1.6, "NR extrapoliert", color="0.35", fontsize=7,
             rotation=90, va="bottom")
    ax2.set(xscale="log", yscale="log", xlabel=r"$\tau$",
            ylabel="Beschleunigung TPF/NR")

    h_n = [plt.Line2D([], [], color=COL[n], lw=1.2) for n in sorted(COL)]
    ax1.legend(h_n, [rf"$n={n}$" for n in sorted(COL)], loc="lower left",
               ncol=2, handlelength=1.6)
    h_p = [plt.Line2D([], [], color="0.3", ls=LS[p], marker=MK[p], mfc="none",
                      mew=0.7) for p in sorted(LS)]
    ax2.legend(h_p, [rf"${p}\,\%$ PV" for p in sorted(LS)], loc="upper left",
               handlelength=2.0)

    fig.savefig(out / "batch_scaling_tau.pgf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Abb. 2: Gesamtlaufzeit ueber tau, eine Netzgroesse je Subplot (Alt-Sweep)
# ----------------------------------------------------------------------
def fig_total_time(leg: pd.DataFrame, out: Path,
                   pv_pick: int | None = None,
                   sizes: tuple[int, ...] | None = None) -> None:
    if leg.empty:
        return

    # PV-Anteil mit der groessten Zahl verfuegbarer Netzgroessen waehlen
    if pv_pick is None:
        pv_pick = int(leg.groupby("pv_pct").n_bus.nunique().idxmax())

    d = leg[leg.pv_pct == pv_pick]
    if sizes:
        d = d[d.n_bus.isin(sizes)]
    if d.empty:
        print(f"[warn] keine Alt-Sweep-Daten fuer pv={pv_pick}%")
        return
    ns = sorted(d.n_bus.unique())

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.LogNorm(vmin=min(ns) * 0.85, vmax=max(ns) * 1.15)

    fig, ax = plt.subplots(figsize=(4.7, 3.3), constrained_layout=True)

    # NR-Referenzen im Hintergrund
    for n in ns:
        g = d[d.n_bus == n].sort_values("tau")
        ax.plot(g.tau, g.t_nr, color="0.75", lw=0.7, ls=(0, (4, 2)), zorder=1)

    for n in ns:
        g = d[d.n_bus == n].sort_values("tau")
        c = cmap(norm(n))
        ax.plot(g.tau, g.t_tpf, color=c, marker="o", ms=2.6, mfc="none",
                mew=0.7, zorder=3,
                label=rf"$n_\mathrm{{bus}}={n}$ ($n_\mathrm{{pv}}="
                      rf"{g.n_pv_est.iloc[0]}$)")
        bad = g[~g.tpf_converged]
        if len(bad):
            ax.plot(bad.tau, bad.t_tpf, ls="none", marker="x", ms=4.4,
                    mew=1.0, color=c, zorder=4)

    # Referenzsteigung propto tau
    g = d[d.n_bus == ns[-1]].sort_values("tau")
    tau_r, t_r = float(g.tau.iloc[-1]), float(g.t_tpf.iloc[-1])
    tt = np.array([tau_r / 100.0, tau_r])
    ax.plot(tt, 0.30 * t_r * tt / tau_r, color="0.35", lw=0.7, ls=(0, (1, 1.5)),
            zorder=2)
    ax.text(tau_r / 30.0, 0.30 * t_r / 22.0, r"$\propto\tau$", color="0.35",
            fontsize=7)

    ax.plot([], [], color="0.75", lw=0.7, ls=(0, (4, 2)), label="NR-Referenz")
    ax.set(xscale="log", yscale="log", xlabel=r"$\tau$",
           ylabel=r"$t_\mathrm{ges}$ in s")
    ax.text(0.975, 0.035, rf"PV-Anteil ${pv_pick}\,\%$", fontsize=7,
            color="0.35", ha="right", transform=ax.transAxes)
    ax.legend(loc="upper left", handlelength=1.5, labelspacing=0.25,
              borderpad=0.3)

    fig.savefig(out / "batch_total_time_tau.pgf")
    plt.close(fig)

# ----------------------------------------------------------------------
# Abb. 3: Wirkung des PV-Anteils im aktuellen Stand
# ----------------------------------------------------------------------
def fig_pv_effect(cfg: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.9),
                                   constrained_layout=True)

    for n, g in cfg.groupby("n"):
        if len(g) < 2:
            continue
        g = g.sort_values("pv")
        ax1.plot(g.pv, g.t_inf, color=COL[n], marker="o", mfc="none", mew=0.8,
                 label=rf"$n={n}$")
        for _, r in g.iterrows():
            ax1.annotate(rf"${r.k_in:.0f}$", (r.pv, r.t_inf), fontsize=6.5,
                         color=COL[n], textcoords="offset points", xytext=(4, 4))
    ax1.set(yscale="log", xlabel=r"PV-Anteil in \%",
            ylabel=r"$t_\infty$ in ms/Szenario")
    ax1.set_xticks([0, 30, 50])
    ax1.legend(loc="upper left", handlelength=1.6)
    ax1.text(0.98, 0.03, r"Beschriftung: $k_\mathrm{in}$", fontsize=6.5,
             color="0.35", ha="right", transform=ax1.transAxes)

    for n, g in cfg.groupby("n"):
        b = g[g.pv == 0]
        if b.empty or len(g) < 2:
            continue
        b = b.iloc[0]
        for _, r in g[g.pv > 0].iterrows():
            ax2.plot(r.k_in / b.k_in, r.t_inf / b.t_inf, marker=MK[r.pv],
                     color=COL[n], mfc="none", mew=0.9, ms=4.5)
            ax2.annotate(rf"${n}$/${r.pv:.0f}\,\%$",
                         (r.k_in / b.k_in, r.t_inf / b.t_inf), fontsize=6.5,
                         color=COL[n], textcoords="offset points", xytext=(5, -4))
    lim = np.array([0.9, 8.0])
    ax2.plot(lim, lim, color="0.35", lw=0.8)
    ax2.text(4.6, 4.0, r"$1:1$", color="0.35", fontsize=7, rotation=38)
    ax2.set(xscale="log", yscale="log", xlim=tuple(lim), ylim=tuple(lim),
            xlabel=r"$k_\mathrm{in}/k_\mathrm{in,PQ}$",
            ylabel=r"$t_\infty/t_{\infty,\mathrm{PQ}}$")
    for a in (2, 4, 8):
        ax2.set_xticks([1, 2, 4, 8]); ax2.set_yticks([1, 2, 4, 8])
    ax2.set_xticklabels(["1", "2", "4", "8"])
    ax2.set_yticklabels(["1", "2", "4", "8"])

    fig.savefig(out / "batch_pv_effect.pgf")
    plt.close(fig)


# ----------------------------------------------------------------------
# Tabellen
# ----------------------------------------------------------------------
def de(x: float, nd: int) -> str:
    return f"{x:.{nd}f}".replace(".", "{,}")


def write_tables(cfg: pd.DataFrame, out: Path) -> None:
    with open(out / "tab_batch_plateau.tex", "w", encoding="utf-8") as f:
        prev = None
        for _, r in cfg.iterrows():
            if prev is not None and r.n != prev:
                f.write("\\midrule\n")
            prev = r.n
            ts = ("{$<1$}" if r.tau_star_fair <= 1.0 else de(r.tau_star_fair, 1))
            ko = "{--}" if r.pv == 0 else f"{r.k_out:.0f}"
            f.write(f"{r.n:.0f} & {r.pv:.0f} & {r.n_pv:.0f} & {r.k_in:.0f} & {ko} & "
                    f"{de(r.t_inf, 4)} & {r.gflops:.0f} & {de(r.t_nr, 1)} & "
                    f"{de(r.su_raw, 1)} & {de(r.su_fair, 1)} & {ts} \\\\\n")

    with open(out / "tab_batch_pv.tex", "w", encoding="utf-8") as f:
        prev = None
        for _, r in cfg.iterrows():
            if prev is not None and r.n != prev:
                f.write("\\midrule\n")
            prev = r.n
            if r.pv == 0:
                f.write(f"{r.n:.0f} & 0 & 0 & {{--}} & {r.k_in:.0f} & {{--}} & "
                        f"{{--}} & {{--}} & {{--}} \\\\\n")
            else:
                f.write(f"{r.n:.0f} & {r.pv:.0f} & {r.n_pv:.0f} & {r.k_out:.0f} & "
                        f"{r.k_in:.0f} & {de(r.k_per_outer, 1)} & {r.cond:.0f} & "
                        f"{de(r.rho, 2)} & {de(r.q_max, 2)} \\\\\n")

# ----------------------------------------------------------------------
# Kennzahlen
# ----------------------------------------------------------------------
def write_numbers(df, cfg, leg, legs, out: Path) -> None:
    L: list[str] = []
    p = L.append
    p("=" * 82)
    p("KENNZAHLEN  Abschnitt 'Mehrere Lastfluesse'")
    p("=" * 82)
    p(f"Laeufe (aktueller Stand): {len(df)}")
    p(f"Szenarien gesamt        : {int(df.tau.sum()):,}".replace(",", " "))
    p(f"min(conv_frac)          : {df.conv_frac.min():.6f}")
    p(f"Wiederholungen          : {sorted(df.repeats.unique())}")
    p(f"warm_mode               : {sorted(df.warm_mode.unique())}")
    p(f"tau-Stuetzstellen       : {sorted(df.tau.unique())}")
    p("")

    p("--- Vorberechnung (Median je Netzgroesse, ms) ---")
    for n, v in df.groupby("n_bus").t_precompute_ms.median().items():
        g = df[df.n_bus == n].t_precompute_ms
        p(f"  n={n:>4}: {v:8.2f}  (min {g.min():7.2f}, max {g.max():8.2f}, "
          f"Streufaktor {g.max()/g.min():4.1f})")
    p("")
    p("--- Anteil Vorberechnung / tau mit t_pre = t_solve ---")
    for _, r in cfg.iterrows():
        p(f"  n={r.n:>4} pv={r.pv:>2}%: tau=1 {100*r.pre_share_tau1:5.1f}%, "
          f"tau={r.tau_max:>6} {100*r.pre_share_max:5.2f}%, "
          f"Umschlag bei tau={r.tau_pre_equal:7.0f}, "
          f"Gewinn tau=1 -> Plateau x{r.gain:6.1f}")
    p(f"  Gewinn-Spanne: {cfg.gain.min():.0f} .. {cfg.gain.max():.0f} "
      f"(Median {cfg.gain.median():.0f})")
    p("")

    p(f"--- Plateau (Median ueber tau >= {TAU_PLATEAU}) ---")
    p(f"{'n':>5}{'pv%':>5}{'n_pv':>6}{'k_in':>6}{'k_out':>6}{'k_in/k_out':>11}"
      f"{'ms/Szen':>10}{'Szen/s':>9}{'GF/s':>7}{'NR ms':>7}{'SU roh':>8}"
      f"{'SU korr':>9}{'tau* roh':>9}{'tau* korr':>10}")
    for _, r in cfg.iterrows():
        p(f"{r.n:>5.0f}{r.pv:>5.0f}{r.n_pv:>6.0f}{r.k_in:>6.0f}{r.k_out:>6.0f}"
          f"{r.k_per_outer:>11.1f}{r.t_inf:>10.4f}{r.thr:>9.0f}{r.gflops:>7.1f}"
          f"{r.t_nr:>7.1f}{r.su_raw:>8.1f}{r.su_fair:>9.1f}"
          f"{r.tau_star_raw:>9.1f}{r.tau_star_fair:>10.1f}")
    p(f"  max. tau* (korr.): {np.nanmax(cfg.tau_star_fair):.1f}")
    p("")

    p("--- PV-Anteil: Verhaeltnisse zu 0 % PV (aktueller Stand) ---")
    for n, g in cfg.groupby("n"):
        if len(g) < 2:
            continue
        b = g[g.pv == 0].iloc[0]
        for _, r in g[g.pv > 0].iterrows():
            p(f"  n={n:>4} pv={r.pv:>2}% (n_pv={r.n_pv:>3}): k_in x{r.k_in/b.k_in:5.2f}"
              f"   t_inf x{r.t_inf/b.t_inf:5.2f}   P {b.gflops:5.1f} -> {r.gflops:5.1f}"
              f"   Erwartung aus k_in und P: x{(r.k_in/b.k_in)*(b.gflops/r.gflops):5.2f}")
    p("")
    p("--- PV-Diagnostik (X_pp) ---")
    for _, r in cfg[cfg.pv > 0].iterrows():
        p(f"  n={r.n:>4} pv={r.pv:>2}% n_pv={r.n_pv:>3}: cond={r.cond:8.1f}  "
          f"rho_Jacobi={r.rho:6.2f}  min(diag/off)={r.diag_off:.4f}  "
          f"|Q|max={r.q_max:5.2f} p.u.  k_out={r.k_out}  "
          f"k_in/k_out={r.k_per_outer:5.1f} (PQ: "
          f"{cfg[(cfg.n==r.n)&(cfg.pv==0)].k_in.iloc[0]})")
    p("")
    p("--- Ensembleeffekt und Spannungen ---")
    for _, r in cfg.iterrows():
        p(f"  n={r.n:>4} pv={r.pv:>2}%: k_in {r.k_in_tau1:>3} -> {r.k_in:>3}   "
          f"k_out {r.k_out_tau1:>2} -> {r.k_out:>2}   "
          f"v_min,min {r.vmin_tau1:.4f} -> {r.vmin_max:.4f}")
    p("")

    m = df[~df.nr_extrapolated]
    p("--- Genauigkeit gegen NR (nur gemessene Referenz) ---")
    p(f"  PQ : dv_max {m[m.pv_pct==0].dv_max_vs_nr.max():.2e} p.u., "
      f"dtheta_max {m[m.pv_pct==0].dtheta_max_deg_vs_nr.max():.2e} deg")
    p(f"  PV : dv_max {m[m.pv_pct>0].dv_max_vs_nr.max():.2e} p.u., "
      f"dtheta_max {m[m.pv_pct>0].dtheta_max_deg_vs_nr.max():.2e} deg")
    p(f"  max. pv_v_err_max {df.pv_v_err_max.max():.2e} p.u., "
      f"max |Q| {df.q_pv_absmax.max():.2f} p.u.")
    p("")

    p(f"--- Jahreszeitreihe 15 min (tau = {TAU_YEAR}) ---")
    for _, r in cfg.iterrows():
        p(f"  n={r.n:>4} pv={r.pv:>2}%: TPF {r.t_inf*TAU_YEAR/1e3:8.2f} s   "
          f"NR {r.t_nr*TAU_YEAR/1e3:8.1f} s ({r.t_nr*TAU_YEAR/6e4:5.1f} min)   "
          f"Faktor {r.su_raw:6.1f}")
    p("")

    if not leg.empty:
        p("=" * 82)
        p("ALT-SWEEP (frueherer Solverstand, Abb. 2)")
        p("=" * 82)
        p(f"Dateien: {leg.file.nunique()}, Messkampagnen: {sorted(leg.stamp.unique())}")
        if leg.attrs.get("dropped"):
            p(f"verworfene Doppelmessungen: {leg.attrs['dropped']}")
        p(f"{'n':>5}{'pv%':>5}{'n_pv':>6}{'ms/Szen':>10}{'x PQ':>7}{'NR ms':>8}"
          f"{'t_ges(tau_max)':>16}{'SU':>7}{'div ab tau':>11}{'Kampagne':>17}")
        for _, r in legs.iterrows():
            p(f"{r.n:>5.0f}{r.pv:>5.0f}{r.n_pv:>6.0f}{r.t_scen:>10.4f}"
              f"{r.ratio_pq:>7.2f}{r.t_nr_scen:>8.2f}{r.t_tot_max:>16.1f}"
              f"{r.su_max_tau:>7.2f}"
              f"{(str(int(r.tau_div)) if r.tau_div else '-'):>11}{r.stamp:>17}")
        p("")
        p("--- Alt gegen aktuell (gleiche (n, pv)) ---")
        for _, r in legs.iterrows():
            c = cfg[(cfg.n == r.n) & (cfg.pv == r.pv)]
            if c.empty:
                continue
            c = c.iloc[0]
            p(f"  n={r.n:>4} pv={r.pv:>2}%: TPF {r.t_scen:8.4f} -> {c.t_inf:8.4f} "
              f"ms/Szen (x{r.t_scen/c.t_inf:5.2f})   "
              f"NR {r.t_nr_scen:6.2f} -> {c.t_nr:6.2f} ms "
              f"(x{r.t_nr_scen/c.t_nr:4.2f})")

    txt = "\n".join(L)
    (out / "batch_key_numbers.txt").write_text(txt, encoding="utf-8")
    print(txt)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="tau_benchmark_results/runs_20260818_145947.csv",
                    type=Path)
    ap.add_argument("--legacy-dir",
                    default=r"C:\Users\sgrigorevski-admin\TensorPowerFlow"
                            r"\TensorPowerFlow-on-steroids\results", type=Path)
    ap.add_argument("--outdir", default="../../../Bachelor_tensorflow/figures",
                    type=Path)
    ap.add_argument("--pv", type=int, default=None,
                    help="PV-Anteil in %% fuer Abb. 2 (Standard: automatisch)")
    ap.add_argument("--sizes", nargs="*", type=int, default=None,
                    help="Netzgroessen fuer Abb. 2 (Standard: alle verfuegbaren)")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    df = load_new(a.csv)
    cfg = per_config(df)
    leg = load_legacy(a.legacy_dir)
    legs = legacy_summary(leg) if not leg.empty else pd.DataFrame()

    fig_tau(df, cfg, a.outdir)
    if not leg.empty:
        fig_total_time(leg, a.outdir, pv_pick=a.pv,
                       sizes=tuple(a.sizes) if a.sizes else None)
    fig_pv_effect(cfg, a.outdir)
    write_tables(cfg, a.outdir)
    write_numbers(df, cfg, leg, legs, a.outdir)
    print(f"\n[ok] PGF-Dateien und Kennzahlen in {a.outdir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# lambda_plots.py
"""
Plots zur Untersuchung 'Einfluss des Lastfaktors'.
Aufruf:  python lambda_plots.py results_lastfaktor [--pgf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

CW = 5.9          # Textbreite in Zoll (LaTeX \textwidth)
MARK = {0.0: "o", 0.10: "s", 0.25: "^", 0.50: "D"}


def setup(pgf: bool):
    if pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex", "text.usetex": True,
            "font.family": "serif", "pgf.rcfonts": False,
            "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
        })
    else:
        matplotlib.rcParams.update({"font.size": 9, "legend.fontsize": 7.5})


def colors(vals):
    from matplotlib import colormaps
    vals = sorted(set(vals))
    cmap = colormaps["viridis"].resampled(max(len(vals), 2))
    return {v: cmap(i / max(len(vals) - 1, 1)) for i, v in enumerate(vals)}


def save(fig, out: Path, name, pgf: bool):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / f"{name}.pdf")
    if pgf:
        fig.savefig(out / f"{name}.pgf")
    else:
        fig.show()
    print(f"  -> {name}")
    matplotlib.pyplot.close(fig)


def load(d, name):
    p = Path(d) / name
    return pd.read_csv(p) if p.exists() else None


# ══════════════════════════════════════════════════════════════════════════
def fig01_inner(d, out, pgf):
    e1 = load(d, "e1_inner.csv")
    if e1 is None: return
    import matplotlib.pyplot as plt
    c = colors(e1.n.unique())
    fig, ax = plt.subplots(1, 2, figsize=(CW, 2.5))
    for n, g in e1.groupby("n"):
        g = g.sort_values("lam")
        ok = g[g.conv]
        ax[0].plot(ok.lam, ok.eta_emp, "-o", ms=2.5, color=c[n], label=f"$n={n}$")
        bad = g[~g.conv]
        ax[0].plot(bad.lam, np.minimum(bad.eta2, 5), "x", color="tab:red", ms=5)
        ax[1].plot(ok.lam, ok.k_in_6, "-o", ms=2.5, color=c[n])
        ref = ok.eta_emp.iloc[0] * (ok.lam / ok.lam.iloc[0]) * \
              (ok.v_min.iloc[0] / ok.v_min) ** 2
        ax[0].plot(ok.lam, ref, ":", lw=0.8, color=c[n])
    ax[0].axhline(1.0, ls=":", c="k", lw=0.8)
    ax[0].set(xlabel=r"$\lambda$", ylabel=r"$\eta_{\mathrm{emp}}$", yscale="log", xscale="log")
    ax[1].set(xlabel=r"$\lambda$", ylabel=r"innere Iterationen $k_{\mathrm{in}}$", xscale="log")
    ax[0].legend(frameon=False)
    save(fig, out, "fig01_inner_eta_kin", pgf)


def fig02_collapse(d, out, pgf):
    e1 = load(d, "derived_e1_kappa.csv")
    if e1 is None:
        e1 = load(d, "e1_inner.csv")
    if e1 is None: return
    if "kappa" not in e1.columns:
        conv_arr = np.asarray(e1.conv).ravel() if isinstance(e1.conv, pd.DataFrame) else np.asarray(e1.conv).ravel()
        e1 = e1[conv_arr].copy()
        e1["kappa"] = e1.eta_emp * e1.v_min ** 2 / (e1.lam * e1.z_rel)
    import matplotlib.pyplot as plt
    c = colors(e1.n.unique())
    fig, ax = plt.subplots(1, 2, figsize=(CW, 2.5))
    for n, g in e1.groupby("n"):
        g = g.sort_values("lam")
        ax[0].plot(g.lam, g.eta_emp, "-o", ms=2.5, color=c[n], label=f"$n={n}$")
        ax[1].plot(g.lam, g.kappa, "-o", ms=2.5, color=c[n])
        ax[1].axhline(g.kappa.mean(), ls=":", lw=0.7, color=c[n])
    for a, yl in zip(ax, [r"$\eta_{\mathrm{emp}}$",
                          r"$\kappa=\eta\,v_{\min}^2/(\lambda z_{\mathrm{rel}})$"]):
        a.set(xlabel=r"$\lambda$", ylabel=yl, xscale="log", yscale="log")
    ax[0].legend(frameon=False)
    save(fig, out, "fig02_collapse", pgf)


def fig03_nose(d, out, pgf):
    e1, e3 = load(d, "e1_inner.csv"), load(d, "e3_limits.csv")
    if e1 is None: return
    import matplotlib.pyplot as plt
    c = colors(e1.n.unique())
    fig, ax = plt.subplots(figsize=(CW * 0.62, 2.6))
    for n, g in e1[e1.conv].groupby("n"):
        g = g.sort_values("lam")
        ax.plot(g.lam, g.v_min, "-o", ms=2.5, color=c[n], label=f"$n={n}$")
        if e3 is not None:
            for crit, mk in (("base_fpi", "v"), ("eta_lt_1", "*"), ("nr_pq", "x")):
                r = e3[(e3.n == n) & (e3.criterion == crit)]
                if len(r) and np.isfinite(r.lam_star.iloc[0]):
                    ax.axvline(r.lam_star.iloc[0], color=c[n], lw=0.6,
                               ls={"base_fpi": "-", "eta_lt_1": "--", "nr_pq": ":"}[crit],
                               alpha=0.6)
    ax.set(xlabel=r"$\lambda$", ylabel=r"$v_{\min}$ [p.u.]")
    ax.legend(frameon=False, title="durchgez./gestr./punkt.:\n"
                                   r"$\lambda^\ast$ FPI/$\eta{=}1$/NR", title_fontsize=6)
    save(fig, out, "fig03_nose", pgf)


def fig04_outer(d, out, pgf):
    e2 = load(d, "e2_outer.csv")
    if e2 is None: return
    import matplotlib.pyplot as plt
    e2 = e2[(e2.setpoint_mode == "calibrated") & (e2.base_ok == True)]
    c = colors(e2.n.unique())
    fig, ax = plt.subplots(1, 2, figsize=(CW, 2.6))
    for (n, pv), g in e2.groupby(["n", "pv_share"]):
        g = g.sort_values("lam")
        ok, bad = g[g.tpf_conv == True], g[g.tpf_conv == False]
        m = MARK.get(round(pv, 2), "o")
        ax[0].plot(ok.lam, ok.k_out, "-" + m, ms=3, lw=0.8, color=c[n],
                   label=f"$n={n}$, pv={pv:.2f}")
        ax[0].plot(bad.lam, bad.k_out, "x", color="tab:red", ms=5)
        ax[1].plot(g.lam, g.eps_med, "-" + m, ms=3, lw=0.8, color=c[n])
    lam = np.array(sorted(e2.lam.unique()))
    ax[1].plot(lam, 0.02 * lam ** 2, "-", color="0.6", lw=0.8)
    ax[1].axhline(0.6, ls=":", c="k", lw=0.8)
    ax[0].set(xlabel=r"$\lambda$", ylabel=r"$k_{\mathrm{out}}$", yscale="log")
    ax[1].set(xlabel=r"$\lambda$", ylabel=r"$\varepsilon_{\mathrm{lin}}$ (Median)",
              xscale="log", yscale="log")
    ax[0].legend(frameon=False, ncol=1, fontsize=6)
    save(fig, out, "fig04_outer_kout_epslin", pgf)


def fig05_qdemand(d, out, pgf):
    e2 = load(d, "e2_outer.csv")
    if e2 is None: return
    import matplotlib.pyplot as plt
    e2 = e2[(e2.base_ok == True)]
    c = colors(e2.n.unique())
    fig, ax = plt.subplots(1, 2, figsize=(CW, 2.5))
    for (n, pv), g in e2[e2.setpoint_mode == "calibrated"].groupby(["n", "pv_share"]):
        g = g.sort_values("lam")
        ax[0].plot(g.lam, g.q_max, "-" + MARK.get(round(pv, 2), "o"), ms=3, lw=0.8,
                   color=c[n], label=f"$n={n}$, pv={pv:.2f}")
        ax[1].plot(g.lam, g.q_util, "-" + MARK.get(round(pv, 2), "o"), ms=3, lw=0.8, color=c[n])
    ax[1].axhline(1.0, ls=":", c="k", lw=0.9)
    ax[0].set(xlabel=r"$\lambda$", ylabel=r"$\max_k|Q_k|$ [p.u.]", yscale="log")
    ax[1].set(xlabel=r"$\lambda$", ylabel=r"$\max_k|Q_k|/Q_{\mathrm{verf}}$", yscale="log")
    ax[0].legend(frameon=False, fontsize=6)
    save(fig, out, "fig05_qdemand", pgf)


def fig06_money(d, out, pgf):
    e2, e7 = load(d, "e2_outer.csv"), load(d, "e7_lambda_rx.csv")
    if e2 is None: return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(CW * 0.62, 2.6))
    a = e2[(e2.base_ok == True)].dropna(subset=["eps_med"])
    ax.scatter(a.eps_med, a.k_out, s=10, marker="o", facecolors="none",
               edgecolors="tab:blue", lw=0.6, label=r"$\lambda$-Sweep")
    bad = a[a.tpf_conv == False]
    ax.scatter(bad.eps_med, bad.k_out, s=22, marker="x", c="tab:red", label="nicht konvergent")
    if e7 is not None and len(e7):
        b = e7.dropna(subset=["eps_med"])
        ax.scatter(b.eps_med, b.k_out, s=10, marker="s", facecolors="none",
                   edgecolors="tab:orange", lw=0.6, label=r"$(\lambda,R/X)$-Gitter")
    ax.axvline(0.6, ls=":", c="k", lw=0.9)
    ax.set(xlabel=r"$\varepsilon_{\mathrm{lin}}$ (Median)", ylabel=r"$k_{\mathrm{out}}$",
           xscale="log", yscale="log")
    ax.legend(frameon=False)
    save(fig, out, "fig06_money_epslin_kout", pgf)


def fig07_optim(d, out, pgf):
    e5 = load(d, "e5_optim.csv")
    if e5 is None: return
    import matplotlib.pyplot as plt
    d5 = e5[e5.conv == True].copy()
    d5["var"] = np.where(d5.warm, "warm", "cold") + "/" + np.where(d5.adaptive, "adapt", "fix")
    p = d5.pivot_table(index=["n", "pv_share", "lam"], columns="var", values="k_in").reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(CW, 2.5))
    if {"cold/fix", "warm/fix", "warm/adapt"} <= set(p.columns):
        p["gain_warm"] = p["cold/fix"] / p["warm/fix"]
        p["gain_total"] = p["cold/fix"] / p["warm/adapt"]
        c = colors(p.n.unique())
        for n, g in p.groupby("n"):
            g = g.sort_values("lam")
            ax[0].plot(g.lam, g.gain_warm, "o", ms=3, color=c[n], label=f"$n={n}$")
            ax[1].plot(g.lam, g.gain_total, "o", ms=3, color=c[n])
        for a, t in zip(ax, ["Warm vs. Cold", "Warm+adaptiv vs. Cold+fix"]):
            a.axhline(1.0, ls=":", c="k", lw=0.8)
            a.set(xlabel=r"$\lambda$", ylabel=r"Einsparung $k_{\mathrm{in}}$", title=t)
            a.title.set_size(8)
        ax[0].legend(frameon=False)
    save(fig, out, "fig07_optim_savings", pgf)


def fig08_timing(d, out, pgf):
    e2 = load(d, "e2_outer.csv")
    if e2 is None or "t_nr_ms" not in e2: return
    import matplotlib.pyplot as plt
    a = e2[(e2.tpf_conv == True) & (e2.setpoint_mode == "calibrated")].dropna(subset=["t_nr_ms"])
    if not len(a): return
    c = colors(a.n.unique())
    fig, ax = plt.subplots(figsize=(CW * 0.62, 2.6))
    for n, g in a.groupby("n"):
        g = g.sort_values("lam")
        ax.plot(g.lam, g.t_tpf_ms, "-o", ms=2.5, color=c[n], label=f"TPF $n={n}$")
        ax.plot(g.lam, g.t_nr_ms, "--", lw=0.8, color=c[n], label=f"NR $n={n}$")
    ax.set(xlabel=r"$\lambda$", ylabel="Rechenzeit [ms]", yscale="log")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    save(fig, out, "fig08_timing", pgf)


def fig09_heatmap(d, out, pgf):
    e2 = load(d, "e2_outer.csv")
    if e2 is None: return
    import matplotlib.pyplot as plt
    a = e2[(e2.setpoint_mode == "calibrated") & (e2.base_ok == True)].copy()
    a["k_tot"] = a.k_in + a.k_out
    a.loc[a.tpf_conv != True, "k_tot"] = np.nan
    piv = a.pivot_table(index="lam", columns="n", values="k_tot")
    fig, ax = plt.subplots(figsize=(CW * 0.62, 2.8))
    im = ax.pcolormesh(np.arange(len(piv.columns) + 1),
                       np.arange(len(piv.index) + 1),
                       piv.values, shading="flat", cmap="magma_r")
    ax.set_xticks(np.arange(len(piv.columns)) + 0.5)
    ax.set_xticklabels([str(c) for c in piv.columns])
    ax.set(xlabel=r"$n_{\mathrm{bus}}$", ylabel=r"$\lambda$")
    fig.colorbar(im, ax=ax, label=r"$k_{\mathrm{in}}+k_{\mathrm{out}}$")
    save(fig, out, "fig09_heatmap_n_lambda", pgf)


def fig10_damping(d, out, pgf):
    e4 = load(d, "e4_damping.csv")
    if e4 is None: return
    import matplotlib.pyplot as plt
    c = colors(e4.n.unique())
    fig, ax = plt.subplots(figsize=(CW * 0.62, 2.5))
    for (n, pv), g in e4.groupby(["n", "pv_share"]):
        g = g.sort_values("omega")
        ax.plot(g.omega, g.lam_star, "-" + MARK.get(round(pv, 2), "o"), ms=3.5, lw=0.8,
                color=c[n], label=f"$n={n}$, pv={pv:.2f}")
    ax.set(xlabel=r"$\omega$", ylabel=r"$\lambda^\ast$")
    ax.legend(frameon=False, fontsize=6)
    save(fig, out, "fig10_damping", pgf)


def fig11_modes(d, out, pgf):
    e9 = load(d, "e9_modes.csv")
    if e9 is None or not len(e9): return
    import matplotlib.pyplot as plt
    a = e9[e9.base_ok == True]
    fig, ax = plt.subplots(1, 2, figsize=(CW, 2.5))
    sty = {"load": "-o", "load_pv": "--s", "pv_only": ":^"}
    c = colors(a.n.unique())
    for (n, mode), g in a.groupby(["n", "scale_mode"]):
        g = g.sort_values("lam")
        ax[0].plot(g.lam, g.k_out, sty.get(mode, "-o"), ms=3, lw=0.8, color=c[n],
                   label=f"$n={n}$/{mode}")
        ax[1].plot(g.lam, g.eps_med, sty.get(mode, "-o"), ms=3, lw=0.8, color=c[n])
    ax[0].set(xlabel=r"$\lambda$", ylabel=r"$k_{\mathrm{out}}$", yscale="log")
    ax[1].set(xlabel=r"$\lambda$", ylabel=r"$\varepsilon_{\mathrm{lin}}$", yscale="log")
    ax[0].legend(frameon=False, fontsize=5.5, ncol=2)
    save(fig, out, "fig11_modes", pgf)


def fig12_continuation(d, out, pgf):
    e6 = load(d, "e6_continuation.csv")
    if e6 is None or not len(e6): return
    import matplotlib.pyplot as plt
    a = e6[e6.conv == True]
    c = colors(a.n.unique())
    fig, ax = plt.subplots(1, 2, figsize=(CW, 2.5))
    for (n, mode), g in a.groupby(["n", "q_init_mode"]):
        g = g.groupby("lam").median(numeric_only=True).reset_index()
        ls = "-o" if mode == "cold_q0" else "--s"
        ax[0].plot(g.lam, g.k_out, ls, ms=3, lw=0.8, color=c[n], label=f"$n={n}$/{mode}")
        ax[1].plot(g.lam, g.eps_med, ls, ms=3, lw=0.8, color=c[n])
    ax[0].set(xlabel=r"$\lambda$", ylabel=r"$k_{\mathrm{out}}$")
    ax[1].set(xlabel=r"$\lambda$", ylabel=r"$\varepsilon_{\mathrm{lin}}$", yscale="log")
    ax[0].legend(frameon=False, fontsize=6)
    save(fig, out, "fig12_continuation", pgf)


FIGS = [fig01_inner, fig02_collapse, fig03_nose, fig04_outer, fig05_qdemand,
        fig06_money, fig07_optim, fig08_timing, fig09_heatmap, fig10_damping,
        fig11_modes, fig12_continuation]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="results_lastfaktor")
    ap.add_argument("--pgf", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    setup(a.pgf)
    out = Path(a.out) if a.out else Path(a.data) / "figures"
    print(f"Plots -> {out}")
    for f in FIGS:
        try:
            f(a.data, out, a.pgf)
        except Exception as e:
            print(f"  {f.__name__} übersprungen: {e!r}")


if __name__ == "__main__":
    main()
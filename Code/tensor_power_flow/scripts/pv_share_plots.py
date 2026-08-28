# =============================================================================
# pv_share_plots.py  —  P1..P9
# =============================================================================
"""
Aufruf:  python pv_share_plots.py --data results_pv_share --fig figures_pv_share
Optional: --pgf  (zusaetzlich .pgf fuer LaTeX)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLR = {40: "#C0392B", 120: "#8E44AD", 200: "#1F6FB2",
       350: "#16A085", 500: "#2E8B57", 1000: "#D68910"}
MRK = {"random": "o", "clustered": "s", "spread": "^", "leaves": "D", "feeders": "v"}


def style(pgf: bool):
    mpl.rcParams.update({
        "figure.figsize": (7.0, 3.0), "font.size": 9, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.markersize": 3.5,
        "lines.linewidth": 1.2, "legend.frameon": False, "legend.fontsize": 7.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    if pgf:
        mpl.rcParams.update({"pgf.texsystem": "pdflatex", "text.usetex": False,
                            "pgf.rcfonts": False})


def save(fig, out: Path, name: str, pgf: bool):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    if pgf:
        fig.savefig(out / f"{name}.pgf")
    plt.close(fig)
    print(f"  {name}")


def load(d: Path, name: str):
    p = d / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


# --- P1 -----------------------------------------------------------------------
def p1(e1, out, pgf):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for n, g in e1.groupby("n_bus"):
        c = CLR.get(n, "k")
        for var, ls in (("coupled", "-"), ("decoupled", "--")):
            gv = g[g.variant == var]
            for k, a in ((("k_out"), ax[0]), (("k_in"), ax[1])):
                med = gv[gv.converged].groupby("n_pv")[k].median()
                a.plot(med.index, med.values, ls, color=c, marker="o",
                       label=f"$n={n}$, {'gek.' if var=='coupled' else 'entk.'}")
            bad = gv[~gv.converged]
            if len(bad):
                ax[0].plot(bad.n_pv, np.full(len(bad), bad.k_out.max()),
                           "x", color=c, ms=5, mew=1.2)
    for a, lab in zip(ax, [r"$k_\mathrm{out}$", r"$k_\mathrm{in}$"]):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(r"$n_\mathrm{pv}$"); a.set_ylabel(lab)
    ax[0].legend(ncol=2)
    fig.suptitle("P1  Aufwand über der PV-Knotenzahl (Kreuze: keine Konvergenz)",
                 fontsize=8)
    save(fig, out, "p1_kout_kin_vs_npv", pgf)


# --- P2 -----------------------------------------------------------------------
def p2(e1, out, pgf):
    g = e1[(e1.variant == "coupled") & e1.converged]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for n, gg in g.groupby("n_bus"):
        c = CLR.get(n, "k")
        m1 = gg.groupby("share")["k_out"].median()
        m2 = gg.groupby("n_pv")["k_out"].median()
        ax[0].plot(m1.index, m1.values, "-o", color=c, label=f"$n={n}$")
        ax[1].plot(m2.index, m2.values, "-o", color=c, label=f"$n={n}$")
    ax[0].set_xlabel(r"PV-Anteil $n_\mathrm{pv}/(n-1)$")
    ax[1].set_xlabel(r"$n_\mathrm{pv}$"); ax[1].set_xscale("log")
    for a in ax:
        a.set_ylabel(r"$k_\mathrm{out}$")
    ax[0].set_title("(a) über Anteil", fontsize=8)
    ax[1].set_title("(b) über absoluter PV-Zahl", fontsize=8)
    ax[1].legend()
    fig.suptitle("P2  Kollapstest zu H1", fontsize=8)
    save(fig, out, "p2_collapse_share_vs_npv", pgf)


# --- P3 -----------------------------------------------------------------------
def p3(e3, out, pgf):
    if e3 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for st, g in e3.groupby("placement"):
        gc = g[g.variant == "coupled"]
        gd = g[g.variant == "decoupled"]
        ax[0].scatter(gc.offdiag_share, gc.rho_jacobi, marker=MRK.get(st, "o"),
                      s=18, label=st)
        ax[1].scatter(gd.rho_jacobi, gd.k_out, marker=MRK.get(st, "o"), s=18,
                      facecolors=np.where(gd.converged, "none", "r"),
                      edgecolors="k", linewidths=0.6, label=st)
    ax[0].axhline(1.0, color="0.4", ls=":")
    ax[0].set_xlabel("Nebendiagonalanteil von $X_{pp}$")
    ax[0].set_ylabel(r"$\rho_\mathrm{J}$"); ax[0].set_yscale("log")
    ax[0].legend(ncol=2)
    ax[1].axvline(1.0, color="0.4", ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\rho_\mathrm{J}$")
    ax[1].set_ylabel(r"$k_\mathrm{out}$ (entkoppelt)")
    fig.suptitle("P3  Platzierung, Kopplungsmaß und Versagen der entkoppelten "
                 "Näherung (H3)", fontsize=8)
    save(fig, out, "p3_placement_coupling", pgf)


# --- P4 -----------------------------------------------------------------------
def p4(e4, out, pgf):
    if e4 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for n, g in e4[e4.placement == "random"].groupby("n_bus"):
        c = CLR.get(n, "k")
        m = g.groupby("n_pv")[["cond", "rho_jacobi"]].median()
        ax[0].plot(m.index, m["cond"], "-o", color=c, label=f"$n={n}$")
        ax[1].plot(m.index, m["rho_jacobi"], "-o", color=c, label=f"$n={n}$")
    for st, g in e4[e4.n_bus == e4.n_bus.max()].groupby("placement"):
        m = g.groupby("n_pv")["rho_jacobi"].median()
        ax[1].plot(m.index, m.values, ":", marker=MRK.get(st, "o"),
                   color="0.35", ms=3, label=f"{st} ($n$ max)")
    ax[1].axhline(1.0, color="r", ls="--", lw=0.8)
    for a, lab in zip(ax, [r"$\mathrm{cond}(X_{pp})$", r"$\rho_\mathrm{J}$"]):
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(r"$n_\mathrm{pv}$"); a.set_ylabel(lab)
    ax[0].legend(); ax[1].legend(ncol=2, fontsize=6)
    fig.suptitle("P4  Struktur von $X_{pp}$ (rein numerisch, ohne Solverlauf)",
                 fontsize=8)
    save(fig, out, "p4_xpp_structure", pgf)


# --- P5 -----------------------------------------------------------------------
def p5(e5, out, pgf):
    if e5 is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9))
    for (mode, d), g in e5.groupby(["sp_mode", "delta"]):
        lab = (r"$\delta=%.3f$" % d) if mode == "delta" else "abs. 1,00 p.u."
        ls = "-" if mode == "delta" else "--"
        g = g[g.converged].sort_values("n_pv")
        ax[0].plot(g.n_pv, g.q_max, ls, marker="o", label=lab)
        ax[1].plot(g.n_pv, g.q_med, ls, marker="o", label=lab)
        ax[2].plot(g.n_pv, g.k_out, ls, marker="o", label=lab)
    for a, lab in zip(ax, [r"$\max_k|Q_k|$ [p.u.]", r"$\mathrm{med}_k|Q_k|$ [p.u.]",
                           r"$k_\mathrm{out}$"]):
        a.set_xscale("log"); a.set_xlabel(r"$n_\mathrm{pv}$"); a.set_ylabel(lab)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].legend(fontsize=6)
    fig.suptitle("P5  Blindleistungsbedarf und Sollwertdefinition (H4/H5)", fontsize=8)
    save(fig, out, "p5_q_vs_npv", pgf)


# --- P6 -----------------------------------------------------------------------
def p6(e6, out, pgf):
    if e6 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for q, g in e6.groupby("q_lim_pu", dropna=False):
        lab = "ohne Grenze" if not np.isfinite(q) else f"$|Q|\\leq{q}$ p.u."
        g = g.sort_values("n_pv")
        ax[0].plot(g.n_pv, g.sat_share, "-o", label=lab)
        ax[1].plot(g.n_pv, g.k_out, "-o", label=lab)
        bad = g[~g.converged]
        ax[1].plot(bad.n_pv, bad.k_out, "x", color="r", ms=5)
    ax[0].set_ylabel("Anteil gesättigter PV-Knoten")
    ax[1].set_ylabel(r"$k_\mathrm{out}$")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel(r"$n_\mathrm{pv}$")
    ax[1].legend()
    fig.suptitle("P6  Blindleistungsgrenzen als praktische Grenze", fontsize=8)
    save(fig, out, "p6_qlims", pgf)


# --- P7 -----------------------------------------------------------------------
def p7(e7, out, pgf):
    if e7 is None:
        return
    pl = e7[e7.tau == e7.tau.max()]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for n, g in pl.groupby("n_bus"):
        g = g.sort_values("n_pv")
        ax[0].plot(g.n_pv, g.t_per_scen_ms, "-o", color=CLR.get(n, "k"),
                   label=f"$n={n}$")
        for _, r in g.iterrows():
            ax[0].annotate(f"{int(r.k_in)}", (r.n_pv, r.t_per_scen_ms),
                           fontsize=5.5, xytext=(2, 3), textcoords="offset points")
        b = g[g.n_pv == 0]
        if len(b):
            ax[1].scatter(g.k_in / b.k_in.values[0], g.t_per_scen_ms / b.t_per_scen_ms.values[0],
                          color=CLR.get(n, "k"), s=22, label=f"$n={n}$")
    lim = [0.8, max(2.0, ax[1].get_xlim()[1])]
    ax[1].plot(lim, lim, "k:", lw=0.8)
    ax[0].set_xlabel(r"$n_\mathrm{pv}$"); ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$t_\infty$ pro Szenario [ms]"); ax[0].legend()
    ax[1].set_xlabel(r"$k_\mathrm{in}/k_\mathrm{in}^\mathrm{PQ}$")
    ax[1].set_ylabel(r"$t_\infty/t_\infty^\mathrm{PQ}$"); ax[1].legend()
    fig.suptitle("P7  PV wirkt ausschließlich über die Iterationszahl (H6)", fontsize=8)
    save(fig, out, "p7_time_vs_iterations", pgf)


# --- P8 -----------------------------------------------------------------------
def p8(e8, out, pgf):
    if e8 is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for a, axis, xl in ((ax[0], "lam", r"$\lambda$"), (ax[1], "rho", r"$R/X$")):
        g = e8[e8.axis == axis]
        piv = g.pivot_table(index="share", columns=axis, values="k_out",
                            aggfunc="median")
        cv = g.pivot_table(index="share", columns=axis, values="converged",
                           aggfunc="min")
        im = a.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis")
        a.set_xticks(range(len(piv.columns)))
        a.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=6)
        a.set_yticks(range(len(piv.index)))
        a.set_yticklabels([f"{i:.2f}" for i in piv.index], fontsize=6)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                bad = not bool(cv.values[i, j])
                a.text(j, i, "x" if bad else f"{v:.0f}", ha="center", va="center",
                       fontsize=5.5, color="r" if bad else "w")
        a.set_xlabel(xl); a.set_ylabel("PV-Anteil"); a.grid(False)
        fig.colorbar(im, ax=a, label=r"$k_\mathrm{out}$")
    fig.suptitle("P8  Interaktion PV-Anteil mit Lastfaktor und $R/X$", fontsize=8)
    save(fig, out, "p8_interaction_heatmaps", pgf)


# --- P9 -----------------------------------------------------------------------
def p9(d: Path, out, pgf):
    f = d / "xpp_matrices.npz"
    if not f.exists():
        return
    z = np.load(f)
    keys = list(z.keys())[:4]
    fig, ax = plt.subplots(1, len(keys), figsize=(2.0 * len(keys), 2.4))
    ax = np.atleast_1d(ax)
    vmax = max(np.abs(z[k]).max() for k in keys)
    for a, k in zip(ax, keys):
        im = a.imshow(np.abs(z[k]), cmap="magma", vmin=0, vmax=vmax)
        a.set_title(k.split("_")[-1], fontsize=7)
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax.tolist(), label=r"$|X_{pp,kj}|$", fraction=0.03)
    fig.suptitle("P9  $X_{pp}$ bei gleicher PV-Zahl, verschiedener Platzierung",
                 fontsize=8)
    save(fig, out, "p9_xpp_heatmaps", pgf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results_pv_share")
    ap.add_argument("--fig", default="figures_pv_share")
    ap.add_argument("--pgf", action="store_true")
    a = ap.parse_args()
    style(a.pgf)
    d, out = Path(a.data), Path(a.fig)
    e1, e3, e4 = load(d, "e1"), load(d, "e3"), load(d, "e4")
    e5, e6, e7, e8 = load(d, "e5"), load(d, "e6"), load(d, "e7"), load(d, "e8")
    print("Plots:")
    if e1 is not None:
        p1(e1, out, a.pgf); p2(e1, out, a.pgf)
    p3(e3, out, a.pgf); p4(e4, out, a.pgf); p5(e5, out, a.pgf)
    p6(e6, out, a.pgf); p7(e7, out, a.pgf); p8(e8, out, a.pgf)
    p9(d, out, a.pgf)


if __name__ == "__main__":
    main()
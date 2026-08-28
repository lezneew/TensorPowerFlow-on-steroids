# tensor_power_flow/scripts/plot_rx_figures.py
"""
Erzeugt genau die Abbildungen und Tabellen, die Abschnitt sec:rx verwendet.

Abbildungen (Hauptteil)   Tabellen (Hauptteil)      Tabellen (Anhang)
  rx_inner_invariance       tab_rx_inner              tab_rx_outer_fail  (A.1)
  rx_collapse               tab_rx_outer_struct       tab_rx_predictor   (A.2)
  rx_outer_iterations       tab_rx_outer              tab_rx_nr          (A.3)

Zusatz: rx_facts.csv (alle im Text zitierten Zahlen),
        rx_text_check.csv (Soll-Ist gegen die Textbehauptungen),
        rx_report.md

Aufruf:
    python -m scripts.plot_rx_figures --csv results/rx_sweep.csv --pgf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

OUT_DEFAULT = Path(r"C:\Users\sgrigorevski-admin\TensorPowerFlow"
                   r"\TensorPowerFlow-on-steroids\Bachelor_tensorflow\figures")

OUTER_CAP = 60        # max_outer aus run_rx_sweep
INNER_CAP = 500       # max_inner der Ratenmessung
V_TOL_SLOW = 1e-4     # Grenzzyklus vs. echte Divergenz
SENS_LIMIT = 0.6      # Fruehwarnschwelle eps_lin
PLATEAU_RHO = 0.68    # obere Grenze des Plateaubereichs in tab:rx-outer
RHO_COLS = (2.15, 6.81)   # Stuetzstellen in tab:rx-outer

MODE = {"const_z": dict(ls="-", marker="o", fill=True, tex=r"const\_z"),
        "const_x": dict(ls="--", marker="s", fill=False, tex=r"const\_x")}
# bewusst nicht 's': das Quadrat ist in Abb. 1/2 fuer const_x reserviert
PV_MARKER = {0.10: "o", 0.25: "^", 0.50: "D"}
XKK_CANDS = ("x_kk_mean", "xpp_diag_mean", "mean_x_kk", "x_diag_mean")

FACTS: list[dict] = []
TABLES: dict[str, pd.DataFrame] = {}
OUT = OUT_DEFAULT


# ----------------------------------------------------------------------
# Setup, Helfer
# ----------------------------------------------------------------------
def setup_mpl(use_pgf: bool):
    if use_pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex", "font.family": "serif",
            "text.usetex": True, "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[utf8]{inputenc}"
                            r"\usepackage[T1]{fontenc}"})
    matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 8,
                                "legend.fontsize": 6.5})
    global plt, VIR
    import matplotlib.pyplot as plt  # noqa: E402
    try:
        from matplotlib import colormaps
        VIR = colormaps["viridis"]
    except Exception:
        from matplotlib.cm import viridis as VIR  # noqa
    globals()["USE_PGF"] = use_pgf


def fact(key, value, unit="", note=""):
    FACTS.append({"key": key, "value": value, "unit": unit, "note": note})


def col(df, name, default=np.nan):
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index, dtype="float64")


def first_col(df, cands):
    for c in cands:
        if c in df.columns and df[c].notna().any():
            return df[c]
    return pd.Series(np.nan, index=df.index, dtype="float64")


def spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 4 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return np.nan
    r = s.rank()
    return float(np.corrcoef(r["a"], r["b"])[0, 1])


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def node_color(n, nodes):
    nodes = list(nodes)
    return VIR(nodes.index(n) / max(len(nodes) - 1, 1))


def _fig(nrows=1, ncols=1, w=5.91, h=2.7, **kw):
    return plt.subplots(nrows, ncols, figsize=(w, h * nrows),
                        constrained_layout=True, **kw)


def _save(fig, name):
    for ext in ("pgf", "pdf"):
        try:
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
        except Exception as e:
            print(f"  savefig {name}.{ext}: {type(e).__name__}: {e}")
    print(f"  -> {name}.pgf/.pdf")
    plt.close(fig) if globals().get("USE_PGF") else plt.show()


def save_table(df, name, caption, label, fmt=None, note=None, header=None):
    """Schreibt CSV, reine Datenzeilen (_rows.tex) und volle table-Umgebung."""
    df = df.reset_index(drop=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    fmt = fmt or {}

    def cell(c, v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "{--}"
        if isinstance(v, str):
            return v
        f = fmt.get(c)
        if f is None:
            return f"{v:g}"
        return f(v) if callable(f) else format(v, f)

    rows = [" & ".join(cell(c, r[c]) for c in df.columns) + r" \\"
            for _, r in df.iterrows()]
    (OUT / f"{name}_rows.tex").write_text("\n".join(rows) + "\n", "utf-8")

    head = header or list(df.columns)
    spec = "l" * len(df.columns)
    tex = ["% auto-generiert von plot_rx_figures.py",
           r"\begin{table}[htbp]", r"\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(str(h) for h in head) + r" \\", r"\midrule",
           *rows, r"\bottomrule", r"\end{tabular}"]
    if note:
        tex += [r"", r"\smallskip", r"\footnotesize", r"\raggedright", note]
    tex += [r"\end{table}", ""]
    (OUT / f"{name}.tex").write_text("\n".join(tex), "utf-8")
    TABLES[name] = df
    print(f"  -> {name}.csv/.tex/_rows.tex")


# ----------------------------------------------------------------------
# Laden, Klassifikation
# ----------------------------------------------------------------------
NUM = ["nodes", "pv_ratio", "n_pv", "rx", "load_factor", "r_ohm_km",
       "x_ohm_km", "z_abs_ohm_km", "z_rel", "nr_iter", "nr_v_min",
       "t_inner_s", "t_outer_s", "inner_iter_pq", "v_min", "eta_emp",
       "eta_1", "eta_2", "eta_inf", "eta_bound", "eta_fit", "eta_fit_r2",
       "kappa_emp", "cond_xpp", "rho_jacobi", "diag_dom_min", "outer_iter",
       "inner_total", "v_err_final", "q_max_final", "sens_error_first",
       "sens_error_median", "sens_error_max"]
BOOL = ["outer_conv", "inner_conv_pq", "nr_conv"]


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for c in BOOL:
        if c in df:
            df[c] = (df[c].map({True: True, False: False, "True": True,
                                "False": False, 1: True, 0: False})
                     .astype("boolean"))
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["variant"] = df.get("variant", "coupled").fillna("coupled")
    df["mode"] = df["mode"].astype(str)
    df["skipped"] = (df.get("skipped", "").fillna("").astype(str)
                     .replace("nan", ""))
    if "z_rel" not in df or df["z_rel"].isna().all():
        base = df.loc[df["mode"] == "const_z", "z_abs_ohm_km"].median()
        df["z_rel"] = df["z_abs_ohm_km"] / base
    df["x_kk_mean"] = first_col(df, XKK_CANDS)

    # publizierte Rate ist eta_emp (so beschriftet in fig:rx-inner);
    # der Fit dient nur der Konsistenzpruefung
    df["eta_pub"] = col(df, "eta_emp")
    both = col(df, "eta_fit").notna() & df["eta_pub"].notna()
    if both.any():
        dev = (col(df, "eta_fit")[both] / df["eta_pub"][both] - 1).abs()
        fact("inner.eta_fit_vs_emp_max_pct", round(100 * float(dev.max()), 3),
             "%", "Konsistenz Log-Fit gegen empirische Rate")
    if "eta_fit_r2" in df:
        r2 = col(df, "eta_fit_r2").dropna()
        if len(r2):
            fact("inner.fit_r2_min", float(f"{r2.min():.6f}"), "",
                 "kleinstes R^2 des geometrischen Fits")

    df["kappa"] = df["eta_pub"] * df["v_min"] ** 2 / df["z_rel"]
    df["inner_per_outer"] = col(df, "inner_total") / col(df, "outer_iter")
    df["ms_per_inner"] = 1e3 * col(df, "t_outer_s") / col(df, "inner_total")

    conv = col(df, "outer_conv").fillna(False).astype(bool)
    at_cap = col(df, "outer_iter").fillna(0) >= OUTER_CAP
    skipped = df["skipped"] != ""
    df["at_cap"] = at_cap
    df["failed"] = (~conv | at_cap) & ~skipped          # zaehlt fuer rho*
    df["cls"] = np.select(
        [skipped, ~df["failed"],
         col(df, "v_err_final") <= V_TOL_SLOW],
        ["skipped", "conv", "Grenzzyklus"], "Divergenz")
    return df


def pq(df):
    return (df[df["pv_ratio"] == 0]
            .drop_duplicates(subset=["nodes", "mode", "rx"]).copy())


def pv(df, variant=None, mode=None):
    d = df[df["pv_ratio"] > 0]
    if variant:
        d = d[d["variant"] == variant]
    if mode:
        d = d[d["mode"] == mode]
    return d.copy()


# ----------------------------------------------------------------------
# Abbildung 1: fig:rx-inner
# ----------------------------------------------------------------------
def fig_inner(df, inset=True):
    d = pq(df)
    if d.empty:
        print("  fig_inner: keine PQ-Daten"); return
    ok = col(d, "inner_conv_pq").fillna(False).astype(bool)
    nodes = sorted(d["nodes"].unique())
    fig, ax = _fig(1, 2, h=2.7)

    for m, st in MODE.items():
        for n in nodes:
            g = d[(d["nodes"] == n) & (d["mode"] == m)].sort_values("rx")
            if g.empty:
                continue
            c = node_color(n, nodes)
            gk = g[ok.reindex(g.index, fill_value=False)]
            kw = dict(color=c, ls=st["ls"], lw=1.2, marker=st["marker"],
                      ms=4, mfc=c if st["fill"] else "none")
            ax[0].plot(gk["rx"], gk["eta_pub"], **kw)
            ax[1].plot(gk["rx"], gk["inner_iter_pq"], **kw)
            gb = g[~ok.reindex(g.index, fill_value=False)]
            if len(gb):
                ax[0].plot(gb["rx"], gb["eta_2"], "x", color="crimson",
                           ms=6, mew=1.4, zorder=5)
                ax[1].plot(gb["rx"], np.full(len(gb), INNER_CAP), "x",
                           color="crimson", ms=6, mew=1.4, zorder=5)

    # graue Referenz, verankert am const_x-Lauf mit den meisten Punkten
    gx = d[(d["mode"] == "const_x") & ok.reindex(d.index, fill_value=False)]
    if len(gx) > 1:
        n_ref = gx["nodes"].value_counts().idxmax()
        gr = gx[gx["nodes"] == n_ref].sort_values("rx")
        r0, e0 = gr["rx"].iloc[0], gr["eta_pub"].iloc[0]
        rr = np.geomspace(d["rx"].min(), d["rx"].max(), 200)
        ax[0].plot(rr, e0 * np.sqrt(1 + rr ** 2) / np.sqrt(1 + r0 ** 2),
                   color="0.55", lw=.9, zorder=0)
    ax[0].axhline(1.0, ls=":", c="k", lw=.8)
    ax[0].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel=r"$\eta_{\mathrm{emp}}$",
              title="(a) Kontraktionsrate")
    ax[1].axhline(INNER_CAP, ls=":", c="k", lw=.8)
    ax[1].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel="innere Iterationen", title="(b) Iterationszahl")
    for a in ax:
        a.grid(alpha=.3, which="both")

    h_m = [plt.Line2D([], [], color="k", ls=st["ls"], marker=st["marker"],
                      mfc="k" if st["fill"] else "none", label=st["tex"])
           for st in MODE.values()]
    h_m.append(plt.Line2D([], [], color="0.55", lw=.9,
                          label=r"$\propto\sqrt{1+(R/X)^2}$"))
    h_n = [plt.Line2D([], [], color=node_color(n, nodes), lw=1.5,
                      label=f"$n={n}$") for n in nodes]
    ax[0].legend(handles=h_m, loc="upper left")
    ax[1].legend(handles=h_n, loc="upper left")

    if inset:
        _inset_crossing(ax[0], d, ok, nodes)
    _save(fig, "rx_inner_invariance")
    _facts_inner(d, ok, nodes)


def _inset_crossing(a, d, ok, nodes):
    """Zoom auf den Punkt, an dem |z| in beiden Modi uebereinstimmt."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    cx = d[d["mode"] == "const_x"]
    if cx.empty:
        return
    rho_c = float(cx.loc[(cx["z_rel"] - 1.0).abs().idxmin(), "rx"])
    n0 = min(nodes)
    win = d[(d["nodes"] == n0) & (d["rx"] <= 2.2 * rho_c)
            & ok.reindex(d.index, fill_value=False)]
    if len(win) < 4:
        return
    axi = inset_axes(a, width="24%", height="26%", loc="upper center")
    for m, st in MODE.items():
        g = win[win["mode"] == m].sort_values("rx")
        c = node_color(n0, nodes)
        axi.plot(g["rx"], g["eta_pub"], color=c, ls=st["ls"], lw=1.1,
                 marker=st["marker"], ms=3.2,
                 mfc=c if st["fill"] else "none")
    axi.axvline(rho_c, ls=":", c="0.4", lw=.8)
    lo, hi = win["eta_pub"].min(), win["eta_pub"].max()
    pad = 0.04 * (hi - lo) if hi > lo else 0.02 * hi
    axi.set(xscale="log", xlim=(win["rx"].min() * .95, 2.2 * rho_c),
            ylim=(lo - pad, hi + pad))
    axi.tick_params(labelsize=5)
    axi.grid(alpha=.2, which="both")
    mark_inset(a, axi, loc1=3, loc2=4, fc="none", ec="0.4", lw=0.8)
    fact("inner.crossing.rho", round(rho_c, 3), "", "|z| in beiden Modi gleich")


def _facts_inner(d, ok, nodes):
    for (n, m), g in d.groupby(["nodes", "mode"]):
        g = g[ok.reindex(g.index, fill_value=False)].sort_values("rx")
        if len(g) < 2:
            continue
        e0, e1 = g["eta_pub"].iloc[0], g["eta_pub"].iloc[-1]
        v0, v1 = g["v_min"].iloc[0], g["v_min"].iloc[-1]
        z0, z1 = g["z_rel"].iloc[0], g["z_rel"].iloc[-1]
        tag = f"{m}.n{int(n)}"
        fact(f"inner.eta_lo.{tag}", float(f"{e0:.5f}"))
        fact(f"inner.eta_hi.{tag}", float(f"{e1:.5f}"),
             note=f"bei rho={g['rx'].iloc[-1]:g}")
        fact(f"inner.eta_growth.{tag}", round(e1 / e0, 2))
        fact(f"inner.eta_rise_pct.{tag}", round(100 * (e1 / e0 - 1), 1), "%")
        fact(f"inner.vmin_lo.{tag}", round(float(v0), 4), "p.u.")
        fact(f"inner.vmin_hi.{tag}", round(float(v1), 4), "p.u.")
        fact(f"inner.k_lo.{tag}", float(g["inner_iter_pq"].iloc[0]))
        fact(f"inner.k_hi.{tag}", float(g["inner_iter_pq"].iloc[-1]))
        pred = e0 * (z1 / z0) * (v0 / v1) ** 2
        fact(f"inner.eta_pred_hi.{tag}", float(f"{pred:.5f}"),
             note="eta_lo * z_rel-Verhaeltnis / v_min^2-Verhaeltnis")
        fact(f"inner.pred_err_pct.{tag}", round(100 * abs(pred / e1 - 1), 2), "%")
        fact(f"inner.vmin_factor.{tag}", round(float((v0 / v1) ** 2), 3))
        # Iterationszahl aus der Rate
        for lbl, row in (("lo", g.iloc[0]), ("hi", g.iloc[-1])):
            k_pred = np.log(1e-12) / np.log(row["eta_pub"])
            fact(f"inner.k_pred_{lbl}.{tag}", round(float(k_pred), 1), "",
                 "ln(tol)/ln(eta)")
    # Schnittpunkt: Raten beider Modi bei gleichem |z|
    if "inner.crossing.rho" in {f["key"] for f in FACTS}:
        rho_c = [f["value"] for f in FACTS if f["key"] == "inner.crossing.rho"][0]
        n0 = min(nodes)
        sub = d[(d["nodes"] == n0) & np.isclose(d["rx"], rho_c)]
        vals = {r["mode"]: r["eta_pub"] for _, r in sub.iterrows()}
        for m, v in vals.items():
            fact(f"inner.crossing.eta.{m}", float(f"{v:.5f}"))
        if len(vals) == 2:
            a, b = list(vals.values())
            fact("inner.crossing.dev_pct", round(100 * abs(a / b - 1), 2), "%")
    okd = d[ok.reindex(d.index, fill_value=False)]
    if len(okd):
        fact("inner.eta_max_success", round(float(okd["eta_pub"].max()), 3))
    for _, r in d[~ok.reindex(d.index, fill_value=False)].iterrows():
        tag = f"{r['mode']}.n{int(r['nodes'])}.rho{r['rx']:g}"
        for k in ("eta_1", "eta_2", "eta_inf", "v_min"):
            if np.isfinite(r.get(k, np.nan)):
                fact(f"inner.nonconv.{k}.{tag}", round(float(r[k]), 3))


# ----------------------------------------------------------------------
# Abbildung 2: fig:rx-collapse  (redundanzfrei: (a) kappa, (b) kappa-bar(n))
# ----------------------------------------------------------------------
def fig_collapse(df):
    d = pq(df)
    ok = col(d, "inner_conv_pq").fillna(False).astype(bool)
    d = d[ok & d["kappa"].notna()]
    if d.empty:
        print("  fig_collapse: keine Daten"); return
    nodes = sorted(d["nodes"].unique())
    stat = (d.groupby("nodes")["kappa"]
            .agg(["mean", "std", "size"]).rename(columns={"mean": "kbar"}))
    stat["spread_pct"] = [100 * float((d.loc[d["nodes"] == n, "kappa"]
                                       - stat.loc[n, "kbar"]).abs().max()
                                      / stat.loc[n, "kbar"]) for n in stat.index]

    fig, ax = _fig(1, 2, h=2.7)
    for n in nodes:
        c = node_color(n, nodes)
        kb, sp = stat.loc[n, "kbar"], stat.loc[n, "spread_pct"] / 100
        ax[0].axhspan(kb * (1 - sp), kb * (1 + sp), color=c, alpha=.12, lw=0)
        ax[0].axhline(kb, color=c, lw=.7, ls=":")
        for m, st in MODE.items():
            g = d[(d["nodes"] == n) & (d["mode"] == m)].sort_values("rx")
            if g.empty:
                continue
            ax[0].plot(g["rx"], g["kappa"], color=c, ls="none",
                       marker=st["marker"], ms=4,
                       mfc=c if st["fill"] else "none")
    ax[0].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel=r"$\kappa=\eta\,v_{\min}^{2}/z_{\mathrm{rel}}$",
              title=r"(a) Kollaps beider Modi, $\pm$Streuband")

    s = loglog_slope(stat.index, stat["kbar"])
    ax[1].errorbar(stat.index, stat["kbar"],
                   yerr=stat["kbar"] * stat["spread_pct"] / 100,
                   fmt="ko", ms=4, lw=1, capsize=2)
    xx = np.array([stat.index.min(), stat.index.max()], float)
    ax[1].plot(xx, stat["kbar"].iloc[0] * (xx / xx[0]) ** s, color="0.5",
               lw=.9, ls="--", label=fr"$\propto n^{{{s:.2f}}}$")
    ax[1].set(xscale="log", yscale="log", xlabel="$n$",
              ylabel=r"$\bar\kappa$", title=r"(b) Skalierung mit $n$")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=.3, which="both")
    h = [plt.Line2D([], [], color=node_color(n, nodes), marker="o", ls="none",
                    label=f"$n={n}$") for n in nodes]
    h += [plt.Line2D([], [], color="k", ls="none", marker=st["marker"],
                     mfc="k" if st["fill"] else "none", label=st["tex"])
          for st in MODE.values()]
    ax[0].legend(handles=h, ncol=2)
    _save(fig, "rx_collapse")

    for n, r in stat.iterrows():
        fact(f"kappa.mean.n{int(n)}", float(f"{r['kbar']:.4f}"))
        fact(f"kappa.spread_pct.n{int(n)}", round(r["spread_pct"], 1), "%")
        fact(f"kappa.points.n{int(n)}", int(r["size"]))
    fact("kappa.spread_pct_max", round(stat["spread_pct"].max(), 1), "%")
    fact("kappa.exponent_in_n", round(s, 3), "", "log-log-Fit kappa_bar(n)")
    fact("kappa.kappa1", float(f"{np.median(stat['kbar'] / stat.index):.2e}"),
         "", "kappa_bar/n, Vorfaktor in Gl. eta-empirisch")

    # divergenter const_x-Lauf: kappa aus eta_2
    bad = pq(df)
    bad = bad[~col(bad, "inner_conv_pq").fillna(False).astype(bool)]
    for _, r in bad.iterrows():
        k = r["eta_2"] * r["v_min"] ** 2 / r["z_rel"]
        if np.isfinite(k):
            fact(f"kappa.from_eta2.n{int(r['nodes'])}.{r['mode']}"
                 f".rho{r['rx']:g}", round(float(k), 3), "",
                 "Einordnung des divergenten Laufs")


# ----------------------------------------------------------------------
# Abbildung 3: fig:rx-outer
# ----------------------------------------------------------------------
def fig_outer(df, mode="const_z"):
    d = pv(df, "coupled", mode)
    if d.empty:
        print("  fig_outer: keine Daten"); return
    nodes = sorted(d["nodes"].unique())
    ratios = sorted(d["pv_ratio"].unique())
    fig, ax = _fig(1, 2, h=2.7)

    for n in nodes:
        c = node_color(n, nodes)
        for r in ratios:
            g = d[(d["nodes"] == n) & (d["pv_ratio"] == r)].sort_values("rx")
            if g.empty:
                continue
            mk = PV_MARKER.get(round(float(r), 2), "o")
            gk = g[g["cls"] == "conv"]
            ax[0].plot(gk["rx"], gk["outer_iter"], color=c, ls="-", lw=1.1,
                       marker=mk, ms=3.8)
            gs = g[g["cls"].isin(["Grenzzyklus", "Divergenz"])]
            ax[0].plot(gs["rx"], gs["outer_iter"].fillna(OUTER_CAP), "x",
                       color="crimson", ms=6, mew=1.3, zorder=5)
            gg = g.dropna(subset=["sens_error_median"])
            ax[1].plot(gg["rx"], gg["sens_error_median"], color=c, ls="-",
                       lw=1.1, marker=mk, ms=3.8)

    ax[0].axhline(OUTER_CAP, ls=":", c="k", lw=.8)
    ax[0].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel=r"\"au\ss ere Iterationen", title="(a) Iterationszahl")

    g0 = d[np.isclose(d["rx"], d["rx"].min())].dropna(
        subset=["sens_error_median"])
    if not g0.empty:
        r0 = float(d["rx"].min())
        s0 = float(g0["sens_error_median"].median())
        rr = np.geomspace(d["rx"].min(), d["rx"].max(), 200)
        ax[1].plot(rr, s0 * (1 + rr ** 2) / (1 + r0 ** 2), color="0.55",
                   lw=.9, zorder=0, label=r"$\propto 1+(R/X)^2$")
    ax[1].axhline(1.0, ls="-", c="0.3", lw=.7)
    ax[1].axhline(SENS_LIMIT, ls="--", c="k", lw=.8,
                  label=fr"$\varepsilon_{{\mathrm{{lin}}}}={SENS_LIMIT}$")
    ax[1].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel=r"$\varepsilon_{\mathrm{lin}}$ (Median)",
              title="(b) Linearisierungsfehler")
    ax[1].legend(loc="lower right")
    for a in ax:
        a.grid(alpha=.3, which="both")
    h = [plt.Line2D([], [], color=node_color(n, nodes), lw=1.5,
                    label=f"$n={n}$") for n in nodes]
    h += [plt.Line2D([], [], color="k", ls="none",
                     marker=PV_MARKER.get(round(float(r), 2), "o"),
                     label=fr"${r*100:.0f}\,\%$ PV") for r in ratios]
    ax[0].legend(handles=h, ncol=2, loc="upper left")
    _save(fig, "rx_outer_iterations")
    _facts_outer(df, d)


def _facts_outer(df, d):
    for r, g in d.groupby("pv_ratio"):
        gm = g.groupby("rx").median(numeric_only=True)
        if len(gm) < 3:
            continue
        fact(f"outer.q_growth.pv{r:.2f}",
             round(float(gm["q_max_final"].iloc[-1]
                         / gm["q_max_final"].iloc[0]), 1))
        fact(f"outer.sens_slope.pv{r:.2f}",
             round(loglog_slope(gm.index, gm["sens_error_median"]), 2), "",
             "log-log-Steigung eps_lin(rho)")
    fact("outer.lever_expected",
         round(float(np.sqrt(1 + d["rx"].max() ** 2)
                     / np.sqrt(1 + d["rx"].min() ** 2)), 1), "",
         "sqrt(1+rho^2) ueber den Sweep")
    fact("outer.sens_corr_q", round(spearman(d["q_max_final"],
                                             d["sens_error_median"]), 2))
    conv = d[d["cls"] == "conv"]
    fact("outer.inner_per_outer_conv_med",
         round(float(conv["inner_per_outer"].median()), 1))
    for cls, g in d[d["cls"] != "conv"].groupby("cls"):
        fact(f"outer.inner_per_outer.{cls}",
             f"{g['inner_per_outer'].min():.1f}-{g['inner_per_outer'].max():.1f}")
    lo = conv[conv["sens_error_median"] <= SENS_LIMIT]["outer_iter"]
    if len(lo):
        fact("outer.k_max_below_limit", int(lo.max()), "",
             f"max. k_out fuer eps_lin<={SENS_LIMIT}")
    for n, g in d.groupby("nodes"):
        fact(f"outer.ms_per_inner_slope.n{int(n)}",
             round(loglog_slope(g["rx"], g["ms_per_inner"]), 2))
    # Variantenvergleich und Modusgegenprobe
    for (m, v), g in pv(df).groupby(["mode", "variant"]):
        q = 100 * float((g["cls"] == "conv").mean())
        fact(f"outer.quote.{m}.{v}", round(q, 1), "%", f"{len(g)} Laeufe")
        gc = g[g["cls"] == "conv"]
        if len(gc):
            fact(f"outer.k_out_med.{m}.{v}", float(gc["outer_iter"].median()))
            fact(f"outer.k_in_med.{m}.{v}", float(gc["inner_total"].median()))
        rs = rho_star_table(g)["rho_star"].dropna()
        if len(rs):
            fact(f"outer.rho_star_med.{m}.{v}", round(float(rs.median()), 2))


def rho_star_table(d) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(["nodes", "n_pv", "mode", "variant"]):
        g = g.sort_values("rx")
        bad = g[g["cls"].isin(["Grenzzyklus", "Divergenz"])]
        rows.append(dict(zip(["nodes", "n_pv", "mode", "variant"], k),
                         rho_star=float(bad["rx"].min()) if len(bad) else np.nan,
                         kind=(bad["cls"].iloc[0] if len(bad) else "")))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Tabellen
# ----------------------------------------------------------------------
def tab_inner(df):
    d = pq(df)
    ok = col(d, "inner_conv_pq").fillna(False).astype(bool)
    rx = np.sort(d["rx"].unique())
    lo, hi = rx[0], rx[-1]
    rows = []
    for m in ("const_z", "const_x"):
        for n in sorted(d["nodes"].unique()):
            g = d[(d["nodes"] == n) & (d["mode"] == m)].set_index("rx")
            o = ok.reindex(g.index.map(
                lambda r: d[(d["nodes"] == n) & (d["mode"] == m)
                            & (d["rx"] == r)].index[0])).values
            g = g.assign(ok=o)
            def val(p, c):
                if p not in g.index or not bool(g.loc[p, "ok"]):
                    return np.nan
                return float(g.loc[p, c])
            rows.append({"$n$": int(n), "Modus": MODE[m]["tex"],
                         "$\\eta(0{,}1)$": val(lo, "eta_pub"),
                         "$\\eta(10)$": val(hi, "eta_pub"),
                         "$k(0{,}1)$": val(lo, "inner_iter_pq"),
                         "$k(10)$": val(hi, "inner_iter_pq")})
    t = pd.DataFrame(rows)
    save_table(t, "tab_rx_inner",
               r"Zahlenwerte zu Abbildung~\ref{fig:rx-inner} "
               r"($n_\mathrm{pv}=0$, Lastfaktor $2{,}0$, Abbruch bei "
               r"$\lVert\Delta\vect{V}\rVert_\infty<10^{-12}\,$p.u.).",
               "tab:rx-inner",
               fmt={"$\\eta(0{,}1)$": ".5f", "$\\eta(10)$": ".5f",
                    "$k(0{,}1)$": ".0f", "$k(10)$": ".0f"},
               note=r",,--`` bezeichnet fehlende Konvergenz innerhalb von "
                    r"$500$ Iterationen.")


def tab_outer_struct(df, mode="const_z", ratios=(0.10, 0.50)):
    d = pv(df, "coupled", mode).dropna(subset=["cond_xpp"])
    if d.empty:
        print("  tab_outer_struct: keine Daten"); return
    full, rows = [], []
    rho_lo = d["rx"].min()
    for (n, r), g in d.groupby(["nodes", "pv_ratio"]):
        ref = g[np.isclose(g["rx"], rho_lo)]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        rec = {"$n$": int(n), "pv": float(r), "$n_\\mathrm{pv}$": int(ref["n_pv"]),
               "$\\mathrm{cond}$": float(ref["cond_xpp"]),
               "$\\rho_{\\mathrm{J}}$": float(ref["rho_jacobi"]),
               "$\\bar{x}_{kk}$": float(ref["x_kk_mean"])}
        full.append(rec)
        if round(float(r), 2) in {round(x, 2) for x in ratios}:
            rows.append(rec)
        for c in ("cond_xpp", "rho_jacobi", "diag_dom_min"):
            v = g[c].dropna()
            if len(v) > 2 and v.min() > 0:
                fact(f"xpp.{c}_var_pct.n{int(n)}.pv{r:.2f}",
                     round(100 * (v.max() / v.min() - 1), 2), "%",
                     "Variation ueber rho")
    pd.DataFrame(full).to_csv(OUT / "tab_rx_outer_struct_full.csv", index=False)
    save_table(pd.DataFrame(rows), "tab_rx_outer_struct",
               r"Kennzahlen des PV--PV-Blocks $\vect{X}_{pp}$ im Modus "
               r"\texttt{const\_z} (Auswahl). Alle Strukturgr\"o\ss en sind "
               r"\"uber die $13$ $\rho$-St\"utzstellen konstant; "
               r"$\bar{x}_{kk}$ ist bei $\rho=0{,}1$ angegeben.",
               "tab:rx-outer-struct",
               fmt={"pv": ".2f", "$\\mathrm{cond}$": ".1f",
                    "$\\rho_{\\mathrm{J}}$": ".2f", "$\\bar{x}_{kk}$": ".5f"})
    var = [f["value"] for f in FACTS if f["key"].startswith("xpp.cond_xpp_var")]
    if var:
        fact("xpp.cond_var_pct_max", max(var), "%",
             "groesste Variation von cond(X_pp) ueber rho")
    for n, g in pd.DataFrame(full).groupby("$n$"):
        g = g.sort_values("$n_\\mathrm{pv}$")
        if len(g) > 1:
            fact(f"xpp.cond_growth_npv.n{int(n)}",
                 round(float(g["$\\mathrm{cond}$"].iloc[-1]
                             / g["$\\mathrm{cond}$"].iloc[0]), 2), "",
                 f"n_pv {int(g['$n_\\mathrm{pv}$'].iloc[0])}"
                 f"->{int(g['$n_\\mathrm{pv}$'].iloc[-1])}")
        gt = g[g["$\\rho_{\\mathrm{J}}$"] > 1]
        if len(gt):
            fact(f"xpp.first_pv_rhoJ_gt1.n{int(n)}",
                 float(gt["pv"].min()), "", "kleinster PV-Anteil mit rho_J>1")


def tab_outer(df, mode="const_z"):
    d = pv(df, "coupled", mode)
    if d.empty:
        return
    rs = rho_star_table(d).set_index(["nodes", "n_pv"])
    rows = []
    for (n, npv), g in d.groupby(["nodes", "n_pv"]):
        g = g.set_index("rx").sort_index()
        plat = g.loc[g.index <= PLATEAU_RHO, "outer_iter"].dropna()
        pv_plat = (int(plat.mode().iloc[0]) if len(plat) else np.nan)
        if len(plat) and plat.nunique() > 1:
            fact(f"outer.plateau_not_constant.n{int(n)}.npv{int(npv)}",
                 f"{plat.min():.0f}-{plat.max():.0f}", "",
                 "Plateau nicht konstant")
        rec = {"$n$": int(n), "$n_\\mathrm{pv}$": int(npv),
               "$\\le0{,}68$": pv_plat}
        for p in RHO_COLS:
            key = g.index[np.argmin(np.abs(g.index - p))]
            r = g.loc[key]
            rec[f"$\\rho={p:g}$".replace(".", "{,}")] = (
                float(r["outer_iter"]) if r["cls"] == "conv" else np.nan)
        star = rs.loc[(n, npv), "rho_star"]
        kind = rs.loc[(n, npv), "kind"]
        rec["$\\rho^\\ast$"] = float(star) if np.isfinite(star) else np.nan
        rec["Versagensart"] = kind if kind else "{--}"
        rows.append(rec)
        fact(f"outer.plateau.n{int(n)}.npv{int(npv)}", pv_plat)
        if np.isfinite(star):
            fact(f"outer.rho_star.{mode}.n{int(n)}.npv{int(npv)}",
                 round(float(star), 2), "", kind)
        top = g.loc[g.index.max()]
        if top["cls"] == "conv":
            fact(f"outer.k_at_rho_max.n{int(n)}.npv{int(npv)}",
                 float(top["outer_iter"]))
    t = pd.DataFrame(rows)
    save_table(t, "tab_rx_outer",
               r"\"Au\ss ere Iterationen $k_{\mathrm{out}}$ im Modus "
               r"\texttt{const\_z} mit gekoppelter Q-Korrektur. "
               r"$\rho^\ast$ ist das kleinste $R/X$ ohne Konvergenz; die "
               r"Versagensart ist in Tabelle~\ref{tab:rx-outer-fail} "
               r"aufgeschl\"usselt.",
               "tab:rx-outer",
               fmt={c: ".0f" for c in t.columns if c.startswith("$\\rho=")}
                   | {"$\\le0{,}68$": ".0f", "$\\rho^\\ast$": ".2f"},
               note=r",,--`` bedeutet, dass im gesamten Sweep bis $\rho=10$ "
                    r"kein Versagen auftritt.")


def tab_outer_fail(df, mode="const_z"):
    d = pv(df, "coupled", mode)
    d = d[d["at_cap"].fillna(False)].sort_values(["nodes", "n_pv", "rx"])
    if d.empty:
        print("  tab_outer_fail: keine Limitlaeufe"); return
    conv = col(d, "outer_conv").fillna(False).astype(bool)
    t = pd.DataFrame({
        "$n$": d["nodes"].astype(int),
        "$n_\\mathrm{pv}$": d["n_pv"].astype(int),
        "$\\rho$": d["rx"].astype(float),
        "$\\varepsilon_V$": d["v_err_final"].astype(float),
        "$\\max_k\\lvert Q_k\\rvert$": d["q_max_final"].astype(float),
        "$k_{\\mathrm{in}}/k_{\\mathrm{out}}$": d["inner_per_outer"].astype(float),
        "$\\varepsilon_{\\mathrm{lin}}$": d["sens_error_median"].astype(float),
        "Verhalten": [c + (r"\rlap{$^{\ast}$}" if k else "")
                      for c, k in zip(d["cls"], conv)]})
    save_table(t, "tab_rx_outer_fail",
               r"Alle L\"aufe im Modus \texttt{const\_z} (gekoppelt), die das "
               r"Iterationslimit von $60$ \"au\ss eren Schritten erreichen.",
               "tab:rx-outer-fail",
               fmt={"$\\rho$": ".2f",
                    "$\\varepsilon_V$": lambda v: f"{v:.2e}".replace("e-0", "e-"),
                    "$\\max_k\\lvert Q_k\\rvert$": ".0f",
                    "$k_{\\mathrm{in}}/k_{\\mathrm{out}}$": ".1f",
                    "$\\varepsilon_{\\mathrm{lin}}$": ".3f"},
               note=r"$^{\ast}$~Unterschreitet die Toleranz im letzten "
                    r"zugelassenen Schritt und gilt formal als konvergent.")
    for cls, g in d.groupby("cls"):
        fact(f"fail.count.{cls}", int(len(g)))
        fact(f"fail.eps_V.{cls}",
             f"{g['v_err_final'].min():.2e}-{g['v_err_final'].max():.2e}", "p.u.")
        fact(f"fail.q_max.{cls}",
             f"{g['q_max_final'].min():.3g}-{g['q_max_final'].max():.3g}", "p.u.")
        fact(f"fail.eps_lin.{cls}",
             f"{g['sens_error_median'].min():.3f}"
             f"-{g['sens_error_median'].max():.3f}")


def tab_predictor(df):
    d = pv(df)
    ok = d[d["cls"] == "conv"]
    if ok.empty:
        return
    names = {"sens_error_median": "Linearisierungsfehler (Median)",
             "sens_error_max": "Linearisierungsfehler (Max.)",
             "rx": r"$\rho=R/X$",
             "q_max_final": r"$\max_k\lvert Q_k\rvert$",
             "v_min": r"$v_{\min}$",
             "rho_jacobi": r"$\rho_{\mathrm{J}}$",
             "cond_xpp": r"$\mathrm{cond}(\vect{X}_{pp})$",
             "n_pv": r"$n_\mathrm{pv}$",
             "eta_pub": r"$\eta$",
             "nodes": "$n$"}
    rows = []
    for c, lbl in names.items():
        if c not in ok:
            continue
        so = spearman(ok[c], ok["outer_iter"])
        si = spearman(ok[c], ok["inner_total"])
        rows.append({"Gr\\\"o\\ss e": lbl,
                     "$\\rho_{\\mathrm{S}}$ zu $k_{\\mathrm{out}}$": so,
                     "$\\rho_{\\mathrm{S}}$ zu $k_{\\mathrm{in}}$": si})
        fact(f"pred.spearman_outer.{c}", round(so, 3))
        fact(f"pred.spearman_inner.{c}", round(si, 3))
    t = (pd.DataFrame(rows)
         .assign(a=lambda x: x.iloc[:, 1].abs())
         .sort_values("a", ascending=False).drop(columns="a"))
    fact("pred.n_runs", int(len(ok)), "", "konvergente PV-Laeufe, alle Modi")
    save_table(t, "tab_rx_predictor",
               r"Rangkorrelation (Spearman) zwischen Netz- bzw.\ Sweep-Gr\"o\ss en "
               r"und der Zahl \"au\ss erer und innerer Iterationen \"uber alle "
               rf"${len(ok)}$ konvergenten PV-L\"aufe beider Modi.",
               "tab:rx-predictor",
               fmt={c: ".3f" for c in t.columns if c.startswith("$\\rho_")})


def tab_nr(df):
    if "nr_conv" not in df.columns:
        print("  tab_nr: keine NR-Referenz"); return
    d = pv(df, "coupled")
    nr = d["nr_conv"].fillna(False).astype(bool)
    tpf = d["cls"] == "conv"
    d = d.assign(kat=np.select([nr & tpf, ~nr & ~tpf, nr & ~tpf],
                               ["beide", "keines", "nur NR"], "nur TPF"))
    order = ["beide", "keines", "nur NR", "nur TPF"]
    rows = []
    for k in order:
        g = d[d["kat"] == k]
        rows.append({"konvergiert": k, "F\\\"alle": int(len(g)),
                     "$v_{\\min}$ (Med.)": float(g["v_min"].median())
                     if len(g) else np.nan,
                     "$v_{\\min}^{\\mathrm{NR}}$": float(g["nr_v_min"].median())
                     if len(g) else np.nan,
                     "$\\eta_2$ (Med.)": float(g["eta_2"].median())
                     if len(g) else np.nan,
                     "$\\rho_{\\min}$": float(g["rx"].min()) if len(g) else np.nan,
                     "$\\rho_{\\max}$": float(g["rx"].max()) if len(g) else np.nan})
        fact(f"nr.faelle.{k}", int(len(g)))
        if len(g):
            fact(f"nr.vmin_med.{k}", round(float(g["v_min"].median()), 3), "p.u.")
            fact(f"nr.eta2_med.{k}", round(float(g["eta_2"].median()), 3))
            fact(f"nr.rho_min.{k}", round(float(g["rx"].min()), 2))
    fact("nr.n_runs", int(len(d)), "", "gekoppelte PV-Laeufe beider Modi")
    fact("nr.skipped_in_keines",
         int(((d["kat"] == "keines") & (d["skipped"] != "")).sum()), "",
         "nicht konstruierbare Faelle")
    save_table(pd.DataFrame(rows), "tab_rx_nr",
               r"Abgrenzung von Verfahrens- und L\"osbarkeitsgrenze \"uber die "
               r"Newton-Raphson-Referenz (\texttt{pandapower}), alle "
               rf"${len(d)}$ L\"aufe mit gekoppelter Q-Korrektur.",
               "tab:rx-nr",
               fmt={"$v_{\\min}$ (Med.)": ".3f",
                    "$v_{\\min}^{\\mathrm{NR}}$": ".3f",
                    "$\\eta_2$ (Med.)": ".3f",
                    "$\\rho_{\\min}$": ".2f", "$\\rho_{\\max}$": ".2f"})


# ----------------------------------------------------------------------
# Soll-Ist gegen die Textbehauptungen
# ----------------------------------------------------------------------
TEXT_CLAIMS = [
    ("inner.eta_rise_pct.const_z.n40", 2.2, 0.3),
    ("inner.eta_rise_pct.const_z.n120", 5.8, 0.3),
    ("inner.eta_rise_pct.const_z.n350", 21.7, 0.5),
    ("inner.vmin_lo.const_z.n350", 0.9725, 0.001),
    ("inner.vmin_hi.const_z.n350", 0.8774, 0.001),
    ("inner.eta_pred_hi.const_z.n350", 0.1343, 0.0005),
    ("inner.eta_hi.const_z.n350", 0.13306, 0.0005),
    ("inner.pred_err_pct.const_z.n350", 0.9, 0.2),
    ("inner.eta_growth.const_x.n40", 13.8, 0.2),
    ("inner.k_hi.const_x.n40", 16, 0),
    ("inner.vmin_factor.const_x.n40", 1.436, 0.01),
    ("inner.crossing.dev_pct", 0.2, 0.1),
    ("inner.eta_max_success", 0.62, 0.02),
    ("kappa.mean.n40", 0.0126, 0.0002),
    ("kappa.mean.n120", 0.0326, 0.0005),
    ("kappa.mean.n350", 0.1034, 0.001),
    ("kappa.spread_pct_max", 3.4, 0.2),
    ("kappa.exponent_in_n", 0.97, 0.02),
    ("xpp.cond_var_pct_max", 0.0, 0.05),
    ("xpp.cond_growth_npv.n40", 21.7, 0.3),
    ("xpp.cond_growth_npv.n120", 28.6, 0.3),
    ("xpp.cond_growth_npv.n350", 21.6, 0.3),
    ("outer.q_growth.pv0.10", 20.2, 0.5),
    ("outer.q_growth.pv0.25", 30.1, 0.5),
    ("outer.q_growth.pv0.50", 18.7, 0.5),
    ("outer.sens_corr_q", 0.67, 0.03),
    ("outer.inner_per_outer_conv_med", 5.5, 0.2),
    ("outer.k_max_below_limit", 7, 0),
    ("outer.quote.const_z.coupled", 88.9, 0.3),
    ("outer.quote.const_x.coupled", 74.4, 0.3),
    ("outer.quote.const_z.decoupled", 12.8, 0.3),
    ("outer.rho_star_med.const_z.coupled", 7.32, 0.05),
    ("outer.rho_star_med.const_x.coupled", 4.64, 0.05),
    ("outer.k_out_med.const_z.coupled", 3, 0),
    ("outer.k_in_med.const_z.coupled", 16.5, 0.5),
    ("pred.spearman_outer.sens_error_median", 0.899, 0.01),
    ("pred.spearman_outer.rx", 0.537, 0.01),
    ("pred.spearman_outer.rho_jacobi", -0.240, 0.01),
    ("pred.spearman_outer.nodes", -0.068, 0.01),
    ("pred.n_runs", 217, 0),
    ("nr.faelle.beide", 191, 0),
    ("nr.faelle.keines", 41, 0),
    ("nr.faelle.nur NR", 2, 0),
    ("nr.faelle.nur TPF", 0, 0),
    ("nr.n_runs", 234, 0),
    ("nr.skipped_in_keines", 15, 0),
]


def text_check():
    have = {f["key"]: f["value"] for f in FACTS}
    rows = []
    for key, soll, tol in TEXT_CLAIMS:
        ist = have.get(key, None)
        if ist is None:
            st, delta = "fehlt", np.nan
        else:
            try:
                delta = float(ist) - float(soll)
                st = "ok" if abs(delta) <= tol + 1e-12 else "ABWEICHUNG"
            except (TypeError, ValueError):
                delta, st = np.nan, "nicht numerisch"
        rows.append({"key": key, "text": soll, "daten": ist,
                     "delta": delta, "tol": tol, "status": st})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "rx_text_check.csv", index=False)
    bad = t[t["status"] != "ok"]
    print(f"  -> rx_text_check.csv  ({len(t)-len(bad)}/{len(t)} ok)")
    if len(bad):
        print(bad.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return t


def write_report(df, check):
    f = pd.DataFrame(FACTS)
    f.to_csv(OUT / "rx_facts.csv", index=False)

    def md(x):
        try:
            return x.to_markdown(index=False)
        except Exception:
            return "```\n" + x.to_string() + "\n```"

    lines = ["# R/X-Abschnitt: Abbildungen, Tabellen, Kennzahlen", "",
             f"Zeilen: {len(df)} | PQ: {len(pq(df))} | PV: {len(pv(df))} "
             f"| PV gekoppelt: {len(pv(df,'coupled'))}", "",
             "## Klassifikation der PV-Laeufe (alle Varianten)", "",
             md(pv(df)["cls"].value_counts().rename_axis("cls")
                .reset_index(name="faelle")), "",
             "## Soll-Ist gegen den Text", "", md(check), "",
             "## Kennzahlen", "", md(f), ""]
    for name, t in TABLES.items():
        lines += [f"## {name}", "", md(t), ""]
    (OUT / "rx_report.md").write_text("\n".join(lines), "utf-8")
    print(f"  -> rx_facts.csv ({len(f)}), rx_report.md")


def main():
    global OUT, OUTER_CAP, INNER_CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="rx_sweep.csv")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--pgf", action="store_true")
    ap.add_argument("--outer-cap", type=int, default=OUTER_CAP)
    ap.add_argument("--inner-cap", type=int, default=INNER_CAP)
    ap.add_argument("--no-inset", action="store_true")
    args = ap.parse_args()

    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    OUTER_CAP, INNER_CAP = args.outer_cap, args.inner_cap
    setup_mpl(args.pgf)

    df = load(args.csv)
    print(f"geladen: {args.csv} ({len(df)} Zeilen)")
    print("Abbildungen:")
    fig_inner(df, inset=not args.no_inset)
    fig_collapse(df)
    fig_outer(df)
    print("Tabellen:")
    tab_inner(df)
    tab_outer_struct(df)
    tab_outer(df)
    tab_outer_fail(df)
    tab_predictor(df)
    tab_nr(df)
    print("Pruefung:")
    write_report(df, text_check())


if __name__ == "__main__":
    main()
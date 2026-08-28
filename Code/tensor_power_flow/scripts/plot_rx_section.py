# tensor_power_flow/scripts/plot_rx_section.py
"""
Auswertung des R/X-Abschnitts - erzeugt Plots, LaTeX-Tabellen, Kennzahlen.

Liest ausschliesslich results/rx_sweep.csv (Header-basiert, tolerant gegen
fehlende Spalten) und schreibt nach OUT:

  Abbildungen : rx_vmin_check, rx_kappa_scaling, rx_xpp_structure_full,
                rx_q_sens, rx_rho_star, rx_variants, rx_predictor,
                rx_nr_categories, rx_timing, rx_outer_histories
  Tabellen    : tab_rx_inner, tab_rx_kappa, tab_rx_outer, tab_rx_variants,
                tab_rx_nr, tab_rx_timing            (je .csv und .tex)
  Kennzahlen  : rx_facts.csv, rx_report.md, rx_missing.csv, rx_missing_cmd.txt

Aufruf:
    python -m scripts.plot_rx_section --csv rx_sweep.csv --pgf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# ----------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------
OUT_DEFAULT = Path(r"C:\Users\sgrigorevski-admin\TensorPowerFlow"
                   r"\TensorPowerFlow-on-steroids\Bachelor_tensorflow\figures")
CSV_DEFAULT = "rx_sweep.csv"

OUTER_CAP = 60          # max_outer aus run_rx_sweep
V_TOL_SLOW = 1e-4       # "fast fertig" -> Klasse slow statt div
SENS_LIMIT = 0.6        # Schwelle im Praediktor-Plot
RX_BANDS = {"MS-Netz": (0.3, 1.5), "NS-Netz": (2.0, 8.0)}
MODE_STYLE = {"const_z": dict(ls="-", marker="o", fill=True),
              "const_x": dict(ls="--", marker="s", fill=False)}
VAR_STYLE = {"decoupled": dict(ls=":", marker="^"),
             "coupled": dict(ls="-", marker="o"),
             "exact": dict(ls="--", marker="D")}

FACTS: list[dict] = []
TABLES: dict[str, pd.DataFrame] = {}
OUT = OUT_DEFAULT


def setup_mpl(use_pgf: bool):
    if use_pgf:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[utf8]{inputenc}"
                            r"\usepackage[T1]{fontenc}",
        })
    global plt, VIR, PLASMA
    import matplotlib.pyplot as plt  # noqa: E402
    try:
        from matplotlib import colormaps
        VIR, PLASMA = colormaps["viridis"], colormaps["plasma"]
    except Exception:                                    # aeltere mpl
        from matplotlib.cm import viridis as VIR, plasma as PLASMA  # noqa
    globals()["USE_PGF"] = use_pgf


# ----------------------------------------------------------------------
# kleine Helfer
# ----------------------------------------------------------------------
def fact(key, value, unit="", note=""):
    FACTS.append({"key": key, "value": value, "unit": unit, "note": note})


def col(df, name, default=np.nan):
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index, dtype="float64")


def dec(s) -> np.ndarray:
    """';'-codierte Historie -> ndarray."""
    if not isinstance(s, str) or not s.strip():
        return np.array([])
    out = []
    for x in s.split(";"):
        try:
            out.append(float(x))
        except ValueError:
            pass
    return np.asarray(out)


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def spearman(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 4 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
        return np.nan
    r = s.rank()
    return float(np.corrcoef(r["a"], r["b"])[0, 1])


def _fig(nrows=1, ncols=1, w=5.91, h=2.6, **kw):
    return plt.subplots(nrows, ncols, figsize=(w, h * nrows),
                        constrained_layout=True, **kw)


def _save(fig, name):
    for ext in ("pgf", "pdf"):
        try:
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
        except Exception as e:
            print(f"  savefig {name}.{ext}: {type(e).__name__}: {e}")
    print(f"  -> {name}")
    if globals().get("USE_PGF"):
        plt.close(fig)
    else:
        plt.show()


def _bands(ax, show_label=False):
    for i, (lbl, (lo, hi)) in enumerate(RX_BANDS.items()):
        ax.axvspan(lo, hi, color="0.85", alpha=.45, zorder=-10,
                   label=lbl if show_label else None)


def node_color(n, nodes):
    nodes = list(nodes)
    return VIR(nodes.index(n) / max(len(nodes) - 1, 1))


def save_table(df, name, caption, label, index=True):
    df.to_csv(OUT / f"{name}.csv", index=index)
    body = df.to_latex(index=index, escape=False, na_rep="{--}",
                       float_format=lambda v: f"{v:.4g}")
    with open(OUT / f"{name}.tex", "w", encoding="utf-8") as f:
        f.write("% auto-generiert von plot_rx_section.py\n"
                "\\begin{table}[htbp]\n\\centering\n"
                f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
                f"{body}\\end{{table}}\n")
    TABLES[name] = df
    print(f"  -> {name}.csv/.tex")


# ----------------------------------------------------------------------
# Laden und Ableiten
# ----------------------------------------------------------------------
NUM = ["nodes", "pv_ratio", "n_pv", "rx", "load_factor", "r_ohm_km",
       "x_ohm_km", "z_abs_ohm_km", "z_rel", "line_factor", "nr_iter",
       "nr_v_min", "t_inner_s", "t_outer_s", "inner_iter_pq", "v_min",
       "eta_emp", "eta_1", "eta_2", "eta_inf", "eta_bound", "eta_fit",
       "eta_fit_r2", "eta_fit_n", "kappa_emp", "cond_xpp", "rho_jacobi",
       "diag_dom_min", "outer_iter", "inner_total", "v_err_final",
       "q_max_final", "sens_error_first", "sens_error_median",
       "sens_error_max"]
BOOL = ["outer_conv", "inner_conv_pq", "nr_conv", "decoupled", "exact_sens"]


def classify(df) -> pd.Series:
    conv = col(df, "outer_conv").fillna(False).astype(bool)
    at_cap = col(df, "outer_iter").fillna(0) >= OUTER_CAP
    verr = col(df, "v_err_final").astype(float)
    out = pd.Series("conv", index=df.index, dtype=object)
    out[conv & at_cap] = "marginal"
    out[~conv & (verr <= V_TOL_SLOW)] = "slow"
    out[~conv & ~(verr <= V_TOL_SLOW)] = "div"
    out[df["skipped"] != ""] = "skipped"
    if "error" in df.columns:
        out[df["error"].notna()] = "error"
    return out


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

    # z_rel notfalls rekonstruieren
    if "z_rel" not in df or df["z_rel"].isna().all():
        base = df.loc[df["mode"] == "const_z", "z_abs_ohm_km"].median()
        df["z_rel"] = df["z_abs_ohm_km"] / base

    # eta: Fit bevorzugen, wenn er belastbar ist
    fit_ok = (col(df, "eta_fit_r2") >= 0.9) & \
             col(df, "inner_conv_pq").fillna(False).astype(bool)
    df["eta_use"] = np.where(fit_ok & col(df, "eta_fit").notna(),
                             col(df, "eta_fit"), col(df, "eta_emp"))
    df["eta_src"] = np.where(fit_ok & col(df, "eta_fit").notna(),
                             "fit", "emp")
    df["kappa"] = df["eta_use"] * df["v_min"] ** 2 / df["z_rel"]
    df["cls"] = classify(df)
    df["inner_per_outer"] = col(df, "inner_total") / col(df, "outer_iter")
    df["ms_per_inner"] = 1e3 * col(df, "t_outer_s") / col(df, "inner_total")
    df["x_pu_rel"] = col(df, "x_ohm_km") / col(df, "x_ohm_km").max()
    return df


def pq(df):
    d = df[df["pv_ratio"] == 0].copy()
    return d.drop_duplicates(subset=["nodes", "mode", "rx"])


def pv(df, variant=None):
    d = df[df["pv_ratio"] > 0].copy()
    return d if variant is None else d[d["variant"] == variant]


# ----------------------------------------------------------------------
# 1) Warum v_min - Verifikation der Schranke
# ----------------------------------------------------------------------
def fig_vmin_check(df):
    d = pq(df)
    d = d[col(d, "inner_conv_pq").fillna(False).astype(bool)
          & d["eta_use"].notna() & d["v_min"].notna()]
    if d.empty:
        print("  fig_vmin_check: keine Daten"); return
    nodes = sorted(d["nodes"].unique())

    rows = []
    for (n, m), g in d.groupby(["nodes", "mode"]):
        g = g.sort_values("rx")
        e0, v0, z0 = g["eta_use"].iloc[0], g["v_min"].iloc[0], g["z_rel"].iloc[0]
        pred = e0 * (g["z_rel"] / z0) * (v0 / g["v_min"]) ** 2
        rel = (g["eta_use"] / pred - 1).abs()
        rows.append(dict(nodes=int(n), mode=m, rx_ref=g["rx"].iloc[0],
                         v_min_lo=v0, v_min_hi=g["v_min"].iloc[-1],
                         eta_lo=e0, eta_hi=g["eta_use"].iloc[-1],
                         eta_growth=g["eta_use"].iloc[-1] / e0,
                         pred_hi=float(pred.iloc[-1]),
                         err_max_pct=100 * float(rel.max()),
                         err_med_pct=100 * float(rel.median())))
        g = g.assign(eta_pred=pred)
        d.loc[g.index, "eta_pred"] = pred
    t = pd.DataFrame(rows).sort_values(["mode", "nodes"])
    save_table(t.set_index(["mode", "nodes"]), "tab_rx_vmin",
               r"Verifikation der Schranke $\eta\propto z_{\mathrm{rel}}/"
               r"v_{\min}^{2}$ je Netzgr\"o\ss e und Sweep-Modus.",
               "tab:rx-vmin")
    for r in rows:
        fact(f"vmin.err_max_pct.{r['mode']}.n{r['nodes']}",
             round(r["err_max_pct"], 2), "%", "Vorhersage vs. Messung")
        fact(f"vmin.range.{r['mode']}.n{r['nodes']}",
             f"{r['v_min_lo']:.3f}->{r['v_min_hi']:.3f}", "p.u.", "v_min")

    fig, ax = _fig(1, 2, w=5.91, h=2.7)
    for m, st in MODE_STYLE.items():
        for n in nodes:
            g = d[(d["nodes"] == n) & (d["mode"] == m)].sort_values("rx")
            if g.empty:
                continue
            c = node_color(n, nodes)
            ax[0].plot(g["rx"], g["v_min"], ls=st["ls"], marker=st["marker"],
                       ms=4, lw=1.2, color=c,
                       mfc=c if st["fill"] else "none")
            ax[1].plot(g["eta_pred"], g["eta_use"], ls="none",
                       marker=st["marker"], ms=4.5, color=c,
                       mfc=c if st["fill"] else "none")
    _bands(ax[0], show_label=True)
    ax[0].set(xscale="log", xlabel="$R/X$", ylabel=r"$v_{\min}$ [p.u.]",
              title="(a) Arbeitspunkt")
    lim = [d["eta_use"].min() * .8, d["eta_use"].max() * 1.25]
    ax[1].plot(lim, lim, color="0.4", lw=.9)
    for f in (1.05, 0.95):
        ax[1].plot(lim, [f * lim[0], f * lim[1]], color="0.7", lw=.7, ls=":")
    ax[1].set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              xlabel=r"$\eta_{\mathrm{pred}}=\eta_{\mathrm{ref}}\,"
                     r"z_{\mathrm{rel}}/v_{\min}^{2}$",
              ylabel=r"$\eta_{\mathrm{gemessen}}$",
              title=r"(b) Parit\"at, $\pm5\,\%$")
    for a in ax:
        a.grid(alpha=.3, which="both")
    h = [plt.Line2D([], [], color=node_color(n, nodes), lw=1.5, label=f"$n={n}$")
         for n in nodes]
    h += [plt.Line2D([], [], color="k", ls="-", marker="o", label=r"const\_z"),
          plt.Line2D([], [], color="k", ls="--", marker="s", mfc="none",
                     label=r"const\_x")]
    ax[1].legend(handles=h, fontsize=6, loc="upper left")
    ax[0].legend(fontsize=6, loc="lower left")
    _save(fig, "rx_vmin_check")


# ----------------------------------------------------------------------
# 2) kappa: Streuung und Skalierung mit n
# ----------------------------------------------------------------------
def fig_kappa_scaling(df):
    d = pq(df)
    d = d[col(d, "inner_conv_pq").fillna(False).astype(bool) & d["kappa"].notna()]
    if d.empty:
        print("  fig_kappa_scaling: keine Daten"); return
    rows = []
    for n, g in d.groupby("nodes"):
        k = g["kappa"].values
        cx = g[g["mode"] == "const_x"].sort_values("rx")
        rows.append(dict(nodes=int(n), punkte=len(k), kappa_mean=k.mean(),
                         spread_pct=100 * np.max(np.abs(k - k.mean())) / k.mean(),
                         cv_pct=100 * k.std(ddof=1) / k.mean(),
                         eta_growth_const_x=(cx["eta_use"].iloc[-1] /
                                             cx["eta_use"].iloc[0])
                         if len(cx) > 1 else np.nan))
    t = pd.DataFrame(rows).sort_values("nodes")
    save_table(t.set_index("nodes"), "tab_rx_kappa",
               r"Streuma\ss{} von $\kappa$ je Netzgr\"o\ss e "
               r"(beide Sweep-Modi, alle konvergenten $\rho$).",
               "tab:rx-collapse")
    slope = loglog_slope(t["nodes"], t["kappa_mean"])
    fact("kappa.exponent_in_n", round(slope, 3), "", "log-log-Fit kappa_mean(n)")
    fact("kappa.spread_max_pct", round(t["spread_pct"].max(), 2), "%",
         "max. Abweichung ueber alle n")
    for _, r in t.iterrows():
        fact(f"kappa.mean.n{int(r['nodes'])}", round(r["kappa_mean"], 5))

    fig, ax = _fig(1, 2, w=5.91, h=2.6)
    nodes = sorted(d["nodes"].unique())
    for m, st in MODE_STYLE.items():
        for n in nodes:
            g = d[(d["nodes"] == n) & (d["mode"] == m)].sort_values("rx")
            if g.empty:
                continue
            c = node_color(n, nodes)
            ax[0].plot(g["rx"], g["kappa"] / t.set_index("nodes")
                       .loc[n, "kappa_mean"], ls=st["ls"], marker=st["marker"],
                       ms=4, lw=1.1, color=c, mfc=c if st["fill"] else "none")
    ax[0].axhline(1, color="0.4", lw=.8)
    _bands(ax[0])
    ax[0].set(xscale="log", xlabel="$R/X$",
              ylabel=r"$\kappa/\bar\kappa(n)$", title="(a) Restschwankung")
    ax[1].plot(t["nodes"], t["kappa_mean"], "ko-", ms=4)
    xx = np.array([t["nodes"].min(), t["nodes"].max()], float)
    ax[1].plot(xx, t["kappa_mean"].iloc[0] * (xx / xx[0]) ** slope,
               color="0.5", lw=.9, ls="--",
               label=fr"$\propto n^{{{slope:.2f}}}$")
    ax[1].set(xscale="log", yscale="log", xlabel="$n$",
              ylabel=r"$\bar\kappa$", title="(b) Skalierung mit $n$")
    ax[1].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=.3, which="both")
    _save(fig, "rx_kappa_scaling")


# ----------------------------------------------------------------------
# 3) Struktur von X_pp: rho vs. n_pv
# ----------------------------------------------------------------------
def fig_xpp_structure_full(df):
    d = pv(df, "coupled").dropna(subset=["cond_xpp"])
    if d.empty:
        print("  fig_xpp_structure_full: keine Daten"); return
    metrics = [("cond_xpp", r"$\mathrm{cond}(X_{pp})$", "log"),
               ("rho_jacobi", r"$\rho_{\mathrm{Jacobi}}$", "linear"),
               ("diag_dom_min", r"$\min_k d_k/\mathrm{off}_k$", "log")]
    nodes = sorted(d["nodes"].unique())
    fig, ax = _fig(2, 3, w=5.91, h=2.3)
    for j, (mcol, lbl, sc) in enumerate(metrics):
        a = ax[0, j]
        for n in nodes:
            for m, st in MODE_STYLE.items():
                for r in sorted(d["pv_ratio"].unique()):
                    g = d[(d["nodes"] == n) & (d["mode"] == m) &
                          (d["pv_ratio"] == r)].sort_values("rx")
                    if g.empty:
                        continue
                    c = VIR(list(sorted(d["pv_ratio"].unique())).index(r) /
                            max(d["pv_ratio"].nunique() - 1, 1))
                    a.plot(g["rx"], g[mcol], ls=st["ls"], marker=st["marker"],
                           ms=3.2, lw=1.0, color=c,
                           mfc=c if st["fill"] else "none")
        _bands(a)
        a.set(xscale="log", yscale=sc, xlabel="$R/X$", ylabel=lbl)
        if mcol != "cond_xpp":
            a.axhline(1.0, ls=":", c="k", lw=.8)
        b = ax[1, j]
        ref = d[np.isclose(d["rx"], d["rx"].min())]
        for n in nodes:
            g = ref[ref["nodes"] == n].sort_values("n_pv")
            if g.empty:
                continue
            b.plot(g["n_pv"], g[mcol], "o-", ms=4, lw=1.2,
                   color=node_color(n, nodes), label=f"$n={n}$")
        b.set(xscale="log", yscale=sc, xlabel=r"$n_{\mathrm{PV}}$", ylabel=lbl)
        if mcol != "cond_xpp":
            b.axhline(1.0, ls=":", c="k", lw=.8)
    for a in ax.ravel():
        a.grid(alpha=.3, which="both")
    ax[0, 0].set_title(r"(a) \"uber $R/X$", fontsize=8)
    ax[1, 0].set_title(r"(b) \"uber $n_{\mathrm{PV}}$ bei $\rho_{\min}$",
                       fontsize=8)
    ax[1, 2].legend(fontsize=6)
    _save(fig, "rx_xpp_structure_full")

    for (n, r, m), g in d.groupby(["nodes", "pv_ratio", "mode"]):
        if len(g) < 3:
            continue
        v = g["cond_xpp"]
        fact(f"xpp.cond_var_pct.n{int(n)}.pv{r:.2f}.{m}",
             round(100 * (v.max() / v.min() - 1), 1), "%",
             "Variation cond(X_pp) ueber rho")
    ref = d[np.isclose(d["rx"], d["rx"].min())]
    for n, g in ref.groupby("nodes"):
        g = g.sort_values("n_pv")
        if len(g) > 1:
            fact(f"xpp.cond_growth_npv.n{int(n)}",
                 round(float(g["cond_xpp"].iloc[-1] / g["cond_xpp"].iloc[0]), 2),
                 "", f"n_pv {int(g['n_pv'].iloc[0])}->{int(g['n_pv'].iloc[-1])}")


# ----------------------------------------------------------------------
# 4) Q-Bedarf und Linearisierungsfehler
# ----------------------------------------------------------------------
def fig_q_sens(df):
    d = pv(df, "coupled").dropna(subset=["q_max_final"])
    if d.empty:
        print("  fig_q_sens: keine Daten"); return
    ratios = sorted(d["pv_ratio"].unique())
    fig, ax = _fig(1, 3, w=5.91, h=2.6)

    for m, st in MODE_STYLE.items():
        for i, r in enumerate(ratios):
            g = d[(d["mode"] == m) & (d["pv_ratio"] == r)].groupby("rx").median(
                numeric_only=True).reset_index()
            if g.empty:
                continue
            c = VIR(i / max(len(ratios) - 1, 1))
            kw = dict(ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.1,
                      color=c, mfc=c if st["fill"] else "none")
            ax[0].plot(g["rx"], g["q_max_final"], **kw)
            ax[1].plot(g["rx"], g["sens_error_median"], **kw)
        gm = d[d["mode"] == m].dropna(subset=["sens_error_median"])
        if not gm.empty:
            ax[2].scatter(gm["q_max_final"], gm["sens_error_median"], s=12,
                          c=np.log10(gm["rx"]), cmap="plasma",
                          marker=st["marker"],
                          facecolors=None if st["fill"] else "none")
    rho = np.geomspace(d["rx"].min(), d["rx"].max(), 60)
    g0 = d[np.isclose(d["rx"], d["rx"].min())]
    if not g0.empty:
        q0 = g0["q_max_final"].median()
        ax[0].plot(rho, q0 * np.sqrt(1 + rho ** 2) /
                   np.sqrt(1 + d["rx"].min() ** 2), color="0.45", lw=.9,
                   label=r"$\propto\sqrt{1+\rho^2}$")
        s0 = g0["sens_error_median"].median()
        ax[1].plot(rho, s0 * (1 + rho ** 2) / (1 + d["rx"].min() ** 2),
                   color="0.45", lw=.9, label=r"$\propto 1+\rho^2$")
        ax[1].plot(rho, s0 * rho / d["rx"].min(), color="0.7", lw=.9, ls="--",
                   label=r"$\propto\rho$")
    for a in ax[:2]:
        _bands(a)
    ax[0].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel=r"$\max_k|Q_k|$ [p.u.]", title="(a) Q-Bedarf")
    ax[1].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel="rel. Sens.-Fehler (Median)",
              title="(b) Linearisierung")
    ax[1].axhline(SENS_LIMIT, ls=":", c="k", lw=.8)
    ax[2].set(xscale="log", yscale="log", xlabel=r"$\max_k|Q_k|$ [p.u.]",
              ylabel="rel. Sens.-Fehler", title=r"(c) Kausalit\"at")
    for a in ax:
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=6)
    _save(fig, "rx_q_sens")

    for (m, r), g in d.groupby(["mode", "pv_ratio"]):
        g = g.groupby("rx").median(numeric_only=True)
        if len(g) < 3:
            continue
        fact(f"q.growth.{m}.pv{r:.2f}",
             round(float(g["q_max_final"].iloc[-1] /
                         g["q_max_final"].iloc[0]), 2), "",
             "max|Q| von rho_min bis rho_max")
        fact(f"sens.slope.{m}.pv{r:.2f}",
             round(loglog_slope(g.index, g["sens_error_median"]), 2), "",
             "log-log-Steigung Sens.-Fehler(rho)")
    fact("q.expected_growth_const_z",
         round(float(np.sqrt(1 + d['rx'].max() ** 2) /
                     np.sqrt(1 + d['rx'].min() ** 2)), 2), "",
         "Erwartung sqrt(1+rho^2)")
    fact("sens.corr_with_q", round(spearman(d["q_max_final"],
                                            d["sens_error_median"]), 3),
         "", "Spearman")


# ----------------------------------------------------------------------
# 5) rho* je (n, n_pv, Modus, Variante)
# ----------------------------------------------------------------------
def rho_star(d) -> pd.DataFrame:
    rows = []
    keys = ["nodes", "n_pv", "mode", "variant"]
    for k, g in d.groupby(keys):
        g = g.sort_values("rx")
        bad = g[~g["cls"].isin(["conv"])]
        ok = g[g["cls"] == "conv"]
        rows.append(dict(zip(keys, k),
                         rho_star=float(bad["rx"].min()) if len(bad) else np.nan,
                         rho_max_ok=float(ok["rx"].max()) if len(ok) else np.nan,
                         n_conv=int(len(ok)), n_total=int(len(g)),
                         outer_med=float(ok["outer_iter"].median())
                         if len(ok) else np.nan,
                         outer_max=float(ok["outer_iter"].max())
                         if len(ok) else np.nan))
    return pd.DataFrame(rows).sort_values(["variant", "mode", "nodes", "n_pv"])


def fig_rho_star(df):
    d = pv(df)
    if d.empty:
        print("  fig_rho_star: keine Daten"); return
    t = rho_star(d)
    save_table(t.set_index(["variant", "mode", "nodes", "n_pv"]),
               "tab_rx_outer",
               r"Konvergenzgrenze $\rho^\ast$ der \"au\ss eren Schleife "
               r"(kleinstes $R/X$ ohne Konvergenz) je Konfiguration.",
               "tab:rx-outer")

    nodes = sorted(d["nodes"].unique())
    fig, ax = _fig(1, 2, w=5.91, h=2.6)
    for var, vst in VAR_STYLE.items():
        sub = t[t["variant"] == var]
        for n in nodes:
            for m, st in MODE_STYLE.items():
                g = sub[(sub["nodes"] == n) & (sub["mode"] == m)
                        ].sort_values("n_pv")
                if g.empty:
                    continue
                c = node_color(n, nodes)
                y = g["rho_star"].fillna(d["rx"].max() * 1.6)
                ax[0].plot(g["n_pv"], y, ls=vst["ls"], marker=vst["marker"],
                           ms=4, lw=1.1, color=c,
                           mfc=c if st["fill"] else "none")
    ax[0].axhline(d["rx"].max(), ls=":", c="k", lw=.8)
    ax[0].text(ax[0].get_xlim()[0], d["rx"].max() * 1.1,
               r"kein $\rho^\ast$ im Sweep", fontsize=6)
    ax[0].set(xscale="log", yscale="log", xlabel=r"$n_{\mathrm{PV}}$",
              ylabel=r"$\rho^\ast$", title=r"(a) \"uber $n_{\mathrm{PV}}$")
    for var, vst in VAR_STYLE.items():
        g = t[t["variant"] == var].groupby("nodes")["rho_star"].median()
        if g.dropna().empty:
            continue
        ax[1].plot(g.index, g.values, ls=vst["ls"], marker=vst["marker"],
                   ms=4, lw=1.2, color="k" if var == "coupled" else None,
                   label=var)
    ax[1].set(xscale="log", yscale="log", xlabel="$n$",
              ylabel=r"$\mathrm{med}(\rho^\ast)$",
              title="(b) \\\"uber $n$, je Variante")
    ax[1].legend(fontsize=6)
    for a in ax:
        a.grid(alpha=.3, which="both")
    h = [plt.Line2D([], [], color=node_color(n, nodes), lw=1.5,
                    label=f"$n={n}$") for n in nodes]
    h += [plt.Line2D([], [], color="k", ls=v["ls"], marker=v["marker"],
                     label=k) for k, v in VAR_STYLE.items()]
    ax[0].legend(handles=h, fontsize=5.5, ncol=2)
    _save(fig, "rx_rho_star")

    for _, r in t.iterrows():
        if np.isfinite(r["rho_star"]):
            fact(f"rhostar.{r['variant']}.{r['mode']}.n{int(r['nodes'])}"
                 f".npv{int(r['n_pv'])}", round(r["rho_star"], 3))
    fact("rhostar.corr_npv", round(spearman(t["n_pv"], t["rho_star"]), 3),
         "", "Spearman rho* vs n_pv")
    fact("rhostar.corr_nodes", round(spearman(t["nodes"], t["rho_star"]), 3),
         "", "Spearman rho* vs n")


# ----------------------------------------------------------------------
# 6) Variantenvergleich
# ----------------------------------------------------------------------
def fig_variants(df):
    d = pv(df)
    if d["variant"].nunique() < 2:
        print("  fig_variants: nur eine Variante vorhanden"); return
    piv = (d.assign(ok=(d["cls"] == "conv").astype(float))
           .pivot_table(index=["mode", "variant"], columns="rx",
                        values="ok", aggfunc="mean"))
    rows = []
    for (mode, var), g in d.groupby(["mode", "variant"]):
        rs = rho_star(g)
        conv_mask = g["cls"] == "conv"
        rows.append({
            "mode": mode, "variant": var,
            "konv_quote": float(conv_mask.mean()),
            "outer_med": float(g.loc[conv_mask, "outer_iter"].median()),
            "inner_med": float(g.loc[conv_mask, "inner_total"].median()),
            "rho_star_med": float(rs["rho_star"].median()),
            "n": int(len(g))})
    summ = pd.DataFrame(rows).set_index(["mode", "variant"])
    save_table(summ, "tab_rx_variants",
               r"Vergleich der Q-Korrekturvarianten \"uber den gesamten "
               r"$R/X$-Sweep.", "tab:rx-variants")

    n_ref = int(d["nodes"].value_counts().idxmax())
    dr = d[d["nodes"] == n_ref]
    ratios = sorted(dr["pv_ratio"].unique())
    r_ref = ratios[len(ratios) // 2]
    fig, ax = _fig(1, 2, w=5.91, h=2.6)
    for var, vst in VAR_STYLE.items():
        for m, st in MODE_STYLE.items():
            g = dr[(dr["variant"] == var) & (dr["mode"] == m) &
                   (dr["pv_ratio"] == r_ref)].sort_values("rx")
            if g.empty:
                continue
            ax[0].plot(g["rx"], g["outer_iter"], ls=vst["ls"],
                       marker=vst["marker"], ms=4, lw=1.2,
                       label=f"{var}, {m}".replace("_", r"\_"))
            bad = g[g["cls"] != "conv"]
            ax[0].plot(bad["rx"], bad["outer_iter"], "x", color="crimson",
                       ms=7, mew=1.4, zorder=5)
    ax[0].axhline(OUTER_CAP, ls="--", c="k", lw=.8)
    _bands(ax[0])
    ax[0].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel=r"\"au\ss ere Iterationen",
              title=fr"(a) $n={n_ref}$, PV-Anteil {r_ref*100:.0f}\%")
    ax[0].legend(fontsize=5.5)
    for i, (var, vst) in enumerate(VAR_STYLE.items()):
        for m, st in MODE_STYLE.items():
            if (m, var) not in piv.index:
                continue
            s = piv.loc[(m, var)]
            ax[1].plot(s.index, 100 * s.values, ls=st["ls"],
                       marker=vst["marker"], ms=4, lw=1.2,
                       label=f"{var}, {m}".replace("_", r"\_"))
    _bands(ax[1])
    ax[1].set(xscale="log", xlabel="$R/X$", ylim=(-5, 105),
              ylabel=r"Konvergenzquote [\%]",
              title="(b) alle Konfigurationen")
    ax[1].legend(fontsize=5.5)
    for a in ax:
        a.grid(alpha=.3, which="both")
    _save(fig, "rx_variants")

    for (m, var), r in summ.iterrows():
        fact(f"variant.quote.{m}.{var}", round(100 * r["konv_quote"], 1), "%")
        fact(f"variant.rhostar_med.{m}.{var}", round(r["rho_star_med"], 3))
        fact(f"variant.outer_med.{m}.{var}", r["outer_med"])


# ----------------------------------------------------------------------
# 7) Praediktor der aeusseren Iterationszahl
# ----------------------------------------------------------------------
def fig_predictor(df):
    d = pv(df).dropna(subset=["outer_iter"])
    if d.empty:
        print("  fig_predictor: keine Daten"); return
    cand = ["rx", "sens_error_median", "sens_error_max", "cond_xpp",
            "rho_jacobi", "eta_use", "v_min", "n_pv", "q_max_final",
            "z_rel", "nodes"]
    ok = d[d["cls"] == "conv"]
    corr = pd.DataFrame(
        [{"groesse": c,
          "spearman_outer": spearman(ok[c], ok["outer_iter"]),
          "spearman_inner": spearman(ok[c], ok["inner_total"]),
          "n": int(ok[c].notna().sum())}
         for c in cand if c in d.columns]).set_index("groesse")
    corr["abs"] = corr["spearman_outer"].abs()
    corr = corr.sort_values("abs", ascending=False).drop(columns="abs")
    save_table(corr, "tab_rx_predictor",
               r"Rangkorrelation (Spearman) zwischen Netz-/Sweep-Gr\"o\ss en "
               r"und der Zahl \"au\ss erer bzw. innerer Iterationen "
               r"(nur konvergente L\"aufe).", "tab:rx-predictor")
    for c, r in corr.iterrows():
        fact(f"pred.spearman_outer.{c}", round(r["spearman_outer"], 3))

    fig, ax = _fig(1, 3, w=5.91, h=2.6)
    ratios = sorted(d["pv_ratio"].unique())
    for m, st in MODE_STYLE.items():
        for i, r in enumerate(ratios):
            g = d[(d["mode"] == m) & (d["pv_ratio"] == r)].sort_values("rx")
            if g.empty:
                continue
            c = VIR(i / max(len(ratios) - 1, 1))
            gg = g[g["cls"] == "conv"]
            ax[0].plot(gg["rx"], gg["outer_iter"], ls=st["ls"],
                       marker=st["marker"], ms=3.5, lw=1.0, color=c,
                       mfc=c if st["fill"] else "none")
            bad = g[g["cls"] != "conv"]
            ax[0].plot(bad["rx"], bad["outer_iter"].fillna(OUTER_CAP), "x",
                       color="crimson", ms=6, mew=1.2)
    _bands(ax[0])
    ax[0].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel=r"\"au\ss ere Iterationen", title="(a) $R/X$ als Achse")
    sc = None
    for m, st in MODE_STYLE.items():
        g = d[(d["mode"] == m)].dropna(subset=["sens_error_median"])
        gg = g[g["cls"] == "conv"]
        sc = ax[1].scatter(gg["sens_error_median"], gg["outer_iter"], s=14,
                           c=np.log10(gg["rx"]), cmap="plasma",
                           marker=st["marker"],
                           edgecolors="k", linewidths=.2)
        bad = g[g["cls"] != "conv"]
        ax[1].scatter(bad["sens_error_median"],
                      bad["outer_iter"].fillna(OUTER_CAP), s=22,
                      marker="x", color="crimson")
    ax[1].axvline(SENS_LIMIT, ls="--", c="k", lw=.8)
    ax[1].set(xscale="log", yscale="log",
              xlabel="rel. Sens.-Fehler (Median)",
              ylabel=r"\"au\ss ere Iterationen",
              title="(b) Fehler als Achse")
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax[1], fraction=.05)
        cb.set_label(r"$\log_{10}(R/X)$", fontsize=6)
        cb.ax.tick_params(labelsize=5)
    corr.head(6)["spearman_outer"].abs().plot.barh(ax=ax[2], color="0.4")
    ax[2].set(xlabel=r"$|\rho_{\mathrm{Spearman}}|$ zu Iterationen", ylabel="",
              title="(c) Rangkorrelation")
    ax[2].tick_params(labelsize=6)
    for a in ax[:2]:
        a.grid(alpha=.3, which="both")
    _save(fig, "rx_predictor")


# ----------------------------------------------------------------------
# 8) Verfahrens- vs. Physikgrenze (NR-Referenz)
# ----------------------------------------------------------------------
def fig_nr_categories(df):
    if "nr_conv" not in df.columns:
        print("  fig_nr_categories: keine NR-Referenz in der CSV"); return
    d = pv(df, "coupled").copy()
    if d.empty:
        d = pq(df).copy()
    nr = d["nr_conv"].fillna(False).astype(bool)
    tpf = (d["cls"] == "conv")
    d["kat"] = np.select([nr & tpf, nr & ~tpf, ~nr & tpf],
                         ["beide", "nur NR", "nur TPF"], "keiner")
    piv = (d.pivot_table(index=["mode", "rx"], columns="kat", values="nodes",
                         aggfunc="count").fillna(0).astype(int))
    summ = (d.groupby("kat").agg(faelle=("nodes", "size"),
                                 v_min_med=("v_min", "median"),
                                 nr_vmin_med=("nr_v_min", "median"),
                                 eta2_med=("eta_2", "median"),
                                 rx_min=("rx", "min"), rx_max=("rx", "max")))
    save_table(summ, "tab_rx_nr",
               r"Abgrenzung Verfahrens- gegen Physikgrenze \"uber die "
               r"Newton-Raphson-Referenz.", "tab:rx-nr")
    for k, r in summ.iterrows():
        fact(f"nr.faelle.{k}", int(r["faelle"]))
        fact(f"nr.vmin_med.{k}", round(float(r["v_min_med"]), 3), "p.u.")

    modes = [m for m in ("const_z", "const_x") if (d["mode"] == m).any()]
    fig, ax = _fig(1, len(modes), w=5.91, h=2.6, sharey=True)
    ax = np.atleast_1d(ax)
    order = ["beide", "nur TPF", "nur NR", "keiner"]
    colors = {"beide": "0.75", "nur TPF": "#2c7fb8",
              "nur NR": "#fdae61", "keiner": "crimson"}
    for j, m in enumerate(modes):
        p = piv.loc[m] if m in piv.index.get_level_values(0) else None
        if p is None:
            continue
        x = np.arange(len(p.index))
        bottom = np.zeros(len(p.index))
        for k in order:
            if k not in p.columns:
                continue
            ax[j].bar(x, p[k].values, bottom=bottom, width=.8,
                      color=colors[k], label=k)
            bottom += p[k].values
        ax[j].set_xticks(x, [f"{v:g}" for v in p.index], rotation=90,
                         fontsize=5)
        ax[j].set(xlabel="$R/X$", title=m.replace("_", r"\_"))
    ax[0].set_ylabel(r"F\"alle")
    ax[-1].legend(fontsize=6)
    _save(fig, "rx_nr_categories")


# ----------------------------------------------------------------------
# 9) Rechenzeit
# ----------------------------------------------------------------------
def fig_timing(df):
    d = pv(df, "coupled").dropna(subset=["ms_per_inner"])
    dq = pq(df).dropna(subset=["t_inner_s", "inner_iter_pq"])
    if d.empty and dq.empty:
        print("  fig_timing: keine Zeitdaten"); return
    dq = dq.assign(ms_per_inner=1e3 * dq["t_inner_s"] / dq["inner_iter_pq"])
    nodes = sorted(set(d["nodes"]) | set(dq["nodes"]))
    fig, ax = _fig(1, 2, w=5.91, h=2.6)
    for n in nodes:
        for m, st in MODE_STYLE.items():
            g = dq[(dq["nodes"] == n) & (dq["mode"] == m)].sort_values("rx")
            if not g.empty:
                c = node_color(n, nodes)
                ax[0].plot(g["rx"], g["ms_per_inner"], ls=st["ls"],
                           marker=st["marker"], ms=3.5, lw=1.0, color=c,
                           mfc=c if st["fill"] else "none")
            g = d[(d["nodes"] == n) & (d["mode"] == m)].groupby("rx").median(
                numeric_only=True).reset_index()
            if not g.empty:
                c = node_color(n, nodes)
                ax[1].plot(g["rx"], 1e3 * g["t_outer_s"], ls=st["ls"],
                           marker=st["marker"], ms=3.5, lw=1.0, color=c,
                           mfc=c if st["fill"] else "none")
    for a in ax:
        _bands(a)
        a.grid(alpha=.3, which="both")
    ax[0].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel="ms je innerer Iteration",
              title="(a) Aufwand pro Iteration (PQ)")
    ax[1].set(xscale="log", yscale="log", xlabel="$R/X$",
              ylabel="ms gesamt (Median)",
              title="(b) Gesamtzeit Methode~A")
    h = [plt.Line2D([], [], color=node_color(n, nodes), lw=1.5,
                    label=f"$n={n}$") for n in nodes]
    ax[0].legend(handles=h, fontsize=6)
    _save(fig, "rx_timing")

    rows = []
    for (n, m), g in dq.groupby(["nodes", "mode"]):
        v = g["ms_per_inner"].dropna()
        if len(v) < 3:
            continue
        rows.append(dict(nodes=int(n), mode=m, ms_med=v.median(),
                         cv_pct=100 * v.std(ddof=1) / v.mean(),
                         slope=loglog_slope(g["rx"], g["ms_per_inner"])))
    if rows:
        t = pd.DataFrame(rows).set_index(["mode", "nodes"])
        save_table(t, "tab_rx_timing",
                   r"Rechenzeit je innerer Iteration: Median, "
                   r"Variationskoeffizient und log-log-Steigung \"uber $R/X$.",
                   "tab:rx-timing")
        for k, r in t.iterrows():
            fact(f"timing.cv_pct.{k[0]}.n{int(k[1])}", round(r["cv_pct"], 1), "%")
            fact(f"timing.ms_med.{k[0]}.n{int(k[1])}", round(r["ms_med"], 4), "ms")


# ----------------------------------------------------------------------
# 10) Divergenzmechanismus: Verlaeufe knapp unter/ueber rho*
# ----------------------------------------------------------------------
def fig_outer_histories(df, max_pairs=3):
    need = {"v_err_hist", "q_max_hist"}
    if not need <= set(df.columns):
        print("  fig_outer_histories: Historien fehlen in der CSV"); return
    d = pv(df)
    t = rho_star(d).dropna(subset=["rho_star"])
    if t.empty:
        print("  fig_outer_histories: kein rho* gefunden"); return
    pairs = []
    for _, r in t.sort_values("rho_star").iterrows():
        g = d[(d["nodes"] == r["nodes"]) & (d["n_pv"] == r["n_pv"]) &
              (d["mode"] == r["mode"]) & (d["variant"] == r["variant"])]
        below = g[(g["rx"] < r["rho_star"]) & (g["cls"] == "conv")]
        if below.empty:
            continue
        pairs.append((below.sort_values("rx").iloc[-1],
                      g[np.isclose(g["rx"], r["rho_star"])].iloc[0]))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        print("  fig_outer_histories: kein Paar unter/ueber rho*"); return

    fig, ax = _fig(len(pairs), 3, w=5.91, h=2.0, squeeze=False)
    for i, (lo, hi) in enumerate(pairs):
        for row, style, lbl in ((lo, dict(color="#2c7fb8", ls="-"),
                                 fr"$\rho={lo['rx']:.2g}$ (konv.)"),
                                (hi, dict(color="crimson", ls="--"),
                                 fr"$\rho={hi['rx']:.2g}$ ({hi['cls']})")):
            for j, key in enumerate(("v_err_hist", "q_max_hist",
                                     "sens_err_hist")):
                y = dec(row.get(key, ""))
                if y.size == 0:
                    continue
                ax[i, j].plot(np.arange(1, y.size + 1), np.abs(y), lw=1.2,
                              marker="o", ms=2.5, label=lbl, **style)
        ttl = (fr"$n={int(lo['nodes'])}$, $n_{{\mathrm{{PV}}}}="
               fr"{int(lo['n_pv'])}$, {lo['mode']}, {lo['variant']}"
               .replace("_", r"\_"))
        ax[i, 0].set_ylabel(r"$\varepsilon_V^{(\ell)}$")
        ax[i, 1].set_ylabel(r"$\max_k|Q_k|$")
        ax[i, 2].set_ylabel("rel. Sens.-Fehler")
        ax[i, 1].set_title(ttl, fontsize=7)
        for j in range(3):
            ax[i, j].set(yscale="log", xlabel=r"\"au\ss ere Iteration $\ell$")
            ax[i, j].grid(alpha=.3, which="both")
        ax[i, 0].legend(fontsize=5.5)
    _save(fig, "rx_outer_histories")

    for lo, hi in pairs:
        y = dec(hi.get("v_err_hist", ""))
        if y.size > 4:
            tail = y[len(y) // 2:]
            osc = float(np.mean(np.diff(np.sign(np.diff(tail))) != 0))
            fact(f"hist.osc_ratio.n{int(hi['nodes'])}.npv{int(hi['n_pv'])}"
                 f".{hi['mode']}.rho{hi['rx']:.2g}", round(osc, 2), "",
                 "Anteil Vorzeichenwechsel in eps_V (Oszillationsmass)")
            fact(f"hist.eps_last.n{int(hi['nodes'])}.npv{int(hi['n_pv'])}"
                 f".{hi['mode']}.rho{hi['rx']:.2g}", float(f"{y[-1]:.3g}"),
                 "p.u.")


# ----------------------------------------------------------------------
# 11) Tabelle rx-inner automatisch
# ----------------------------------------------------------------------
def tab_inner(df):
    d = pq(df)
    if d.empty:
        return
    rx_all = np.sort(d["rx"].unique())
    picks = [rx_all[0], rx_all[np.argmin(np.abs(rx_all - 1))], rx_all[-1]]
    rows = []
    for (n, m), g in d.groupby(["nodes", "mode"]):
        g = g.set_index("rx").sort_index()
        r = {"n": int(n), "Modus": m.replace("_", r"\_")}
        for p in picks:
            e = g["eta_use"].get(p, np.nan)
            r[f"eta({p:g})"] = e
        for p in (picks[0], picks[-1]):
            k = g["inner_iter_pq"].get(p, np.nan)
            okk = bool(g["inner_conv_pq"].get(p, False))
            r[f"k({p:g})"] = k if okk else np.nan
        lo, hi = r[f"eta({picks[0]:g})"], r[f"eta({picks[-1]:g})"]
        r["Faktor"] = hi / lo if np.isfinite(lo) and np.isfinite(hi) else np.nan
        r["v_min(hi)"] = g["v_min"].get(picks[-1], np.nan)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["Modus", "n"]).set_index(["Modus", "n"])
    save_table(t, "tab_rx_inner",
               r"Innere Fixpunktiteration im $R/X$-Sweep: gemessene "
               r"Kontraktionsrate, Iterationszahl und Arbeitspunkt. "
               r"Leere Felder bedeuten keine Konvergenz.", "tab:rx-inner")
    for k, r in t.iterrows():
        if np.isfinite(r["Faktor"]):
            fact(f"inner.eta_growth.{k[0]}.n{int(k[1])}",
                 round(float(r["Faktor"]), 2))


# ----------------------------------------------------------------------
# 12) Datenlücken
# ----------------------------------------------------------------------
def missing_report(df):
    nodes = sorted(df["nodes"].dropna().unique())
    modes = sorted(df["mode"].unique())
    ratios = sorted(df["pv_ratio"].dropna().unique())
    rxs = sorted(df["rx"].dropna().unique())
    variants = sorted(df.loc[df["pv_ratio"] > 0, "variant"].unique()) or ["coupled"]

    have = set(zip(df["nodes"], df["pv_ratio"].round(6), df["rx"].round(6),
                   df["mode"], df["variant"]))
    miss = []
    for n in nodes:
        for m in modes:
            for r in ratios:
                vs = ["coupled"] if r == 0 else variants
                for v in vs:
                    for rx in rxs:
                        if (n, round(r, 6), round(rx, 6), m, v) not in have:
                            miss.append(dict(nodes=int(n), pv_ratio=r, rx=rx,
                                             mode=m, variant=v,
                                             grund="nicht gerechnet"))
    sk = df[(df["skipped"] != "") | (df["cls"] == "error")]
    for _, r in sk.iterrows():
        miss.append(dict(nodes=int(r["nodes"]), pv_ratio=r["pv_ratio"],
                         rx=r["rx"], mode=r["mode"], variant=r["variant"],
                         grund=r["skipped"] or "error"))
    t = pd.DataFrame(miss)
    t.to_csv(OUT / "rx_missing.csv", index=False)
    fact("data.missing_jobs", int((t["grund"] == "nicht gerechnet").sum()))
    fact("data.skipped_jobs", int((t["grund"] != "nicht gerechnet").sum()))

    cmds = []
    if not t.empty:
        gaps = t[t["grund"] == "nicht gerechnet"]
        for (m, v), g in gaps.groupby(["mode", "variant"]):
            cmds.append(
                "python -m scripts.run_rx_sweep --resume --out rx_sweep.csv "
                f"--modes {m} --variants {v} "
                f"--nodes {' '.join(str(int(x)) for x in sorted(g['nodes'].unique()))} "
                f"--pv-ratios {' '.join(f'{x:g}' for x in sorted(g['pv_ratio'].unique()))}")
    (OUT / "rx_missing_cmd.txt").write_text("\n".join(cmds), encoding="utf-8")
    print(f"  -> rx_missing.csv ({len(t)} Zeilen), rx_missing_cmd.txt")


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def write_report(df):
    f = pd.DataFrame(FACTS)
    f.to_csv(OUT / "rx_facts.csv", index=False)

    def md(x):
        try:
            return x.to_markdown()
        except Exception:
            return "```\n" + x.to_string() + "\n```"

    lines = ["# R/X-Abschnitt - Auswertung", "",
             f"Zeilen in CSV: {len(df)}, "
             f"davon PQ: {len(pq(df))}, PV: {len(pv(df))}", "",
             "## Klassifikation der PV-Laeufe", "",
             md(pv(df)["cls"].value_counts().to_frame("faelle")), "",
             "## Kennzahlen", "", md(f), ""]
    for name, t in TABLES.items():
        lines += [f"## {name}", "", md(t), ""]
    (OUT / "rx_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> rx_facts.csv ({len(f)} Kennzahlen), rx_report.md")


def main():
    global OUT, OUTER_CAP

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFAULT)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--pgf", action="store_true")
    ap.add_argument("--outer-cap", type=int, default=OUTER_CAP)
    args = ap.parse_args()

    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    OUTER_CAP = args.outer_cap
    setup_mpl(args.pgf)

    df = load(args.csv)
    print(f"geladen: {args.csv} ({len(df)} Zeilen)")
    tab_inner(df)
    fig_vmin_check(df)
    fig_kappa_scaling(df)
    fig_xpp_structure_full(df)
    fig_q_sens(df)
    fig_rho_star(df)
    fig_variants(df)
    fig_predictor(df)
    fig_nr_categories(df)
    fig_timing(df)
    fig_outer_histories(df)
    missing_report(df)
    write_report(df)


if __name__ == "__main__":
    main()
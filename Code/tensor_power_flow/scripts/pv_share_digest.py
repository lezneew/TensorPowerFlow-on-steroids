# =============================================================================
# pv_share_report.py  —  erzeugt pv_share_report.md (T1..T4 + Verdikte H1..H6)
# =============================================================================
"""
Aufruf:  python pv_share_report.py --data results_pv_share --out pv_share_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FMT = "{:.4g}"


def md(df: pd.DataFrame, floatfmt: str = "%.4g") -> str:
    if df is None or df.empty:
        return "_keine Daten_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "--" if pd.isna(v) else floatfmt % v)
        else:
            d[c] = d[c].astype(str)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in d.values)
    return f"{head}\n{sep}\n{body}\n"


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    p = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    r = np.corrcoef(np.log(x[m]), np.polyval(p, np.log(x[m])))[0, 1] ** 2
    return float(p[0]), float(r)


def load(d, n):
    p = d / f"{n}.csv"
    return pd.read_csv(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results_pv_share")
    ap.add_argument("--out", default="pv_share_report.md")
    a = ap.parse_args()
    d = Path(a.data)
    e1, e1b, e3 = load(d, "e1"), load(d, "e1b"), load(d, "e3")
    e4, e5, e6, e7, e8 = (load(d, "e4"), load(d, "e5"), load(d, "e6"),
                          load(d, "e7"), load(d, "e8"))
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}

    L: list[str] = []
    A = L.append
    A("# Auswertung: Einfluss des PV-Anteils\n")
    A("Automatisch erzeugt aus den Sweep-CSVs. Alle Blindleistungen in "
      "Injektionskonvention.\n")
    A("## 0 Randbedingungen\n")
    A(md(pd.DataFrame([meta])) if meta else "_meta.json fehlt_\n")

    cpl = e1[(e1.variant == "coupled")] if e1 is not None else None
    dec = e1[(e1.variant == "decoupled")] if e1 is not None else None

    # ---------------- T1 -----------------------------------------------------
    A("## T1 Plateaukennzahlen der gekoppelten Variante (Median über Seeds)\n")
    if cpl is not None:
        g = cpl[cpl.converged].groupby(["n_bus", "n_pv"]).agg(
            share=("share", "median"), k_out=("k_out", "median"),
            k_in=("k_in", "median"), k_ratio=("k_ratio", "median"),
            cond=("cond", "median"), rho_J=("rho_jacobi", "median"),
            offdiag=("offdiag_share", "median"), v_min=("v_min", "median"),
            q_med=("q_med", "median"), q_max=("q_max", "median"),
            q_sum=("q_sum", "median"), eta_pq=("eta_pq", "median"),
            eta_fin=("eta_final", "median"),
            eps_lin=("eps_lin_meas", "median"),
            t_ms=("t_per_scen_ms", "median"),
            err_vm=("err_vm", "max"), err_q=("err_q_pu", "max"),
        ).reset_index()
        A(md(g))

    # ---------------- T2 -----------------------------------------------------
    A("## T2 Konvergenzquote gekoppelt vs. entkoppelt\n")
    if e1 is not None:
        t2 = e1.groupby(["n_bus", "variant"]).agg(
            n=("converged", "size"), quote=("converged", "mean"),
            k_out_med=("k_out", "median"), k_in_med=("k_in", "median"),
        ).reset_index()
        A(md(t2))
    if e3 is not None:
        A("### T2b Nach Platzierung (E3)\n")
        t2b = e3.groupby(["placement", "variant"]).agg(
            n=("converged", "size"), quote=("converged", "mean"),
            rho_J_med=("rho_jacobi", "median"),
            offdiag_med=("offdiag_share", "median"),
            k_out_med=("k_out", "median")).reset_index()
        A(md(t2b))
        A("### T2c Trennschärfe des Kriteriums $\\rho_\\mathrm{J}<1$ "
          "(entkoppelte Läufe)\n")
        gd = e3[e3.variant == "decoupled"].dropna(subset=["rho_jacobi"])
        pred = gd.rho_jacobi < 1.0
        cm = pd.crosstab(pred.rename("rho_J<1"), gd.converged.rename("konvergiert"))
        A(md(cm.reset_index()))
        acc = float((pred == gd.converged).mean())
        A(f"\nTrefferquote: **{acc:.3f}**  "
          f"(n = {len(gd)}); größtes $\\rho_\\mathrm{{J}}$ mit Konvergenz: "
          f"{gd[gd.converged].rho_jacobi.max():.3g}; kleinstes ohne: "
          f"{gd[~gd.converged].rho_jacobi.min():.3g}\n")

    # ---------------- T3 -----------------------------------------------------
    A("## T3 Spearman-Rangkorrelationen (gekoppelt, konvergente Läufe)\n")
    if cpl is not None:
        g = cpl[cpl.converged & (cpl.n_pv > 0)]
        preds = ["n_pv", "share", "cond", "rho_jacobi", "offdiag_share",
                 "coupling_mean", "min_diag_off", "xkk_mean", "depth_pv_mean",
                 "v_min", "eta_pq", "eps_lin_meas", "q_max", "q_med", "n_bus"]
        t3 = pd.DataFrame([dict(Prädiktor=p,
                                k_out=sp(g[p], g.k_out),
                                k_in=sp(g[p], g.k_in),
                                t_pro_Szenario=sp(g[p], g.t_per_scen_ms))
                           for p in preds if p in g])
        A(md(t3.sort_values("k_out", key=abs, ascending=False)))

    # ---------------- T4 -----------------------------------------------------
    A("## T4 Skalierungsexponenten\n")
    rows = []
    if e4 is not None:
        for (n, st), g in e4.groupby(["n_bus", "placement"]):
            gg = g[g.n_pv > 1]
            s1, r1 = loglog_slope(gg.n_pv, gg["cond"])
            s2, r2 = loglog_slope(gg.n_pv, gg.rho_jacobi)
            npv_crit = np.nan
            m = gg.sort_values("n_pv")
            over = m[m.rho_jacobi > 1.0]
            if len(over):
                npv_crit = float(over.n_pv.iloc[0])
            rows.append(dict(Größe="X_pp", n_bus=n, Platzierung=st,
                             a_cond=s1, R2_cond=r1, b_rhoJ=s2, R2_rhoJ=r2,
                             npv_bei_rhoJ_1=npv_crit))
    if cpl is not None:
        for n, g in cpl[cpl.converged & (cpl.n_pv > 0)].groupby("n_bus"):
            s, r = loglog_slope(g.n_pv, g.k_in)
            rows.append(dict(Größe="k_in(n_pv)", n_bus=n, Platzierung="random",
                             a_cond=s, R2_cond=r, b_rhoJ=np.nan, R2_rhoJ=np.nan,
                             npv_bei_rhoJ_1=np.nan))
    if e7 is not None:
        for n, g in e7[e7.tau == e7.tau.max()].groupby("n_bus"):
            s, r = loglog_slope(g.k_in, g.t_per_scen_ms)
            rows.append(dict(Größe="t_inf(k_in)", n_bus=n, Platzierung="-",
                             a_cond=s, R2_cond=r, b_rhoJ=np.nan, R2_rhoJ=np.nan,
                             npv_bei_rhoJ_1=np.nan))
    A(md(pd.DataFrame(rows)))

    # ---------------- Hypothesen --------------------------------------------
    A("## Verdikte\n")

    # H1
    A("### H1  Treiber ist $n_\\mathrm{pv}$, nicht der Anteil\n")
    if cpl is not None:
        g = cpl[cpl.converged & (cpl.n_pv > 0)]
        def cv(by):
            out = []
            for _, gg in g.groupby(by):
                if gg.n_bus.nunique() < 2:
                    continue
                m = gg.groupby("n_bus").k_out.median()
                out.append(m.std() / max(m.mean(), 1e-12))
            return float(np.mean(out)) if out else np.nan
        cv_npv = cv("n_pv")
        cv_shr = cv(g.share.round(2).rename("share_r"))
        A(md(pd.DataFrame([dict(Gruppierung="gleiche n_pv",
                                CV_über_Netzgröße=cv_npv),
                           dict(Gruppierung="gleicher Anteil",
                                CV_über_Netzgröße=cv_shr)])))
        A(f"\nSpearman: $n_\\mathrm{{pv}}\\to k_\\mathrm{{out}}$ "
          f"{sp(g.n_pv, g.k_out):+.3f}, Anteil $\\to k_\\mathrm{{out}}$ "
          f"{sp(g.share, g.k_out):+.3f}, $n_\\mathrm{{bus}}\\to k_\\mathrm{{out}}$ "
          f"{sp(g.n_bus, g.k_out):+.3f}\n")
        A(f"\n**Verdikt:** H1 {'bestätigt' if cv_npv < cv_shr else 'nicht bestätigt'} "
          f"(CV {cv_npv:.3f} vs. {cv_shr:.3f}).\n")

    # H2
    A("### H2  Strukturinvarianz der gekoppelten Korrektur\n")
    if cpl is not None:
        g = cpl[cpl.converged & (cpl.n_pv > 1)]
        A(f"- Spannweite $\\mathrm{{cond}}(X_{{pp}})$: "
          f"{g['cond'].min():.3g} … {g['cond'].max():.3g} "
          f"(Faktor {g['cond'].max()/max(g['cond'].min(),1e-12):.1f})\n")
        A(f"- Spannweite $k_\\mathrm{{out}}$: {g.k_out.min():.0f} … "
          f"{g.k_out.max():.0f} (Faktor "
          f"{g.k_out.max()/max(g.k_out.min(),1):.2f})\n")
        A(f"- Spearman(cond, $k_\\mathrm{{out}}$) = {sp(g['cond'], g.k_out):+.3f}, "
          f"Spearman($\\rho_\\mathrm{{J}}$, $k_\\mathrm{{out}}$) = "
          f"{sp(g.rho_jacobi, g.k_out):+.3f}\n")
        A(f"- Konvergenzquote gekoppelt insgesamt: "
          f"{cpl.converged.mean():.3f}\n")

    # H3
    A("### H3  Versagen der entkoppelten Näherung bei $\\rho_\\mathrm{J}>1$\n")
    if e3 is not None:
        gd = e3[e3.variant == "decoupled"]
        tab = []
        for st, g in gd.groupby("placement"):
            ok = g[g.converged]
            bad = g[~g.converged]
            tab.append(dict(Platzierung=st,
                            npv_max_konvergent=(ok.n_pv.max() if len(ok) else np.nan),
                            npv_min_divergent=(bad.n_pv.min() if len(bad) else np.nan),
                            rhoJ_max_konv=(ok.rho_jacobi.max() if len(ok) else np.nan),
                            rhoJ_min_div=(bad.rho_jacobi.min() if len(bad) else np.nan)))
        A(md(pd.DataFrame(tab)))
        A("\nDie Schwelle in $n_\\mathrm{pv}$ verschiebt sich mit der Platzierung, "
          "die Schwelle in $\\rho_\\mathrm{J}$ nicht — genau das ist die Aussage "
          "von H3.\n")

    # H4 / H5
    A("### H4/H5  Stabilisierung und Sollwertartefakt\n")
    if cpl is not None:
        g = cpl[cpl.converged & (cpl.n_pv > 0)]
        A(f"- Spearman($n_\\mathrm{{pv}}$, $v_\\mathrm{{min}}$) = "
          f"{sp(g.n_pv, g.v_min):+.3f}\n")
        A(f"- Spearman($n_\\mathrm{{pv}}$, med$|Q_k|$) = "
          f"{sp(g.n_pv, g.q_med):+.3f}; ($n_\\mathrm{{pv}}$, $\\sum|Q_k|$) = "
          f"{sp(g.n_pv, g.q_sum):+.3f}\n")
        A(f"- Spearman($n_\\mathrm{{pv}}$, $\\eta_\\mathrm{{final}}$) = "
          f"{sp(g.n_pv, g.eta_final):+.3f}\n")
    if e5 is not None:
        A("\n**E5, Sollwertmodus (n = 200):**\n")
        t = e5[e5.converged].groupby(["sp_mode", "delta"]).agg(
            n=("k_out", "size"), k_out=("k_out", "median"),
            k_in=("k_in", "median"), q_max=("q_max", "median"),
            q_med=("q_med", "median"), eps_lin=("eps_lin_meas", "median"),
            v_min=("v_min", "median"), quote=("converged", "mean")).reset_index()
        A(md(t))
        for mode in e5.sp_mode.unique():
            gg = e5[(e5.sp_mode == mode) & e5.converged & (e5.n_pv > 0)]
            A(f"- Modus `{mode}`: Spearman($n_\\mathrm{{pv}}$, med$|Q|$) = "
              f"{sp(gg.n_pv, gg.q_med):+.3f}, "
              f"Spearman($n_\\mathrm{{pv}}$, $k_\\mathrm{{out}}$) = "
              f"{sp(gg.n_pv, gg.k_out):+.3f}\n")
        A("\nBleibt das Vorzeichen im Modus `abs` erhalten, ist H4 echt; "
          "kippt es, war der Effekt Kalibrierungsartefakt (H5).\n")

    # H6
    A("### H6  Laufzeit ausschließlich über $k_\\mathrm{in}$\n")
    if e7 is not None:
        pl = e7[e7.tau == e7.tau.max()]
        rows = []
        for n, g in pl.groupby("n_bus"):
            b = g[g.n_pv == 0]
            if not len(b):
                continue
            for _, r in g.iterrows():
                rows.append(dict(n_bus=n, n_pv=r.n_pv,
                                 k_rel=r.k_in / b.k_in.values[0],
                                 t_rel=r.t_per_scen_ms / b.t_per_scen_ms.values[0],
                                 gflops=r.gflops,
                                 Verhältnis=(r.t_per_scen_ms / b.t_per_scen_ms.values[0])
                                 / max(r.k_in / b.k_in.values[0], 1e-12)))
        t = pd.DataFrame(rows)
        A(md(t))
        tt = t[t.n_pv > 0]
        A(f"\nAbweichung von der Proportionalität: Median "
          f"{tt.Verhältnis.median():.3f}, Spanne "
          f"{tt.Verhältnis.min():.3f} … {tt.Verhältnis.max():.3f}. "
          f"GFLOP/s-Streuung je Netzgröße: "
          + ", ".join(f"n={int(n)}: {g.gflops.min():.1f}–{g.gflops.max():.1f}"
                      for n, g in pl.groupby("n_bus")) + "\n")
        A("\n### Batch-Verhalten (E7 vollständig)\n")
        A(md(e7[["n_bus", "n_pv", "tau", "k_out", "k_in", "t_pre_ms",
                 "t_solve_ms", "t_per_scen_ms", "gflops", "conv_share"]]))

    # Praxisgrenze
    A("## Praxisgrenze: Blindleistungsfähigkeit (E6)\n")
    if e6 is not None:
        t = e6.groupby(["n_bus", "q_lim_pu"], dropna=False).agg(
            n=("converged", "size"), quote=("converged", "mean"),
            sat=("sat_share", "median"), k_out=("k_out", "median"),
            v_err=("pv_v_err", "median")).reset_index()
        A(md(t))
        ub = e6[e6.q_lim_pu.isna()]
        A(f"\nOhne Grenzen auftretendes $\\max|Q_k|$: bis "
          f"{ub.q_max.max():.3g} p.u.; erste Nichtkonvergenz mit "
          f"$|Q|\\leq0{{,}}33$ p.u. bei "
          f"{e6[(e6.q_lim_pu==0.33)&(~e6.converged)].n_pv.min()} PV-Knoten.\n")

    # Interaktionen
    A("## Interaktionen (E8)\n")
    if e8 is not None:
        for axis in e8.axis.unique():
            g = e8[e8.axis == axis]
            A(f"\n### PV-Anteil × `{axis}` — $k_\\mathrm{{out}}$ "
              f"(x = keine Konvergenz)\n")
            piv = g.pivot_table(index="share", columns=axis, values="k_out",
                                aggfunc="median")
            cv = g.pivot_table(index="share", columns=axis, values="converged",
                               aggfunc="min")
            show = piv.where(cv.astype(bool)).round(0)
            A(md(show.reset_index()))
            A(f"\nSpearman(Anteil, $k_\\mathrm{{out}}$) = {sp(g.share, g.k_out):+.3f}, "
              f"Spearman(`{axis}`, $k_\\mathrm{{out}}$) = {sp(g[axis], g.k_out):+.3f}\n")

    # Genauigkeit
    A("## E9 Genauigkeit gegen Newton-Raphson\n")
    if cpl is not None:
        g = cpl.dropna(subset=["err_vm"])
        t = g.groupby("n_bus").agg(
            n=("err_vm", "size"), err_vm_max=("err_vm", "max"),
            err_va_max=("err_va_deg", "max"), err_q_max=("err_q_pu", "max"),
            pv_v_err_max=("pv_v_err", "max"),
            nr_iter_med=("nr_iter", "median")).reset_index()
        A(md(t))
        A(f"\nSpearman($n_\\mathrm{{pv}}$, $\\max|\\Delta V|$) = "
          f"{sp(g.n_pv, g.err_vm):+.3f} — die Genauigkeit ist "
          "unabhängig von der PV-Zahl, wenn dieser Wert nahe null liegt.\n")
        A("\n### Kreuzvergleich Konvergenz\n")
        gg = cpl.dropna(subset=["nr_conv"])
        cm = pd.crosstab(gg.converged.rename("Methode A"),
                         gg.nr_conv.rename("NR"))
        A(md(cm.reset_index()))

    # Kontrolle Lastbilanz
    A("## Kontrolle: Wirkleistungseinspeisung der PV-Knoten (E1b)\n")
    if e1b is not None:
        t = e1b.groupby(["pv_p_factor", "n_pv"]).agg(
            k_out=("k_out", "median"), k_in=("k_in", "median"),
            v_min=("v_min", "median"), v_max=("v_max", "median"),
            q_max=("q_max", "median"), quote=("converged", "mean")).reset_index()
        A(md(t))
        A("\nUnterscheiden sich die Blöcke deutlich, ist die Wirkleistungsbilanz "
          "eine relevante Störgröße und muss im Text getrennt berichtet werden.\n")

    A("\n## Abbildungsverweise\n")
    A("| Datei | Aussage |\n|---|---|\n"
      "| p1_kout_kin_vs_npv | Plateau (gekoppelt) vs. Abbruchkante (entkoppelt) |\n"
      "| p2_collapse_share_vs_npv | H1-Kollaps |\n"
      "| p3_placement_coupling | H3, Platzierung und $\\rho_\\mathrm{J}$ |\n"
      "| p4_xpp_structure | cond / $\\rho_\\mathrm{J}$ über $n_\\mathrm{pv}$ |\n"
      "| p5_q_vs_npv | H4/H5, Blindleistung und Sollwert |\n"
      "| p6_qlims | Praxisgrenze Q-Limits |\n"
      "| p7_time_vs_iterations | H6 |\n"
      "| p8_interaction_heatmaps | Interaktion mit $\\lambda$ und $R/X$ |\n"
      "| p9_xpp_heatmaps | $X_{pp}$ geclustert vs. verteilt |\n")

    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"geschrieben: {a.out}")


if __name__ == "__main__":
    main()
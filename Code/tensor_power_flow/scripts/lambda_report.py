#!/usr/bin/env python3
# lambda_report.py
"""
Erzeugt results_lastfaktor.md aus den CSV-Dateien von lambda_sweep.py.
Aufruf:  python lambda_report.py results_lastfaktor
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

FMT = {"eta_emp": "{:.5f}", "eta1": "{:.4f}", "eta2": "{:.4f}", "etainf": "{:.4f}",
       "eta2_nom": "{:.4f}", "kappa": "{:.5f}", "kappa_n": "{:.3e}",
       "v_min": "{:.4f}", "v_max": "{:.4f}", "v_min_base": "{:.4f}",
       "lam": "{:.2f}", "lam_star": "{:.2f}", "eps_med": "{:.3f}", "eps_max": "{:.3f}",
       "eps_cf_med": "{:.3f}", "q_max": "{:.3e}", "q_util": "{:.2f}",
       "t_ms": "{:.3f}", "t_tpf_ms": "{:.3f}", "t_nr_ms": "{:.2f}",
       "err_final": "{:.2e}", "dv_max_vs_nr": "{:.2e}", "feas_min": "{:.2f}",
       "rho_s": "{:+.3f}", "speedup": "{:.2f}", "dev_pct": "{:.2f}", "r2": "{:.5f}"}


def md_table(df, fmt=None, max_rows=None):
    if df is None or len(df) == 0:
        return "_keine Daten_"
    d = df if max_rows is None else df.head(max_rows)
    fmt = {**FMT, **(fmt or {})}
    cols = list(d.columns)

    def cell(v, c):
        if isinstance(v, (bool, np.bool_)):
            return "ja" if v else "nein"
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "--"
            return fmt.get(c, "{:.4g}").format(v)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        return str(v)

    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(cell(r[c], c) for c in cols) + " |")
    return "\n".join(out)


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    yh = a + b * np.log(x[m])
    ss = np.sum((np.log(y[m]) - np.log(y[m]).mean()) ** 2)
    r2 = 1 - np.sum((np.log(y[m]) - yh) ** 2) / ss if ss > 0 else np.nan
    return float(b), float(r2)


def load(d: Path, name):
    p = d / name
    return pd.read_csv(p) if p.exists() else None


def build(out: Path):
    out = Path(out)
    e1 = load(out, "e1_inner.csv")
    e2 = load(out, "e2_outer.csv")
    st = load(out, "e2_outer_steps.csv")
    e3 = load(out, "e3_limits.csv")
    e4 = load(out, "e4_damping.csv")
    e5 = load(out, "e5_optim.csv")
    e6 = load(out, "e6_continuation.csv")
    e7 = load(out, "e7_lambda_rx.csv")
    e8 = load(out, "e8_batch.csv")
    e9 = load(out, "e9_modes.csv")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) \
        if (out / "meta.json").exists() else {}

    S, H = [], {}
    A = S.append

    A("# Ergebnisdaten: Einfluss des Lastfaktors\n")
    A("Automatisch erzeugt von `lambda_sweep.py` / `lambda_report.py`.\n")

    # ── 0 Setup ──────────────────────────────────────────────────────────
    A("## 0 Versuchsaufbau\n")
    cfgd = meta.get("config", {})
    keys = ["n_bus_list", "pv_share_list", "lam_lo", "lam_hi", "n_lam", "rx_ref",
            "rx_mode", "z0_ohm_km", "len_km", "vn_kv", "s_base_mva", "cos_phi",
            "v_min_target_lam1", "pv_p_total_ratio", "inverter_oversize",
            "dv_setpoint", "tol_inner", "tol_rate", "tol_pv", "max_inner",
            "max_outer", "omega", "omega_list", "branch_span", "seed", "repeats"]
    A(md_table(pd.DataFrame([{"Parameter": k, "Wert": str(cfgd.get(k))} for k in keys if k in cfgd])))
    A("")
    A(f"- Produktions-Solver verfügbar: **{meta.get('production_solver', {}).get('ok')}** "
      f"({meta.get('production_solver', {}).get('reason', '')[:120]})")
    A(f"- pandapower-Referenz: **{meta.get('pandapower')}**, Laufzeit Sweep: "
      f"{meta.get('runtime_s', float('nan')):.1f} s")
    A("- Normierung: `load_scale` je Netz so kalibriert, dass $v_{\\min}(\\lambda=1)=$ "
      f"{cfgd.get('v_min_target_lam1')}. Damit ist $\\lambda$ über $n$ hinweg "
      "betriebspunkt-normiert; absolute MW-Werte sind sekundär.")
    A("- Sollwerte der PV-Knoten je $\\lambda$ neu kalibriert (`setpoint_mode=calibrated`, "
      "$|v_{pv}^{base}|+\\delta$) bzw. fest auf $1{,}00$ p.u. (`fixed`).")
    A("- Blindleistung in Lastkonvention des Solvers (negativ = Einspeisung); berichtet wird $\\max_k|Q_k|$.\n")

    # ── 1 Innere Schleife ────────────────────────────────────────────────
    if e1 is not None:
        A("## 1 Innere Fixpunktiteration (H1)\n")
        c = e1[e1.conv]
        piv = c.pivot_table(index="lam", columns="n", values="eta_emp")
        A("### 1.1 Gemessene Kontraktionsrate $\\eta_{\\mathrm{emp}}$ über $\\lambda$\n")
        A(md_table(piv.reset_index().rename(columns={cc: f"n={cc}" for cc in piv.columns}),
                   fmt={f"n={cc}": "{:.5f}" for cc in piv.columns}))
        A("")
        pk = c.pivot_table(index="lam", columns="n", values="k_in_6")
        A("### 1.2 Innere Iterationen bis $10^{-6}$\n")
        A(md_table(pk.reset_index().rename(columns={cc: f"n={cc}" for cc in pk.columns}),
                   fmt={f"n={cc}": "{:.0f}" for cc in pk.columns}))
        A("")
        rows = []
        for n, g in c.groupby("n"):
            sl_lam, r2l = loglog_slope(g.lam, g.eta_emp)
            rows.append(dict(n=n, points=len(g),
                             lam_min=g.lam.min(), lam_max=g.lam.max(),
                             eta_min=g.eta_emp.min(), eta_max=g.eta_emp.max(),
                             faktor=g.eta_emp.max() / g.eta_emp.min(),
                             slope_lnEta_lnLam=sl_lam, r2=r2l,
                             v_min_max=g.v_min.max(), v_min_min=g.v_min.min()))
        dfr = pd.DataFrame(rows)
        A("### 1.3 Skalierung\n")
        A(md_table(dfr, fmt={"faktor": "{:.2f}", "slope_lnEta_lnLam": "{:.3f}",
                             "eta_min": "{:.5f}", "eta_max": "{:.5f}",
                             "v_min_max": "{:.4f}", "v_min_min": "{:.4f}"}))
        A("")
        H["H1_slope_lam"] = float(np.nanmedian(dfr.slope_lnEta_lnLam))
        # Vorhersagetest
        pr = []
        for n, g in c.groupby("n"):
            g = g.sort_values("lam")
            lo, hi = g.iloc[0], g.iloc[-1]
            pred = lo.eta_emp * (hi.lam / lo.lam) * (lo.v_min / hi.v_min) ** 2
            pr.append(dict(n=n, lam_1=lo.lam, lam_2=hi.lam, eta_1=lo.eta_emp,
                           eta_2_pred=pred, eta_2_mess=hi.eta_emp,
                           dev_pct=100 * (pred - hi.eta_emp) / hi.eta_emp))
        A("### 1.4 Vorhersagetest $\\eta(\\lambda_2)\\approx\\eta(\\lambda_1)\\,"
          "\\frac{\\lambda_2}{\\lambda_1}\\bigl(v_{\\min,1}/v_{\\min,2}\\bigr)^2$\n")
        A(md_table(pd.DataFrame(pr), fmt={"eta_1": "{:.5f}", "eta_2_pred": "{:.5f}",
                                          "eta_2_mess": "{:.5f}"}))
        A("")
        H["H1_pred_dev"] = float(np.nanmax(np.abs(pd.DataFrame(pr).dev_pct)))
        # k_in Vorhersage
        kk = c.dropna(subset=["k_pred"])
        A(f"- Geometrischer Abfall: Median $R^2$ des Log-Fits = "
          f"**{np.nanmedian(c.eta_r2):.6f}**, "
          f"mittlere Abweichung $k_{{\\mathrm{{pred}}}}$ vs. $k_{{\\mathrm{{in}}}}$ = "
          f"**{np.nanmean(np.abs(kk.k_pred - kk.k_in_12)):.2f}** Iterationen\n")

        # Collapse
        A("## 2 Reskalierung auf eine gemeinsame Kennzahl (H1/H2)\n")
        c = c.copy()
        c["kappa"] = c.eta_emp * c.v_min ** 2 / (c.lam * c.z_rel)
        c["kappa_n"] = c.kappa / c.n
        rows = []
        for n, g in c.groupby("n"):
            k = g.kappa.dropna()
            rows.append(dict(n=n, punkte=len(k), kappa_mean=k.mean(),
                             dev_pct=100 * np.max(np.abs(k - k.mean())) / k.mean(),
                             eta_wachstum=g.eta_emp.max() / g.eta_emp.min()))
        dfk = pd.DataFrame(rows)
        A("$\\kappa \\coloneqq \\eta\\,v_{\\min}^2/(\\lambda\\,z_{\\mathrm{rel}})$ — "
          "parameterfrei, kann scheitern.\n")
        A(md_table(dfk, fmt={"kappa_mean": "{:.5f}", "eta_wachstum": "{:.2f}"}))
        A("")
        sl_n, r2n = loglog_slope(dfk.n, dfk.kappa_mean)
        A(f"- Exponent in $n$: $\\kappa\\propto n^{{{sl_n:.3f}}}$ ($R^2={r2n:.4f}$) "
          f"$\\Rightarrow\\ \\eta\\approx\\kappa_1\\,n\\lambda\\lvert z\\rvert/v_{{\\min}}^2$ mit "
          f"$\\kappa_1\\approx{np.nanmean(dfk.kappa_mean / dfk.n):.3e}$")
        A(f"- Maximale Restschwankung von $\\kappa$: **{dfk.dev_pct.max():.1f} %** bei "
          f"$\\eta$-Wachstum bis Faktor **{dfk.eta_wachstum.max():.1f}**\n")
        H["H1_kappa_dev"] = float(dfk.dev_pct.max())
        H["H1_kappa_slope_n"] = float(sl_n)
        c.to_csv(out / "derived_e1_kappa.csv", index=False)

    # ── 3 Nase / Lösbarkeit ─────────────────────────────────────────────
    if e3 is not None:
        A("## 3 Grenzen: $\\lambda^\\ast$ je Kriterium (H2/H3/H6)\n")
        piv = e3.pivot_table(index=["n", "pv_share"], columns="criterion",
                             values="lam_star").reset_index()
        A(md_table(piv, fmt={c: "{:.2f}" for c in piv.columns if c not in ("n", "pv_share")}))
        A("")
        A("Lesart: `base_fpi` = reine PQ-Lösbarkeit über FPI, `eta_lt_1` = Verlust der "
          "Kontraktion, `nr_pq`/`nr_pv` = Newton-Raphson-Referenz, `tpf_methode_a` = "
          "äußere Schleife, `eps_lin_0.6` = Frühwarnschwelle, `q_reserve` = "
          "Überschreitung der Wechselrichter-Blindleistungsreserve.\n")
        try:
            d = piv.dropna(subset=["eta_lt_1", "base_fpi"])
            H["H2_eta_vs_base"] = float(np.nanmedian(d.eta_lt_1 / d.base_fpi))
            d2 = piv.dropna(subset=["tpf_methode_a", "base_fpi"])
            H["H3_outer_vs_inner"] = float(np.nanmedian(d2.tpf_methode_a / d2.base_fpi))
            d3 = piv.dropna(subset=["q_reserve", "tpf_methode_a"])
            H["H_prakt_q_vs_num"] = float(np.nanmedian(d3.q_reserve / d3.tpf_methode_a))
        except Exception:
            pass

    # ── 4 Äußere Schleife ────────────────────────────────────────────────
    if e2 is not None:
        A("## 4 Äußere Blindleistungskorrektur (H3/H4)\n")
        cal = e2[(e2.setpoint_mode == "calibrated") & (e2.base_ok == True)]
        conv = cal[cal.tpf_conv == True]
        A("### 4.1 Äußere Iterationen $k_{\\mathrm{out}}$ über $\\lambda$ (kalibrierte Sollwerte)\n")
        piv = conv.pivot_table(index="lam", columns=["n", "pv_share"], values="k_out")
        piv.columns = [f"n{a}/pv{b:.2f}" for a, b in piv.columns]
        A(md_table(piv.reset_index(), fmt={c: "{:.0f}" for c in piv.columns}))
        A("")
        A("### 4.2 Linearisierungsfehler $\\varepsilon_{\\mathrm{lin}}$ (Median über äußere Schritte)\n")
        piv2 = conv.pivot_table(index="lam", columns=["n", "pv_share"], values="eps_med")
        piv2.columns = [f"n{a}/pv{b:.2f}" for a, b in piv2.columns]
        A(md_table(piv2.reset_index(), fmt={c: "{:.3f}" for c in piv2.columns}))
        A("")
        rows = []
        for (n, pv), g in cal.groupby(["n", "pv_share"]):
            s1, r1 = loglog_slope(g.lam, g.eps_med)
            s2, r2 = loglog_slope(g.lam, g.q_max)
            rows.append(dict(n=n, pv_share=pv,
                             slope_eps_lam=s1, r2_eps=r1,
                             slope_q_lam=s2, r2_q=r2,
                             eps_min=g.eps_med.min(), eps_max=g.eps_med.max(),
                             q_faktor=g.q_max.max() / max(g.q_max.min(), 1e-30)))
        A("### 4.3 Skalierung von $\\varepsilon_{\\mathrm{lin}}$ und $\\max_k|Q_k|$ über $\\lambda$\n")
        A(md_table(pd.DataFrame(rows), fmt={"slope_eps_lam": "{:.3f}", "slope_q_lam": "{:.3f}",
                                            "r2_eps": "{:.3f}", "r2_q": "{:.3f}",
                                            "eps_min": "{:.3f}", "eps_max": "{:.3f}",
                                            "q_faktor": "{:.1f}"}))
        A("")
        H["H3_eps_slope"] = float(np.nanmedian([r["slope_eps_lam"] for r in rows]))

        # Frühwarnschwelle
        A("### 4.4 Trennschärfe der Schwelle $\\varepsilon_{\\mathrm{lin}}\\lesssim 0{,}6$\n")
        b = cal.dropna(subset=["eps_med"])
        lo = b[b.eps_med <= 0.6]; hi = b[b.eps_med >= 0.92]
        A(md_table(pd.DataFrame([
            dict(klasse="eps_med <= 0.60", faelle=len(lo),
                 anteil_konvergent=100 * lo.tpf_conv.mean() if len(lo) else np.nan,
                 k_out_max=lo.k_out.max() if len(lo) else np.nan,
                 k_out_med=lo.k_out.median() if len(lo) else np.nan),
            dict(klasse="0.60 < eps_med < 0.92",
                 faelle=len(b) - len(lo) - len(hi),
                 anteil_konvergent=100 * b[(b.eps_med > 0.6) & (b.eps_med < 0.92)].tpf_conv.mean()
                 if len(b) - len(lo) - len(hi) else np.nan,
                 k_out_max=b[(b.eps_med > 0.6) & (b.eps_med < 0.92)].k_out.max()
                 if len(b) - len(lo) - len(hi) else np.nan,
                 k_out_med=np.nan),
            dict(klasse="eps_med >= 0.92", faelle=len(hi),
                 anteil_konvergent=100 * hi.tpf_conv.mean() if len(hi) else np.nan,
                 k_out_max=hi.k_out.max() if len(hi) else np.nan, k_out_med=np.nan),
        ]), fmt={"anteil_konvergent": "{:.1f}"}))
        A("")

        # Klassifikation + NR-Kreuztabelle
        A("### 4.5 Versagensklassen\n")
        cl = e2.groupby("cls").size().rename("Fälle").reset_index()
        A(md_table(cl))
        A("")
        fails = e2[(e2.tpf_conv == False) & (e2.base_ok == True)][
            ["n", "pv_share", "setpoint_mode", "lam", "err_final", "q_max",
             "kin_per_kout", "eps_med", "v_min", "nr_conv", "cls"]]
        A("Alle nicht konvergenten Läufe mit lösbarer Basis:\n")
        A(md_table(fails.sort_values(["n", "pv_share", "lam"]), max_rows=60,
                   fmt={"kin_per_kout": "{:.1f}"}))
        A("")
        ct = e2.dropna(subset=["nr_conv"]).copy()
        ct["nr_conv"] = ct.nr_conv.astype(bool)
        tab = pd.crosstab(ct.tpf_conv, ct.nr_conv)
        A("### 4.6 Kreuztabelle TPF vs. Newton-Raphson\n")
        A(md_table(tab.reset_index().rename(columns={False: "NR nein", True: "NR ja",
                                                     "tpf_conv": "TPF konvergiert"})))
        A("")
        only_tpf = int(((ct.tpf_conv == True) & (ct.nr_conv == False)).sum())
        only_nr = int(((ct.tpf_conv == False) & (ct.nr_conv == True)).sum())
        neither = int(((ct.tpf_conv == False) & (ct.nr_conv == False)).sum())
        A(f"- nur NR (= echte Verfahrensgrenze): **{only_nr}**, nur TPF: **{only_tpf}**, "
          f"keines (= keine erreichbare Lösung): **{neither}**\n")
        H["H6_only_nr"] = only_nr
        H["H6_only_tpf"] = only_tpf

        # Genauigkeit
        acc = cal.dropna(subset=["dv_max_vs_nr"])
        if len(acc):
            A(f"- Maximaler Spannungsfehler gegenüber NR über alle konvergenten Läufe: "
              f"**{acc.dv_max_vs_nr.max():.2e} p.u.**\n")

        # Setpoint-Modus
        if e2.setpoint_mode.nunique() > 1:
            A("### 4.7 Kalibrierter vs. fester Sollwert (1,00 p.u.)\n")
            g = e2[e2.base_ok == True].groupby("setpoint_mode").agg(
                faelle=("lam", "size"), konv_pct=("tpf_conv", lambda x: 100 * np.mean(x == True)),
                k_out_med=("k_out", "median"), eps_med=("eps_med", "median"),
                q_max_med=("q_max", "median"), q_util_max=("q_util", "max")).reset_index()
            A(md_table(g, fmt={"konv_pct": "{:.1f}", "k_out_med": "{:.1f}",
                               "q_max_med": "{:.3e}", "q_util_max": "{:.2f}"}))
            A("")

        # Q-Bedarf
        A("### 4.8 Blindleistungsbedarf vs. Wechselrichterreserve\n")
        qq = cal.dropna(subset=["q_util"]).groupby(["n", "pv_share"]).agg(
            q_util_lam_min=("q_util", "min"), q_util_lam_max=("q_util", "max"),
            lam_at_q1=("lam", lambda s: np.nan)).reset_index()
        firsts = []
        for (n, pv), g in cal.groupby(["n", "pv_share"]):
            g = g.sort_values("lam")
            v = g[g.q_util > 1.0]
            firsts.append(dict(n=n, pv_share=pv,
                               lam_q_util_gt_1=float(v.lam.iloc[0]) if len(v) else np.nan,
                               q_util_max=g.q_util.max()))
        A(md_table(pd.DataFrame(firsts), fmt={"lam_q_util_gt_1": "{:.2f}", "q_util_max": "{:.2f}"}))
        A("")

        # Spearman
        if HAVE_SCIPY:
            A("### 4.9 Rangkorrelationen (Spearman) über alle konvergenten PV-Läufe\n")
            d = conv.copy()
            cand = ["eps_med", "eps_max", "eps_cf_med", "lam", "v_min", "v_min_base",
                    "q_max", "eta_emp", "eta2", "feas_min", "n", "n_pv", "pv_share"]
            rows = []
            for c in cand:
                if c not in d:
                    continue
                x = d[c].astype(float)
                for tgt in ("k_out", "k_in"):
                    m = np.isfinite(x) & np.isfinite(d[tgt].astype(float))
                    if m.sum() < 8:
                        continue
                    rho = spearmanr(x[m], d[tgt][m]).correlation
                    rows.append((c, tgt, rho))
            piv = pd.DataFrame(rows, columns=["Größe", "Ziel", "rho_s"]).pivot(
                index="Größe", columns="Ziel", values="rho_s").reset_index()
            piv = piv.reindex(piv.iloc[:, 1:].abs().max(axis=1).sort_values(ascending=False).index)
            A(md_table(piv, fmt={"k_out": "{:+.3f}", "k_in": "{:+.3f}"}))
            A("")
            try:
                H["H4_rho_eps"] = float(piv.set_index("Größe").loc["eps_med", "k_out"])
                H["H4_rho_lam"] = float(piv.set_index("Größe").loc["lam", "k_out"])
                H["H_rho_vmin"] = float(piv.set_index("Größe").loc["v_min", "k_out"])
            except Exception:
                pass

        # Rechenzeit
        A("### 4.10 Rechenzeit TPF (Methode A) vs. Newton-Raphson\n")
        t = conv.dropna(subset=["t_nr_ms"]).groupby(["n"]).agg(
            t_tpf_med=("t_tpf_ms", "median"), t_tpf_min=("t_tpf_ms", "min"),
            t_tpf_max=("t_tpf_ms", "max"), t_nr_med=("t_nr_ms", "median")).reset_index()
        t["speedup_med"] = t.t_nr_med / t.t_tpf_med
        A(md_table(t, fmt={"t_tpf_med": "{:.3f}", "t_tpf_min": "{:.3f}",
                           "t_tpf_max": "{:.3f}", "t_nr_med": "{:.2f}",
                           "speedup_med": "{:.2f}"}))
        A("")
        br = []
        for n, g in conv.dropna(subset=["t_nr_ms"]).groupby("n"):
            g = g.sort_values("lam")
            bad = g[g.t_tpf_ms > g.t_nr_ms]
            br.append(dict(n=n, lam_break_even=float(bad.lam.iloc[0]) if len(bad) else np.nan))
        A("Kleinstes $\\lambda$, ab dem TPF langsamer als NR ist:\n")
        A(md_table(pd.DataFrame(br)))
        A("")

        if "prod_k_out" in e2.columns and e2.prod_k_out.notna().any():
            d = e2.dropna(subset=["prod_k_out"])
            same = int((d.prod_k_out == d.k_out).sum())
            A(f"- Gegenprobe Produktions-Solver: $k_{{\\mathrm{{out}}}}$ identisch in "
              f"**{same}/{len(d)}** Läufen; "
              f"max. Abweichung $k_{{\\mathrm{{in}}}}$ = "
              f"**{int(np.nanmax(np.abs(d.prod_k_in - d.k_in)))}**\n")

    # ── 5 Per-Schritt-Verläufe ───────────────────────────────────────────
    if st is not None and len(st):
        A("## 5 Verlauf innerhalb der äußeren Schleife\n")
        g = st.groupby(["n", "pv_share", "setpoint_mode", "lam"]).agg(
            k_out=("ell", "max"), k_in=("k_in", "sum"),
            kin_per_kout=("k_in", "mean"), eps_first=("eps_med", "first"),
            eps_last=("eps_med", lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
            q_max=("q_max", "max")).reset_index()
        A("Aggregat je Lauf (Auszug):\n")
        A(md_table(g.sort_values(["n", "pv_share", "lam"]), max_rows=40,
                   fmt={"kin_per_kout": "{:.1f}", "eps_first": "{:.3f}", "eps_last": "{:.3f}"}))
        A("")

    # ── 6 Dämpfung ───────────────────────────────────────────────────────
    if e4 is not None:
        A("## 6 Wirkung der Dämpfung $\\omega$\n")
        piv = e4.pivot_table(index=["n", "pv_share"], columns="omega",
                             values="lam_star").reset_index()
        piv.columns = [c if isinstance(c, str) else f"omega={c}" for c in piv.columns]
        A(md_table(piv, fmt={c: "{:.2f}" for c in piv.columns if c.startswith("omega")}))
        A("")
        try:
            base = e4[e4.omega == e4.omega.max()].set_index(["n", "pv_share"]).lam_star
            best = e4.groupby(["n", "pv_share"]).lam_star.max()
            H["H_damping_gain"] = float(np.nanmedian(best / base))
            A(f"- Median-Verschiebung von $\\lambda^\\ast$ durch die beste Dämpfung: "
              f"Faktor **{H['H_damping_gain']:.2f}**\n")
        except Exception:
            pass

    # ── 7 Optimierungen ─────────────────────────────────────────────────
    if e5 is not None:
        A("## 7 Warm Start und adaptive innere Toleranz über $\\lambda$ (H5)\n")
        d = e5[e5.conv == True].copy()
        d["variante"] = np.where(d.warm, "warm", "cold") + "/" + np.where(d.adaptive, "adapt", "fix")
        piv = d.pivot_table(index=["n", "pv_share", "lam"], columns="variante",
                            values="k_in").reset_index()
        for a, b, nm in (("cold/fix", "warm/fix", "gain_warm"),
                         ("warm/fix", "warm/adapt", "gain_adapt"),
                         ("cold/fix", "warm/adapt", "gain_total")):
            if a in piv and b in piv:
                piv[nm] = piv[a] / piv[b]
        A("Innere Iterationen je Variante (Auszug):\n")
        A(md_table(piv, max_rows=40, fmt={c: "{:.0f}" for c in
                                          ("cold/fix", "warm/fix", "cold/adapt", "warm/adapt")}
                   | {"gain_warm": "{:.2f}", "gain_adapt": "{:.2f}", "gain_total": "{:.2f}"}))
        A("")
        agg = []
        for n, g in piv.groupby("n"):
            r = dict(n=n)
            for nm in ("gain_warm", "gain_adapt", "gain_total"):
                if nm in g:
                    r[nm + "_med"] = float(np.nanmedian(g[nm]))
                    r[nm + "_min"] = float(np.nanmin(g[nm]))
                    r[nm + "_max"] = float(np.nanmax(g[nm]))
            agg.append(r)
        A("Einsparungsfaktoren je Netzgröße:\n")
        A(md_table(pd.DataFrame(agg), fmt={k: "{:.2f}" for k in
                                           sum([[f"{a}_{b}" for b in ("med", "min", "max")]
                                                for a in ("gain_warm", "gain_adapt", "gain_total")], [])}))
        A("")
        if "gain_total" in piv:
            sl, r2 = loglog_slope(piv.lam, piv.gain_total)
            H["H5_gain_med"] = float(np.nanmedian(piv.gain_total))
            H["H5_gain_slope_lam"] = sl
            A(f"- Median-Gesamteinsparung: **{H['H5_gain_med']:.2f}**, "
              f"Trend über $\\lambda$: $\\propto\\lambda^{{{sl:.2f}}}$ ($R^2={r2:.2f}$)\n")

    # ── 8 Fortsetzung ────────────────────────────────────────────────────
    if e6 is not None:
        A("## 8 Q-Startwert: $\\vect{Q}=\\vect{0}$ vs. $\\lambda$-Fortsetzung\n")
        d = e6[e6.conv == True]
        piv = d.pivot_table(index=["n", "pv_share", "lam"], columns="q_init_mode",
                            values=["k_out", "k_in", "eps_med"]).reset_index()
        piv.columns = ["_".join([str(x) for x in c if x]) for c in piv.columns]
        A(md_table(piv, max_rows=40))
        A("")
        try:
            a = d[d.q_init_mode == "cold_q0"].set_index(["n", "pv_share", "lam"])
            b = d[d.q_init_mode == "continuation"].set_index(["n", "pv_share", "lam"])
            j = a.join(b, lsuffix="_cold", rsuffix="_cont", how="inner")
            H["H_cont_kout"] = float(np.nanmedian(j.k_out_cold / j.k_out_cont))
            H["H_cont_eps"] = float(np.nanmedian(j.eps_med_cont / j.eps_med_cold))
            A(f"- Median $k_{{\\mathrm{{out}}}}$-Einsparung durch Fortsetzung: "
              f"**{H['H_cont_kout']:.2f}**; Median-Verhältnis "
              f"$\\varepsilon_{{\\mathrm{{lin}}}}$ (cont/cold): **{H['H_cont_eps']:.2f}**\n")
        except Exception:
            pass

    # ── 9 (lambda, R/X) ──────────────────────────────────────────────────
    if e7 is not None and len(e7):
        A("## 9 Gemeinsamer Prädiktor: $(\\lambda, R/X)$\n")
        piv = e7.pivot_table(index="lam", columns="rx", values="eps_med")
        A(md_table(piv.reset_index().rename(columns={c: f"rx={c}" for c in piv.columns}),
                   fmt={f"rx={c}": "{:.3f}" for c in piv.columns}))
        A("")
        piv2 = e7.pivot_table(index="lam", columns="rx", values="k_out")
        A(md_table(piv2.reset_index().rename(columns={c: f"rx={c}" for c in piv2.columns}),
                   fmt={f"rx={c}": "{:.0f}" for c in piv2.columns}))
        A("")
        if HAVE_SCIPY:
            d = e7[e7.conv == True].dropna(subset=["eps_med"])
            if len(d) > 8:
                A(f"- Spearman $\\varepsilon_{{\\mathrm{{lin}}}}\\to k_{{\\mathrm{{out}}}}$ "
                  f"im 2D-Gitter: **{spearmanr(d.eps_med, d.k_out).correlation:+.3f}** "
                  f"(vs. $\\lambda$: {spearmanr(d.lam, d.k_out).correlation:+.3f}, "
                  f"vs. $R/X$: {spearmanr(d.rx, d.k_out).correlation:+.3f})\n")

    # ── 10 Batch ─────────────────────────────────────────────────────────
    if e8 is not None and len(e8):
        A("## 10 Batch mit gemischten $\\lambda$\n")
        A(md_table(e8, fmt={"t_ms": "{:.1f}"}))
        A("")

    # ── 11 Modi ──────────────────────────────────────────────────────────
    if e9 is not None and len(e9):
        A("## 11 Skalierungsmodus (Last / Last+PV / nur PV)\n")
        d = e9[e9.base_ok == True]
        g = d.groupby(["n", "pv_share", "scale_mode"]).agg(
            faelle=("lam", "size"), konv_pct=("tpf_conv", lambda x: 100 * np.mean(x == True)),
            k_out_med=("k_out", "median"), eps_med=("eps_med", "median"),
            q_max_med=("q_max", "median"), v_min=("v_min", "min"),
            v_max=("v_max", "max")).reset_index()
        A(md_table(g, fmt={"konv_pct": "{:.1f}", "k_out_med": "{:.1f}",
                           "q_max_med": "{:.3e}"}))
        A("")

    # ── 12 Hypothesen ────────────────────────────────────────────────────
    A("## 12 Hypothesen-Check\n")
    rows = [
        dict(H="H1", Aussage="eta überlinear in lambda; kappa-Kollaps trägt",
             Kennzahl=f"slope(lnEta/lnLam)={H.get('H1_slope_lam', float('nan')):.2f}, "
                      f"max dev kappa={H.get('H1_kappa_dev', float('nan')):.1f} %, "
                      f"kappa ~ n^{H.get('H1_kappa_slope_n', float('nan')):.2f}",
             Verdikt="gestützt" if H.get("H1_kappa_dev", 99) < 10 else "zu prüfen"),
        dict(H="H2", Aussage="Kontraktionsverlust = Lösbarkeitsgrenze",
             Kennzahl=f"lambda*(eta=1)/lambda*(base_fpi)={H.get('H2_eta_vs_base', float('nan')):.2f}",
             Verdikt="gestützt" if abs(H.get("H2_eta_vs_base", 0) - 1) < 0.15 else "zu prüfen"),
        dict(H="H3", Aussage="äußere Schleife versagt früher als innere",
             Kennzahl=f"lambda*(TPF)/lambda*(base_fpi)={H.get('H3_outer_vs_inner', float('nan')):.2f}, "
                      f"slope(eps_lin)={H.get('H3_eps_slope', float('nan')):.2f}",
             Verdikt="gestützt" if H.get("H3_outer_vs_inner", 9) < 1.0 else "zu prüfen"),
        dict(H="H4", Aussage="eps_lin bester Prädiktor, Schwelle 0,6 überträgt sich",
             Kennzahl=f"rho_s(eps)={H.get('H4_rho_eps', float('nan')):+.2f} vs. "
                      f"rho_s(lambda)={H.get('H4_rho_lam', float('nan')):+.2f}",
             Verdikt="gestützt" if abs(H.get("H4_rho_eps", 0)) > abs(H.get("H4_rho_lam", 1))
             else "zu prüfen"),
        dict(H="H5", Aussage="Warm Start + adaptive Toleranz bleiben über lambda wirksam",
             Kennzahl=f"Median-Gewinn={H.get('H5_gain_med', float('nan')):.2f}, "
                      f"Trend lambda^{H.get('H5_gain_slope_lam', float('nan')):.2f}",
             Verdikt="gestützt" if H.get("H5_gain_med", 0) > 1.2 else "zu prüfen"),
        dict(H="H6", Aussage="kein Fall, in dem nur TPF scheitert und NR löst? / Robustheit",
             Kennzahl=f"nur NR={H.get('H6_only_nr', '--')}, nur TPF={H.get('H6_only_tpf', '--')}",
             Verdikt="siehe Kreuztabelle"),
    ]
    A(md_table(pd.DataFrame(rows)))
    A("")
    A("Weitere abgeleitete Kennzahlen:\n")
    A(md_table(pd.DataFrame([{"Kennzahl": k, "Wert": (f"{v:.3f}" if isinstance(v, float) else v)}
                             for k, v in H.items()])))
    A("")

    p = out / "results_lastfaktor.md"
    p.write_text("\n".join(S), encoding="utf-8")
    (out / "derived_hypotheses.json").write_text(json.dumps(H, indent=2), encoding="utf-8")
    print(f"Report geschrieben: {p}")


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "results_lastfaktor"))
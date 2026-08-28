#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_lambda_figures.py
Erzeugt die PGF-Abbildungen zur Section "Einfluss des Lastfaktors" direkt aus
den Sweep-Ergebnissen (CSV) bzw. ersatzweise aus results_lastfaktor.md.

Beispiele:
    python make_lambda_figures.py --csv-dir out --outdir figures
    python make_lambda_figures.py --md results_lastfaktor.md --outdir figures --pdf
    python make_lambda_figures.py --csv-dir out --only eta kin --n-ref 40 --pv-ref 0.1

Erzeugte Dateien (Zielbreiten bezogen auf \\textwidth = 15,0 cm):
    lam_eta.pgf       0.48\\textwidth      lam_q.pgf         0.78\\textwidth
    lam_kin.pgf       0.48\\textwidth      lam_opt.pgf       0.85\\textwidth
    lam_kout_pv.pgf   0.48\\textwidth      lam_damping.pgf   0.48\\textwidth
    lam_kout_n.pgf    0.48\\textwidth
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pgf")                     # muss vor pyplot stehen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

# ======================================================================
# 1  Layout, abgestimmt auf packages.tex / geometry
# ======================================================================
TEXTWIDTH_CM = 15.0        # 21.0 - 3.0 - 2.5 - 0.5 (bindingoffset)

PGF_PREAMBLE = "\n".join([
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{lmodern}",
    r"\usepackage{amsmath}",
    r"\usepackage{bm}",
])

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": PGF_PREAMBLE,
    "font.family": "serif",
    "font.serif": [],
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "lines.linewidth": 0.9,
    "lines.markersize": 2.6,
    "grid.linewidth": 0.4,
    "grid.color": "0.86",
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.labelspacing": 0.28,
    "legend.borderaxespad": 0.25,
    "savefig.bbox": None,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
})

CYCLE = ["#B01414", "#12468F", "#1A7A33", "#8A5A00", "#5B2D8E"]
MARKERS = ["o", "s", "^", "v", "D"]
LINESTYLES = ["-", (0, (3.2, 2.0)), (0, (1.2, 1.4)), (0, (5.0, 1.6, 1.0, 1.6))]
GRAY = "0.45"
DASH = (0, (3.2, 2.0))


def figsize_cm(w_cm: float, h_cm: float) -> tuple[float, float]:
    return (w_cm / 2.54, h_cm / 2.54)


def de(v, dec: int | None = None) -> str:
    """Zahl mit deutschem Dezimalkomma als Mathe-Label."""
    if dec is None:
        s = f"{v:g}"
    else:
        s = f"{v:.{dec}f}"
    return "$" + s.replace(".", "{,}").replace("-", r"\text{-}") + "$"


DE_FMT = FuncFormatter(lambda v, _: de(v))


def style_axes(ax, grid_axis: str = "y"):
    ax.grid(True, axis=grid_axis, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out")


def nice_log_ticks(lo: float, hi: float) -> list[float]:
    base = [1, 2, 5]
    ticks = []
    for e in range(-6, 7):
        for b in base:
            v = b * 10.0 ** e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def vline_lambda_star(ax, lam_star, label=r"$\lambda^\ast$", frac=0.06):
    if lam_star is None or not np.isfinite(lam_star):
        return
    ax.axvline(lam_star, color=GRAY, ls=DASH, lw=0.6, zorder=1)
    y0, y1 = ax.get_ylim()
    if ax.get_yscale() == "log":
        y = 10 ** (np.log10(y0) + frac * (np.log10(y1) - np.log10(y0)))
    else:
        y = y0 + frac * (y1 - y0)
    ax.text(lam_star, y, label, color=GRAY, fontsize=7,
            rotation=90, ha="right", va="bottom")


def slope_triangle(ax, x, y, dex=0.35, frac_x=0.32, frac_y=0.40):
    """Steigungsdreieck aus dem Log-Log-Fit der uebergebenen Reihe."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return None
    s, c = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    lx = np.log(x[m])
    x0 = float(np.exp(lx.min() + frac_x * (lx.max() - lx.min())))
    y0 = frac_y * float(np.exp(c) * x0 ** s)
    x1 = x0 * 10 ** dex
    y1 = y0 * 10 ** (s * dex)
    ax.plot([x0, x1], [y0, y0], color=GRAY, lw=0.5, zorder=2)
    ax.plot([x1, x1], [y0, y1], color=GRAY, lw=0.5, zorder=2)
    ax.plot([x0, x1], [y0, y1], color=GRAY, lw=0.7, ls=DASH, zorder=2)
    ax.text(x1 * 1.06, float(np.sqrt(y0 * y1)), de(round(s, 2), 2),
            color=GRAY, fontsize=7, ha="left", va="center")
    return s


def pad_lin(vals, lo_zero=True, pad=0.08):
    v = np.asarray([x for x in np.ravel(vals) if np.isfinite(x)], float)
    hi = v.max() * (1 + pad)
    lo = 0.0 if lo_zero else v.min() - pad * (v.max() - v.min())
    return lo, hi


def pad_log(vals, pad=0.12):
    v = np.asarray([x for x in np.ravel(vals) if np.isfinite(x) and x > 0], float)
    lo, hi = np.log10(v.min()), np.log10(v.max())
    d = max(hi - lo, 0.3)
    return 10 ** (lo - pad * d), 10 ** (hi + pad * d)


# ======================================================================
# 2  IO: Spaltennormalisierung und Alias-Auflösung
# ======================================================================
ALIASES = {
    # Achsen / Schluessel
    "lam": "lam", "lambda": "lam", "lam_load": "lam", "load_factor": "lam",
    "lastfaktor": "lam", "lam_value": "lam",
    "n": "n", "n_bus": "n", "n_buses": "n", "nbus": "n", "n_bus_total": "n",
    "pv_share": "pv_share", "pv": "pv_share", "pvshare": "pv_share",
    "share_pv": "pv_share", "pv_anteil": "pv_share",
    "setpoint_mode": "setpoint_mode", "sp_mode": "setpoint_mode",
    "setpoint": "setpoint_mode", "mode_setpoint": "setpoint_mode",
    "scale_mode": "scale_mode", "omega": "omega", "damping": "omega",
    # Kennzahlen
    "eta": "eta_emp", "eta_emp": "eta_emp", "eta_empirical": "eta_emp",
    "eta_measured": "eta_emp", "eta_mess": "eta_emp",
    "k_in": "k_in", "kin": "k_in", "iter_inner": "k_in", "n_inner": "k_in",
    "inner_iters": "k_in", "k_in_total": "k_in",
    "k_out": "k_out", "kout": "k_out", "iter_outer": "k_out",
    "n_outer": "k_out", "outer_iters": "k_out",
    "eps_med": "eps_med", "eps_lin_med": "eps_med", "eps_median": "eps_med",
    "eps_lin_median": "eps_med", "eps_medium": "eps_med",
    "q_max": "q_max", "q_abs_max": "q_max", "max_q": "q_max",
    "q_max_pu": "q_max", "qmax": "q_max",
    "v_min": "v_min", "vmin": "v_min", "v_min_base": "v_min_base",
    "cls": "cls", "class": "cls", "fail_class": "cls", "klasse": "cls",
    "converged": "converged", "conv": "converged",
    # Loesbarkeitsgrenzen
    "lam_star": "lam_star", "lambda_star": "lam_star", "lam_crit": "lam_star",
    "base_fpi": "base_fpi", "eta_lt_1": "eta_lt_1",
    "tpf_methode_a": "tpf_a", "tpf_method_a": "tpf_a", "tpf_a": "tpf_a",
    "lam_star_tpf": "tpf_a", "nr_pq": "nr_pq", "nr_pv": "nr_pv",
    "eps_lin_0.6": "eps_lin_06", "eps_lin_06": "eps_lin_06",
    "q_reserve": "q_reserve",
    # Warm Start / adaptive Toleranz
    "cold_fix": "k_cold_fix", "cold_adapt": "k_cold_adapt",
    "warm_fix": "k_warm_fix", "warm_adapt": "k_warm_adapt",
    "k_in_cold_fix": "k_cold_fix", "k_in_warm_fix": "k_warm_fix",
    "k_in_warm_adapt": "k_warm_adapt", "k_in_cold_adapt": "k_cold_adapt",
}

MISSING = {"", "--", "-", "—", "n/a", "na", "nan", "none", "null"}


def norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("$", "").replace("\\", "").replace("{,}", ".")
    s = s.replace("%", "pct").replace("/", "_").replace(" ", "_")
    s = re.sub(r"[^0-9a-z_=.\-]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def parse_num(x):
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace("*", "").replace("~", "")
    if s.lower() in MISSING:
        return np.nan
    s = s.replace(" ", "")
    if s.count(",") == 1 and s.count(".") == 0:      # deutsches Komma
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [ALIASES.get(norm_key(c), norm_key(c)) for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    for c in out.columns:
        if c in {"setpoint_mode", "cls", "scale_mode"}:
            out[c] = out[c].astype(str).str.strip().str.lower()
        else:
            out[c] = out[c].map(parse_num)
    if "n" in out:
        out["n"] = out["n"].astype("Int64")
    return out


def load_csv_tables(csv_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for p in sorted(csv_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as exc:                      # noqa: BLE001
            print(f"  Warnung: {p.name} nicht lesbar ({exc})", file=sys.stderr)
            continue
        tables[p.stem] = normalize_frame(df)
    return tables


def pick_table(tables: dict[str, pd.DataFrame], required: set[str],
               need_any: set[str] | None = None,
               optional: set[str] | None = None) -> pd.DataFrame | None:
    """Waehlt die CSV-Tabelle mit den passenden Spalten (beste Deckung)."""
    best, score_best = None, (-1, -1)
    for name, df in tables.items():
        cols = set(df.columns)
        if not required <= cols:
            continue
        if need_any and not (need_any & cols):
            continue
        score = (len((optional or set()) & cols), len(df))
        if score > score_best:
            best, score_best = df.copy(), score
    return best


# ======================================================================
# 3  Fallback: Markdown-Report parsen
# ======================================================================
def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_markdown_tables(path: Path) -> list[tuple[str, pd.DataFrame]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out, heading, i = [], "", 0
    while i < len(lines):
        raw = lines[i].strip()
        if raw.startswith("#"):
            heading = raw.lstrip("#").strip()
            i += 1
            continue
        is_sep = (i + 1 < len(lines) and lines[i + 1].strip().startswith("|")
                  and "-" in lines[i + 1]
                  and set(lines[i + 1].strip()) <= set("|-: "))
        if raw.startswith("|") and is_sep:
            header = _split_row(raw)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = _split_row(lines[i].strip())
                r = (r + [""] * len(header))[:len(header)]
                rows.append(r)
                i += 1
            out.append((heading, pd.DataFrame(rows, columns=header)))
            continue
        i += 1
    return out


def md_find(tabs, section: str) -> pd.DataFrame | None:
    for heading, df in tabs:
        if heading.split(" ")[0].rstrip(".") == section:
            return df
    return None


def _melt(df: pd.DataFrame, pattern: str, keys: list[str], id_col: str,
          value_name: str) -> pd.DataFrame:
    """Wide -> long anhand eines Spaltenmuster-Regex."""
    rows = []
    for col in df.columns:
        m = re.fullmatch(pattern, norm_key(col))
        if not m:
            continue
        meta = {k: parse_num(v) for k, v in zip(keys, m.groups())}
        for _, r in df.iterrows():
            rec = dict(meta)
            rec[id_col] = parse_num(r[df.columns[0]]) if id_col == "lam" else None
            rec[value_name] = parse_num(r[col])
            rows.append(rec)
    return pd.DataFrame(rows)


def bundle_from_markdown(path: Path) -> dict[str, pd.DataFrame]:
    tabs = parse_markdown_tables(path)
    b: dict[str, pd.DataFrame] = {}

    # 1.1 / 1.2: Spalten "n=40" ...
    eta = _melt(md_find(tabs, "1.1"), r"n=(\d+)", ["n"], "lam", "eta_emp")
    kin = _melt(md_find(tabs, "1.2"), r"n=(\d+)", ["n"], "lam", "k_in")
    if len(eta) and len(kin):
        b["inner"] = eta.merge(kin, on=["n", "lam"], how="outer")
    elif len(eta):
        b["inner"] = eta

    # 4.1 / 4.2: Spalten "n40/pv0.10" ...
    kout = _melt(md_find(tabs, "4.1"), r"n(\d+)_pv([\d.]+)",
                 ["n", "pv_share"], "lam", "k_out")
    eps = _melt(md_find(tabs, "4.2"), r"n(\d+)_pv([\d.]+)",
                ["n", "pv_share"], "lam", "eps_med")
    if len(kout):
        b["outer"] = (kout.merge(eps, on=["n", "pv_share", "lam"], how="left")
                      if len(eps) else kout)
        b["outer"]["setpoint_mode"] = "calibrated"

    # 3: Loesbarkeitsgrenzen, 5: Laufaggregat, 6: Daempfung, 7: Warm Start
    for sec, key in (("3", "lamstar"), ("5", "runs")):
        t = md_find(tabs, sec)
        if t is not None:
            b[key] = normalize_frame(t)
    t6 = md_find(tabs, "6")
    if t6 is not None:
        base = normalize_frame(t6[[t6.columns[0], t6.columns[1]]])
        dmp = _melt(t6, r"omega=([\d.]+)", ["omega"], "lam", "lam_star")
        dmp = dmp.drop(columns=["lam"])
        k = len(base)
        dmp["n"] = list(base["n"]) * (len(dmp) // k)
        dmp["pv_share"] = list(base["pv_share"]) * (len(dmp) // k)
        b["damping"] = dmp
    t7 = md_find(tabs, "7")
    if t7 is not None:
        b["warm"] = normalize_frame(t7)
    return b


# ======================================================================
# 4  Datenbündel aufbauen (CSV bevorzugt, Markdown als Rückfall)
# ======================================================================
SPECS = {
    "inner":   dict(required={"n", "lam"}, need_any={"eta_emp", "k_in"},
                    optional={"eta_emp", "k_in", "v_min"}),
    "outer":   dict(required={"n", "pv_share", "lam", "k_out"}, need_any=None,
                    optional={"setpoint_mode", "eps_med", "q_max", "k_in", "cls"}),
    "runs":    dict(required={"n", "pv_share", "lam", "q_max"},
                    need_any={"setpoint_mode"},
                    optional={"k_out", "k_in", "eps_med", "setpoint_mode"}),
    "lamstar": dict(required={"n"}, need_any={"base_fpi", "eta_lt_1", "tpf_a"},
                    optional={"pv_share", "base_fpi", "eta_lt_1", "tpf_a"}),
    "warm":    dict(required={"n", "lam"},
                    need_any={"k_cold_fix", "k_warm_fix", "k_warm_adapt"},
                    optional={"pv_share", "k_cold_fix", "k_cold_adapt",
                              "k_warm_fix", "k_warm_adapt"}),
    "damping": dict(required={"n", "omega", "lam_star"}, need_any=None,
                    optional={"pv_share"}),
}


def build_bundle(csv_dir: Path | None, md_path: Path | None) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    if csv_dir and csv_dir.is_dir():
        tables = load_csv_tables(csv_dir)
        print(f"CSV-Quelle: {csv_dir} ({len(tables)} Dateien)")
        for key, spec in SPECS.items():
            df = pick_table(tables, spec["required"], spec["need_any"],
                            spec["optional"])
            if df is not None:
                bundle[key] = df
    missing = [k for k in SPECS if k not in bundle]
    if missing and md_path and md_path.is_file():
        print(f"Markdown-Rückfall für: {', '.join(missing)}")
        md = bundle_from_markdown(md_path)
        for key in missing:
            if key in md:
                bundle[key] = md[key]
    for key, df in bundle.items():
        df.dropna(how="all", axis=1, inplace=True)
        print(f"  {key:8s}: {len(df):4d} Zeilen, Spalten = {list(df.columns)}")
    return bundle


def series_by(df: pd.DataFrame, group: str, xcol: str, ycol: str):
    """Sortierte (Label-Wert, x, y)-Tupel je Gruppe."""
    if df is None or ycol not in df or xcol not in df:
        return []
    d = df.dropna(subset=[xcol, ycol])
    out = []
    for gv, sub in d.groupby(group, dropna=True):
        sub = sub.sort_values(xcol)
        out.append((gv, sub[xcol].to_numpy(float), sub[ycol].to_numpy(float)))
    return sorted(out, key=lambda t: float(t[0]))


def lam_star_of(bundle, col, agg="median", pv_filter=None):
    df = bundle.get("lamstar")
    if df is None or col not in df:
        return None
    d = df
    if pv_filter is not None and "pv_share" in d:
        d = d[np.isclose(d["pv_share"].astype(float), pv_filter, equal_nan=False)] \
            if pv_filter > 0 else d[d["pv_share"].astype(float) == 0]
    v = pd.to_numeric(d[col], errors="coerce").dropna()
    if v.empty:
        return None
    return float(getattr(v, agg)())


# ======================================================================
# 5  Abbildungen
# ======================================================================
def fig_eta(b, cfg):
    df = b.get("inner")
    if df is None or "eta_emp" not in df:
        return None
    ser = series_by(df, "n", "lam", "eta_emp")
    fig, ax = plt.subplots(figsize=figsize_cm(0.48 * TEXTWIDTH_CM, 5.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    for i, (n, x, y) in enumerate(ser):
        ax.plot(x, y, color=CYCLE[i % len(CYCLE)], ls=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)], mfc="none", mew=0.6, ms=2.2,
                label=rf"$n={int(n)}$", zorder=4 + i)
    ax.axhline(1.0, color=GRAY, ls=DASH, lw=0.6, zorder=1)
    allx = np.concatenate([s[1] for s in ser])
    ally = np.concatenate([s[2] for s in ser])
    ax.set_xlim(*pad_log(allx, 0.06))
    ax.set_ylim(min(pad_log(ally)[0], 4e-3), max(pad_log(ally)[1], 1.5))
    ax.text(ax.get_xlim()[0] * 1.05, 1.08, r"$\eta=1$", color=GRAY,
            fontsize=7, va="bottom")
    slope_triangle(ax, ser[-1][1], ser[-1][2])
    vline_lambda_star(ax, lam_star_of(b, "eta_lt_1") or lam_star_of(b, "base_fpi"))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\eta_{\mathrm{emp}}$")
    ax.xaxis.set_major_locator(FixedLocator(nice_log_ticks(*ax.get_xlim())))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(DE_FMT)
    style_axes(ax, "both")
    ax.legend(loc="lower right")
    return fig


def fig_kin(b, cfg):
    df = b.get("inner")
    if df is None or "k_in" not in df:
        return None
    ser = series_by(df, "n", "lam", "k_in")
    fig, ax = plt.subplots(figsize=figsize_cm(0.48 * TEXTWIDTH_CM, 5.6))
    for i, (n, x, y) in enumerate(ser):
        ax.plot(x, y, color=CYCLE[i % len(CYCLE)], ls=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)], mfc="none", mew=0.6, ms=2.2,
                label=rf"$n={int(n)}$")
    ax.set_xlim(0, max(s[1].max() for s in ser) * 1.08)
    ax.set_ylim(*pad_lin([s[2] for s in ser]))
    vline_lambda_star(ax, lam_star_of(b, "base_fpi") or lam_star_of(b, "eta_lt_1"))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$k_{\mathrm{in}}$")
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.xaxis.set_major_formatter(DE_FMT)
    ax.yaxis.set_major_formatter(DE_FMT)
    style_axes(ax)
    ax.legend(loc="upper left")
    return fig


def _kout_frame(b):
    df = b.get("outer")
    if df is None or "k_out" not in df:
        return None
    if "setpoint_mode" in df:
        cal = df[df["setpoint_mode"].astype(str).str.startswith("cal")]
        if len(cal):
            df = cal
    if "cls" in df:
        df = df[~df["cls"].astype(str).isin({"divergence", "no_solution"})]
    return df.dropna(subset=["lam", "k_out"])


def _kout_axes(ser, xlabel_star=None):
    fig, ax = plt.subplots(figsize=figsize_cm(0.48 * TEXTWIDTH_CM, 5.6))
    for i, (lbl, x, y) in enumerate(ser):
        ax.plot(x, y, color=CYCLE[i % len(CYCLE)], ls=LINESTYLES[i % len(LINESTYLES)],
                marker=MARKERS[i % len(MARKERS)], mfc="none", mew=0.6, ms=2.2,
                label=lbl)
    ax.set_xlim(0, max(s[1].max() for s in ser) * 1.08)
    ax.set_ylim(*pad_lin([s[2] for s in ser]))
    vline_lambda_star(ax, xlabel_star)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$k_{\mathrm{out}}$")
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.xaxis.set_major_formatter(DE_FMT)
    ax.yaxis.set_major_formatter(DE_FMT)
    style_axes(ax)
    ax.legend(loc="upper left")
    return fig


def fig_kout_pv(b, cfg):
    df = _kout_frame(b)
    if df is None:
        return None
    d = df[df["n"].astype(float) == cfg["n_ref"]]
    ser = [(rf"${100 * float(pv):.0f}\,\%$ PV", x, y)
           for pv, x, y in series_by(d, "pv_share", "lam", "k_out")]
    if not ser:
        return None
    return _kout_axes(ser, lam_star_of(b, "tpf_a", agg="min"))


def fig_kout_n(b, cfg):
    df = _kout_frame(b)
    if df is None:
        return None
    d = df[np.isclose(df["pv_share"].astype(float), cfg["pv_ref"])]
    ser = [(rf"$n={int(n)}$", x, y)
           for n, x, y in series_by(d, "n", "lam", "k_out")]
    if not ser:
        return None
    return _kout_axes(ser)


def fig_q(b, cfg):
    df = b.get("runs")
    if df is None or "q_max" not in df or "setpoint_mode" not in df:
        return None
    d = df[(df["n"].astype(float) == cfg["n_ref"])
           & np.isclose(df["pv_share"].astype(float), cfg["pv_ref"])]
    d = d.dropna(subset=["lam", "q_max"])
    if d.empty:
        return None
    labels = {"fixed": r"fester Sollwert $1{,}00$~p.u.",
              "calibrated": r"kalibrierter Sollwert"}
    fig, ax = plt.subplots(figsize=figsize_cm(0.78 * TEXTWIDTH_CM, 6.4))
    for i, (mode, sub) in enumerate(sorted(d.groupby("setpoint_mode"))):
        sub = sub.sort_values("lam")
        ax.plot(sub["lam"], sub["q_max"], color=CYCLE[i % len(CYCLE)],
                ls=LINESTYLES[i % len(LINESTYLES)], marker=MARKERS[i % len(MARKERS)],
                mfc="none", mew=0.6, label=labels.get(mode, mode))
    ax.set_xlim(0, d["lam"].max() * 1.08)
    ax.set_ylim(*pad_lin(d["q_max"]))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\max_k|Q_k|$ in p.u.")
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(DE_FMT)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 2)))
    style_axes(ax)
    ax.legend(loc="upper left")
    return fig


def fig_opt(b, cfg):
    df = b.get("warm")
    if df is None:
        return None
    d = df.copy()
    if "n" in d:
        d = d[d["n"].astype(float) == cfg["n_ref"]]
    if "pv_share" in d:
        d = d[np.isclose(d["pv_share"].astype(float), cfg["pv_ref"])]
    d = d.dropna(subset=["lam"]).sort_values("lam")
    variants = [("k_cold_fix", r"kalt, fixe Toleranz"),
                ("k_warm_fix", r"Warm Start, fixe Toleranz"),
                ("k_warm_adapt", r"Warm Start $+$ adaptive Toleranz")]
    avail = [(c, lbl) for c, lbl in variants if c in d and d[c].notna().any()]
    if not avail:
        return None
    fig, ax = plt.subplots(figsize=figsize_cm(0.85 * TEXTWIDTH_CM, 6.6))
    ax.set_yscale("log")
    vals = []
    for i, (col, lbl) in enumerate(avail):
        s = d.dropna(subset=[col])
        vals.append(s[col].to_numpy(float))
        ax.plot(s["lam"], s[col], color=CYCLE[i % len(CYCLE)],
                ls=LINESTYLES[i % len(LINESTYLES)], marker=MARKERS[i % len(MARKERS)],
                mfc="none", mew=0.6, label=lbl)
    ax.set_xlim(0, d["lam"].max() * 1.08)
    ax.set_ylim(*pad_log(np.concatenate(vals)))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$k_{\mathrm{in}}$ (gesamt)")
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.xaxis.set_major_formatter(DE_FMT)
    ax.yaxis.set_major_locator(FixedLocator(nice_log_ticks(*ax.get_ylim())))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(DE_FMT)
    style_axes(ax)
    ax.legend(loc="upper left")
    return fig


def fig_damping(b, cfg):
    df = b.get("damping")
    if df is None:
        return None
    d = df.dropna(subset=["omega", "lam_star"])
    agg = (d.groupby(["n", "omega"])["lam_star"].median().reset_index()
           .sort_values(["n", "omega"]))
    if agg.empty:
        return None
    fig, ax = plt.subplots(figsize=figsize_cm(0.48 * TEXTWIDTH_CM, 5.6))
    for i, (n, sub) in enumerate(agg.groupby("n")):
        ax.plot(sub["omega"], sub["lam_star"], color=CYCLE[i % len(CYCLE)],
                ls=LINESTYLES[i % len(LINESTYLES)], marker=MARKERS[i % len(MARKERS)],
                mfc="none", mew=0.6, label=rf"$n={int(n)}$")
    lo, hi = agg["lam_star"].min(), agg["lam_star"].max()
    pad = 0.12 * max(hi - lo, 0.1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(agg["omega"].min() - 0.06, agg["omega"].max() + 0.06)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\lambda^\ast_{\mathrm{TPF}}$")
    ax.xaxis.set_major_locator(FixedLocator(sorted(agg["omega"].unique())))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 1)))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 1)))
    style_axes(ax)
    ax.legend(loc="lower left")
    return fig


FIGURES = {
    "eta": ("lam_eta", fig_eta),
    "kin": ("lam_kin", fig_kin),
    "kout_pv": ("lam_kout_pv", fig_kout_pv),
    "kout_n": ("lam_kout_n", fig_kout_n),
    "q": ("lam_q", fig_q),
    "opt": ("lam_opt", fig_opt),
    "damping": ("lam_damping", fig_damping),
}


# ======================================================================
# 6  main
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", type=Path, default=Path("out"),
                    help="Verzeichnis mit den CSV-Exporten von lambda_sweep.py")
    ap.add_argument("--md", type=Path, default=Path("results_lastfaktor.md"),
                    help="Markdown-Report als Rückfallquelle")
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    ap.add_argument("--n-ref", type=float, default=40, help="Referenznetzgröße")
    ap.add_argument("--pv-ref", type=float, default=0.5,
                    help="Referenz-PV-Anteil für lam_kout_n")
    ap.add_argument("--pv-q", type=float, default=0.1,
                    help="PV-Anteil für lam_q und lam_opt")
    ap.add_argument("--pdf", action="store_true", help="zusätzlich PDF-Preview")
    args = ap.parse_args()

    bundle = build_bundle(args.csv_dir, args.md)
    if not bundle:
        sys.exit("Keine verwertbaren Daten gefunden.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    cfg_main = {"n_ref": args.n_ref, "pv_ref": args.pv_ref}
    cfg_low = {"n_ref": args.n_ref, "pv_ref": args.pv_q}
    per_fig_cfg = {"q": cfg_low, "opt": cfg_low, "kout_pv": cfg_low}

    for key in (args.only or sorted(FIGURES)):
        name, builder = FIGURES[key]
        fig = builder(bundle, per_fig_cfg.get(key, cfg_main))
        if fig is None:
            print(f"  übersprungen: {name} (Daten fehlen)", file=sys.stderr)
            continue
        fig.savefig(args.outdir / f"{name}.pgf")
        if args.pdf:
            fig.savefig(args.outdir / f"{name}.pdf")
        plt.close(fig)
        print(f"geschrieben: {args.outdir / (name + '.pgf')}")


if __name__ == "__main__":
    main()
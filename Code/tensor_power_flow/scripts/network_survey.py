# tensor_power_flow/scripts/network_survey.py
"""
Netzwerk-Übersicht: Netzdaten + Topologie + Kontraktionsfaktor η
=================================================================
Berechnet für viele pandapower-Testnetze (klein → groß):
- Busanzahl, Branches, Slack/PV/PQ
- Netztyp (Verteilnetz / Übertragungsnetz)
- Topologie (radial / schwach vermascht / vermascht)
- Vermaschungsgrad μ = (n_branches - n_buses + 1) / n_branches
- R/X-Verhältnisse
- Kontraktionsfaktor η
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ══════════════════════════════════════════════════════════════════════
#  Netztyp-Klassifikation
# ══════════════════════════════════════════════════════════════════════

def classify_voltage_level(vn_kv_values):
    """
    Klassifiziert das Netz anhand der Spannungsebenen.

    Returns
    -------
    str: "NS" (Niederspannung), "MS" (Mittelspannung),
         "HS" (Hochspannung), "HöS" (Höchstspannung), "Gemischt"
    """
    v_max = np.max(vn_kv_values)
    v_min = np.min(vn_kv_values)

    if v_max <= 1.0:
        return "NS"        # Niederspannung (≤1 kV)
    elif v_max <= 35.0:
        return "MS"        # Mittelspannung (1-35 kV)
    elif v_max <= 110.0:
        if v_min < 35.0:
            return "MS/HS"
        return "HS"        # Hochspannung (35-110 kV)
    elif v_max <= 230.0:
        return "HS/HöS"   # Hoch-/Höchstspannung
    else:
        return "HöS"       # Höchstspannung (>230 kV)


def classify_network_type(vn_kv_values, rx_mean, n_pv, n_bus):
    """
    Klassifiziert: Verteilnetz (VN) oder Übertragungsnetz (ÜN).

    Kriterien:
    - Spannung: ≤110 kV → tendenziell VN; >110 kV → ÜN
    - R/X: hoch (>0.5) → typisch VN; niedrig (<0.5) → typisch ÜN
    - PV-Anteil: hoch → tendenziell ÜN
    """
    v_max = np.max(vn_kv_values)
    pv_ratio = n_pv / max(n_bus, 1)

    if v_max > 110.0:
        return "ÜN"
    elif v_max <= 35.0 and rx_mean > 0.5:
        return "VN"
    elif v_max <= 110.0 and pv_ratio < 0.05 and rx_mean > 0.3:
        return "VN"
    elif v_max <= 110.0 and pv_ratio > 0.1:
        return "ÜN"
    else:
        return "VN/ÜN"


def classify_topology(n_buses_total, n_branches, n_connected_components=1):
    """
    Klassifiziert die Netztopologie.

    μ = (n_branches - n_buses + n_components) / n_branches
      = Anteil der "überschüssigen" Leitungen (Maschen)

    μ = 0     → radial (Baum)
    μ < 0.1   → schwach vermascht
    μ < 0.3   → mäßig vermascht
    μ ≥ 0.3   → stark vermascht

    Returns
    -------
    topology_class : str
    n_loops : int (Anzahl unabhängiger Maschen)
    mesh_degree : float (μ)
    """
    n_loops = n_branches - n_buses_total + n_connected_components

    if n_branches == 0:
        return "unbekannt", 0, 0.0

    mesh_degree = n_loops / n_branches

    if n_loops == 0:
        topo_class = "radial"
    elif mesh_degree < 0.1:
        topo_class = "schwach vermascht"
    elif mesh_degree < 0.3:
        topo_class = "mäßig vermascht"
    else:
        topo_class = "stark vermascht"

    return topo_class, n_loops, mesh_degree


# ══════════════════════════════════════════════════════════════════════
#  Kontraktionsfaktor
# ══════════════════════════════════════════════════════════════════════

def compute_eta(Y_dd, s_nom, v_min, alpha_p=None):
    """
    η = ||Z_B · diag(α_P ⊙ s*)||₁ / v_min²
    """
    n = Y_dd.shape[0]
    if n == 0 or v_min < 1e-6:
        return np.inf

    if alpha_p is None:
        alpha_p = np.ones(n)

    try:
        Z_B = np.linalg.inv(Y_dd)
    except np.linalg.LinAlgError:
        return np.inf

    scaling = alpha_p * np.conj(s_nom)
    M = Z_B * scaling.reshape(1, -1)
    matrix_1_norm = np.max(np.sum(np.abs(M), axis=0))
    eta = matrix_1_norm / (v_min ** 2)
    return eta


# ══════════════════════════════════════════════════════════════════════
#  Netzanalyse
# ══════════════════════════════════════════════════════════════════════

def analyze_network(net, name):
    """Analysiert ein pandapower-Netz vollständig."""
    result = {
        "name": name,
        "n_bus": 0,
        "n_branch": 0,
        "n_slack": 0,
        "n_pv": 0,
        "n_pq": 0,
        "base_mva": 0,
        "p_load_mw": 0,
        "q_load_mvar": 0,
        "p_gen_mw": 0,
        "rx_min": 0,
        "rx_mean": 0,
        "rx_max": 0,
        "v_min": 0,
        "v_max": 0,
        "vn_max_kv": 0,
        "vn_min_kv": 0,
        "nr_iter": -1,
        "nr_converged": False,
        "eta_pq": np.inf,
        "eta_all": np.inf,
        "nr_time_ms": 0,
        "d_block_size": 0,
        # Neu:
        "voltage_level": "",
        "net_type": "",
        "topology": "",
        "n_loops": 0,
        "mesh_degree": 0.0,
        "is_radial": False,
        "error": None,
    }

    try:
        # ── NR lösen ──
        t0 = time.perf_counter()
        pp.runpp(net, algorithm="nr", tolerance_mva=1e-8, max_iteration=100)
        t_nr = (time.perf_counter() - t0) * 1000

        if not net.converged:
            result["error"] = "NR diverged"
            return result

        result["nr_time_ms"] = t_nr
        result["nr_converged"] = True

        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        base_mva = ppc["baseMVA"]

        slack_idx = np.where(bus_types == 3)[0]
        pv_idx = np.where(bus_types == 2)[0]
        pq_idx = np.where(bus_types == 1)[0]
        d_idx = np.sort(np.concatenate([pq_idx, pv_idx]))

        n_bus_total = len(bus_types)
        n_branches = ppc["branch"].shape[0]

        result["n_bus"] = n_bus_total
        result["n_branch"] = n_branches
        result["n_slack"] = len(slack_idx)
        result["n_pv"] = len(pv_idx)
        result["n_pq"] = len(pq_idx)
        result["base_mva"] = base_mva
        result["d_block_size"] = len(d_idx)

        # ── Spannungsebenen ──
        vn_kv = net.bus["vn_kv"].values
        result["vn_max_kv"] = float(np.max(vn_kv))
        result["vn_min_kv"] = float(np.min(vn_kv))

        # ── Netztyp-Klassifikation ──
        r = ppc["branch"][:, 2]
        x = ppc["branch"][:, 3]
        x_safe = np.where(np.abs(x) > 1e-12, x, 1e-12)
        rx = np.abs(r / x_safe)
        result["rx_min"] = float(np.min(rx)) if len(rx) > 0 else 0
        result["rx_mean"] = float(np.mean(rx)) if len(rx) > 0 else 0
        result["rx_max"] = float(np.max(rx)) if len(rx) > 0 else 0

        result["voltage_level"] = classify_voltage_level(vn_kv)
        result["net_type"] = classify_network_type(
            vn_kv, result["rx_mean"], len(pv_idx), n_bus_total
        )

        # ── Topologie ──
        topo_class, n_loops, mesh_degree = classify_topology(
            n_bus_total, n_branches
        )
        result["topology"] = topo_class
        result["n_loops"] = n_loops
        result["mesh_degree"] = mesh_degree
        result["is_radial"] = (n_loops == 0)

        # ── Last & Erzeugung ──
        result["p_load_mw"] = np.sum(ppc["bus"][:, 2])
        result["q_load_mvar"] = np.sum(ppc["bus"][:, 3])
        result["p_gen_mw"] = np.sum(ppc["gen"][:, 1])

        # ── Spannungen ──
        vm = ppc["bus"][:, 7]
        result["v_min"] = float(np.min(vm[d_idx])) if len(d_idx) > 0 else 1.0
        result["v_max"] = float(np.max(vm[d_idx])) if len(d_idx) > 0 else 1.0

        # ── NR Iterationen ──
        result["nr_iter"] = ppc.get("iterations", -1)

        # ── Y-Bus aufbauen ──
        from pandapower.pypower.makeYbus import makeYbus
        Y_bus, _, _ = makeYbus(base_mva, ppc["bus"], ppc["branch"])
        Y_bus = Y_bus.toarray()

        # ── η für PQ-only ──
        if len(pq_idx) > 0:
            Y_dd_pq = Y_bus[np.ix_(pq_idx, pq_idx)]
            gen_buses = ppc["gen"][:, 0].astype(int)
            s_pq = np.zeros(len(pq_idx), dtype=np.complex128)
            for i, bus in enumerate(pq_idx):
                p_load = ppc["bus"][bus, 2] / base_mva
                q_load = ppc["bus"][bus, 3] / base_mva
                gm = gen_buses == bus
                p_gen = np.sum(ppc["gen"][gm, 1]) / base_mva if np.any(gm) else 0
                q_gen = np.sum(ppc["gen"][gm, 2]) / base_mva if np.any(gm) else 0
                s_pq[i] = (p_load - p_gen) + 1j * (q_load - q_gen)
            v_min_pq = float(np.min(vm[pq_idx]))
            result["eta_pq"] = compute_eta(Y_dd_pq, s_pq, v_min_pq)

        # ── η für PQ+PV ──
        if len(d_idx) > 0:
            Y_dd_all = Y_bus[np.ix_(d_idx, d_idx)]
            gen_buses = ppc["gen"][:, 0].astype(int)
            s_all = np.zeros(len(d_idx), dtype=np.complex128)
            for i, bus in enumerate(d_idx):
                p_load = ppc["bus"][bus, 2] / base_mva
                q_load = ppc["bus"][bus, 3] / base_mva
                gm = gen_buses == bus
                p_gen = np.sum(ppc["gen"][gm, 1]) / base_mva if np.any(gm) else 0
                q_gen = np.sum(ppc["gen"][gm, 2]) / base_mva if np.any(gm) else 0
                s_all[i] = (p_load - p_gen) + 1j * (q_load - q_gen)
            v_min_all = float(np.min(vm[d_idx]))
            result["eta_all"] = compute_eta(Y_dd_all, s_all, v_min_all)

    except Exception as e:
        result["error"] = str(e)[:60]

    return result


# ══════════════════════════════════════════════════════════════════════
#  Testnetz-Katalog
# ══════════════════════════════════════════════════════════════════════

def get_test_networks():
    """Geordnete Liste von (Name, Constructor), klein → groß."""
    networks = []

    # Kleine Netze (< 10 Busse)
    networks.append(("case4gs", lambda: pn.case4gs()))
    networks.append(("case5", lambda: pn.case5()))
    networks.append(("case6ww", lambda: pn.case6ww()))
    networks.append(("case9", lambda: pn.case9()))

    # Mittelgroße (10-50 Busse)
    networks.append(("case14", lambda: pn.case14()))
    networks.append(("case_ieee30", lambda: pn.case_ieee30()))
    networks.append(("case33bw", lambda: pn.case33bw()))
    networks.append(("case39", lambda: pn.case39()))

    # Groß (50-300 Busse)
    networks.append(("case57", lambda: pn.case57()))
    networks.append(("case_ieee118", lambda: pn.case_ieee118()))

    try:
        networks.append(("GBnetwork", lambda: pn.GBnetwork()))
    except:
        pass

    networks.append(("case300", lambda: pn.case300()))

    # Sehr groß (>300)
    try:
        networks.append(("case1354pegase", lambda: pn.case1354pegase()))
    except:
        pass
    try:
        networks.append(("case1888rte", lambda: pn.case1888rte()))
    except:
        pass
    try:
        networks.append(("case2848rte", lambda: pn.case2848rte()))
    except:
        pass
    try:
        networks.append(("case2869pegase", lambda: pn.case2869pegase()))
    except:
        pass
    try:
        networks.append(("case9241pegase", lambda: pn.case9241pegase()))
    except:
        pass

    return networks


# ══════════════════════════════════════════════════════════════════════
#  Hauptprogramm
# ══════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════════════════════════╗")
    print("║  NETZWERK-ÜBERSICHT: Topologie, Netztyp, Kontraktionsfaktor η                   ║")
    print("║  η < 1: FPI konvergiert garantiert | η ≥ 1: Konvergenz nicht garantiert          ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════════╝")

    networks = get_test_networks()
    results = []

    print(f"\n  Analysiere {len(networks)} Netze...\n")

    for name, constructor in networks:
        try:
            net = constructor()
            r = analyze_network(net, name)
            results.append(r)
            status = "✓" if r["nr_converged"] else "✗"
            print(f"  {status} {name:<20} ({r['n_bus']} Busse, {r['topology']})")
        except Exception as e:
            print(f"  ✗ {name:<20} FEHLER: {e}")
            results.append({"name": name, "error": str(e)[:40],
                           "n_bus": 0, "nr_converged": False})

    # ══════════════════════════════════════════════════════════════
    #  Tabelle 1: Netzstruktur & Topologie
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*130}")
    print(f"  TABELLE 1: Netzstruktur, Typ & Topologie")
    print(f"{'═'*130}")
    print(f"  {'Netz':<18} {'n_bus':<6} {'n_br':<6} {'Sl':<4} {'PV':<5} "
          f"{'PQ':<5} {'U [kV]':<10} {'Ebene':<7} {'Typ':<6} "
          f"{'Topologie':<20} {'Maschen':<8} {'μ':<6} {'Radial'}")
    print(f"  {'─'*128}")

    for r in results:
        if r.get("error") and r["n_bus"] == 0:
            print(f"  {r['name']:<18} — FEHLER: {r.get('error','?')}")
            continue
        if not r.get("nr_converged"):
            print(f"  {r['name']:<18} — NR nicht konvergiert")
            continue

        vn_str = f"{r.get('vn_min_kv',0):.0f}-{r.get('vn_max_kv',0):.0f}"
        radial_str = "✓" if r.get("is_radial") else "✗"

        print(f"  {r['name']:<18} {r['n_bus']:<6} {r['n_branch']:<6} "
              f"{r['n_slack']:<4} {r['n_pv']:<5} {r['n_pq']:<5} "
              f"{vn_str:<10} {r.get('voltage_level','?'):<7} "
              f"{r.get('net_type','?'):<6} "
              f"{r.get('topology','?'):<20} {r.get('n_loops',0):<8} "
              f"{r.get('mesh_degree',0):<6.3f} {radial_str}")

    # ══════════════════════════════════════════════════════════════
    #  Tabelle 2: R/X und Spannungsprofil
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*115}")
    print(f"  TABELLE 2: Leitungsparameter & Spannungsprofil")
    print(f"{'═'*115}")
    print(f"  {'Netz':<18} {'Typ':<6} {'Topo':<12} {'R/X min':<9} "
          f"{'R/X mean':<10} {'R/X max':<9} {'V_min':<8} {'V_max':<8} "
          f"{'NR It':<6} {'NR t[ms]':<10}")
    print(f"  {'─'*113}")

    for r in results:
        if r.get("error") and r["n_bus"] == 0:
            continue
        if not r.get("nr_converged"):
            continue
        topo_short = r.get("topology", "?")[:10]
        print(f"  {r['name']:<18} {r.get('net_type','?'):<6} "
              f"{topo_short:<12} {r['rx_min']:<9.4f} "
              f"{r['rx_mean']:<10.4f} {r['rx_max']:<9.4f} "
              f"{r['v_min']:<8.4f} {r['v_max']:<8.4f} "
              f"{r['nr_iter']:<6} {r['nr_time_ms']:<10.1f}")

    # ══════════════════════════════════════════════════════════════
    #  Tabelle 3: Kontraktionsfaktor η
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*120}")
    print(f"  TABELLE 3: Kontraktionsfaktor η + Netztypologie")
    print(f"{'═'*120}")
    print(f"  η < 1.0 → FPI konvergiert garantiert (Banach)")
    print(f"  η ≥ 1.0 → Konvergenz NICHT garantiert")
    print(f"{'─'*120}")
    print(f"  {'Netz':<18} {'n_bus':<6} {'Typ':<6} {'Topo':<14} {'Radial':<7} "
          f"{'n_PV':<5} {'η (PQ)':<12} {'η (PQ+PV)':<12} "
          f"{'Stat PQ':<10} {'Stat All':<10} {'V_min':<7} {'R/X':<6}")
    print(f"  {'─'*118}")

    for r in results:
        if r.get("error") and r["n_bus"] == 0:
            continue
        if not r.get("nr_converged"):
            continue

        eta_pq = r["eta_pq"]
        eta_all = r["eta_all"]

        def status(eta):
            if eta < 0.5:
                return "✓ sicher"
            elif eta < 1.0:
                return "⚠ knapp"
            elif eta < 2.0:
                return "✗ >1"
            else:
                return "✗✗ >>1"

        eta_pq_str = f"{eta_pq:.4f}" if eta_pq < 100 else f"{eta_pq:.1f}"
        eta_all_str = f"{eta_all:.4f}" if eta_all < 100 else f"{eta_all:.1f}"
        topo_short = r.get("topology", "?")[:12]
        radial_str = "ja" if r.get("is_radial") else "nein"
        rx_str = f"{r['rx_mean']:.2f}"

        print(f"  {r['name']:<18} {r['n_bus']:<6} "
              f"{r.get('net_type','?'):<6} {topo_short:<14} "
              f"{radial_str:<7} {r['n_pv']:<5} "
              f"{eta_pq_str:<12} {eta_all_str:<12} "
              f"{status(eta_pq):<10} {status(eta_all):<10} "
              f"{r['v_min']:<7.4f} {rx_str:<6}")

    # ══════════════════════════════════════════════════════════════
    #  Zusammenfassung
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'═'*90}")
    print(f"  ZUSAMMENFASSUNG & INTERPRETATION")
    print(f"{'═'*90}")

    converged = [r for r in results if r.get("nr_converged")]
    if converged:
        # Statistik
        vn_nets = [r for r in converged if r.get("net_type") == "VN"]
        un_nets = [r for r in converged if r.get("net_type") == "ÜN"]
        radial_nets = [r for r in converged if r.get("is_radial")]
        meshed_nets = [r for r in converged if not r.get("is_radial")]

        eta_pq_vals = [r["eta_pq"] for r in converged if r["eta_pq"] < np.inf]
        eta_all_vals = [r["eta_all"] for r in converged if r["eta_all"] < np.inf]

        n_eta_pq_ok = sum(1 for e in eta_pq_vals if e < 1.0)
        n_eta_all_ok = sum(1 for e in eta_all_vals if e < 1.0)

        print(f"\n  Analysierte Netze:       {len(results)}")
        print(f"  NR konvergiert:          {len(converged)}")
        print(f"  Verteilnetze (VN):       {len(vn_nets)}")
        print(f"  Übertragungsnetze (ÜN):  {len(un_nets)}")
        print(f"  Radialnetze:             {len(radial_nets)}")
        print(f"  Vermaschte Netze:        {len(meshed_nets)}")
        print(f"  η(PQ) < 1:              {n_eta_pq_ok}/{len(eta_pq_vals)}")
        print(f"  η(PQ+PV) < 1:           {n_eta_all_ok}/{len(eta_all_vals)}")

        # η nach Netztyp
        if vn_nets:
            eta_vn = [r["eta_pq"] for r in vn_nets if r["eta_pq"] < np.inf]
            if eta_vn:
                print(f"\n  η(PQ) Verteilnetze:   min={min(eta_vn):.4f}, "
                      f"max={max(eta_vn):.4f}, median={np.median(eta_vn):.4f}")
        if un_nets:
            eta_un = [r["eta_pq"] for r in un_nets if r["eta_pq"] < np.inf]
            if eta_un:
                print(f"  η(PQ) Übertragung:    min={min(eta_un):.4f}, "
                      f"max={max(eta_un):.4f}, median={np.median(eta_un):.4f}")

        # η nach Topologie
        if radial_nets:
            eta_rad = [r["eta_pq"] for r in radial_nets if r["eta_pq"] < np.inf]
            if eta_rad:
                print(f"\n  η(PQ) radiale Netze:  min={min(eta_rad):.4f}, "
                      f"max={max(eta_rad):.4f}, median={np.median(eta_rad):.4f}")
        if meshed_nets:
            eta_mesh = [r["eta_pq"] for r in meshed_nets if r["eta_pq"] < np.inf]
            if eta_mesh:
                print(f"  η(PQ) vermaschte:     min={min(eta_mesh):.4f}, "
                      f"max={max(eta_mesh):.4f}, median={np.median(eta_mesh):.4f}")

    # ── Interpretation für die BA ──
    print(f"\n  {'─'*70}")
    print(f"  INTERPRETATION FÜR DEINE BACHELORARBEIT:")
    print(f"  {'─'*70}")
    print(f"""
  • VERTEILNETZE (radial, hohes R/X):
    → η << 1 → TPF konvergiert schnell & sicher
    → Ideales Einsatzgebiet des TPF
    → PV-Integration (Methode B) funktioniert problemlos

  • ÜBERTRAGUNGSNETZE (vermascht, niedriges R/X, viele PV):
    → η ≥ 1 möglich → Standard-FPI divergiert
    → Q-Dämpfung (ω < 1) in Methode B erforderlich
    → NICHT das primäre Einsatzgebiet des TPF

  • TOPOLOGIE-EFFEKT:
    → Radiale Netze: η tendenziell niedriger (weniger Kopplung)
    → Vermaschung erhöht Kopplung → kann η senken ODER erhöhen
      (abhängig von der Lastverteilung)

  • PV-KNOTEN-EFFEKT:
    → η(PQ+PV) > η(PQ only), da d-Block größer wird
    → Mehr PV-Knoten → größerer d-Block → stärkere Kopplung
""")
    print(f"{'═'*90}")


if __name__ == "__main__":
    main()
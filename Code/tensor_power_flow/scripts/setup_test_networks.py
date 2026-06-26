"""
Testnetze für die PV-Knoten-Validierung
=========================================
Definiert verschiedene Szenarien mit bekannter Anzahl PV-Knoten.
"""
import numpy as np
import pandapower as pp
import pandapower.networks as pn
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tpf.solvers.nr_reference import PandapowerNRSolver


def create_4bus_1pv():
    """
    Minimales Testnetz: 4 Busse, 1 Slack, 1 PV, 2 PQ.

        Slack (Bus 0) ---Line--- PV (Bus 1) ---Line--- PQ (Bus 2)
                                     |
                                   Line
                                     |
                                 PQ (Bus 3)
    """
    net = pp.create_empty_network()

    # Busse
    b0 = pp.create_bus(net, vn_kv=20.0, name="Slack")
    b1 = pp.create_bus(net, vn_kv=20.0, name="PV-Bus")
    b2 = pp.create_bus(net, vn_kv=20.0, name="PQ-Bus 1")
    b3 = pp.create_bus(net, vn_kv=20.0, name="PQ-Bus 2")

    # Slack
    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid")

    # PV-Knoten (Generator mit fester Spannung und P)
    pp.create_gen(net, bus=b1, p_mw=5.0, vm_pu=1.03, name="Gen PV")

    # Lasten
    pp.create_load(net, bus=b1, p_mw=2.0, q_mvar=0.5, name="Load 1")
    pp.create_load(net, bus=b2, p_mw=4.0, q_mvar=1.5, name="Load 2")
    pp.create_load(net, bus=b3, p_mw=3.0, q_mvar=1.0, name="Load 3")

    # Leitungen
    pp.create_line_from_parameters(
        net, b0, b1, length_km=5, r_ohm_per_km=0.2,
        x_ohm_per_km=0.4, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_line_from_parameters(
        net, b1, b2, length_km=8, r_ohm_per_km=0.3,
        x_ohm_per_km=0.5, c_nf_per_km=0, max_i_ka=1
    )
    pp.create_line_from_parameters(
        net, b1, b3, length_km=6, r_ohm_per_km=0.25,
        x_ohm_per_km=0.45, c_nf_per_km=0, max_i_ka=1
    )

    return net


def create_33bus_with_dg():
    """
    IEEE 33-Bus Radiales Verteilnetz + 2 DG-Einspeiser als PV-Knoten.
    DG an Bus 12 und Bus 24 (typische DG-Platzierung).
    """
    net = pn.case33bw()

    # DG 1 an Bus 12: 0.5 MW, |V| = 1.02 p.u.
    pp.create_gen(net, bus=12, p_mw=0.5, vm_pu=1.02, name="DG1")

    # DG 2 an Bus 24: 0.4 MW, |V| = 1.01 p.u.
    pp.create_gen(net, bus=24, p_mw=0.4, vm_pu=1.01, name="DG2")

    return net


def get_ieee30():
    """
    IEEE 30-Bus: Standard-Testnetz mit 6 PV-Knoten.
    (2 Generatoren + 4 Synchronkompensatoren)
    """
    return pn.case_ieee30()


def get_ieee57():
    """IEEE 57-Bus: 7 PV-Knoten."""
    return pn.case57()


# ══════════════════════════════════════════════════════════════════════
#  Übersicht aller Testnetze
# ══════════════════════════════════════════════════════════════════════

TEST_NETWORKS = {
    "4bus_1pv": {
        "constructor": create_4bus_1pv,
        "description": "4 Busse, 1 PV, 2 PQ (minimal)",
        "n_pv": 1,
    },
    "33bus_2dg": {
        "constructor": create_33bus_with_dg,
        "description": "IEEE 33 + 2 DG (PV-Knoten)",
        "n_pv": 2,
    },
    "ieee30": {
        "constructor": get_ieee30,
        "description": "IEEE 30-Bus (6 PV-Knoten)",
        "n_pv": 5,  # Bus 2,5,8,11,13 sind PV
    },
    "ieee57": {
        "constructor": get_ieee57,
        "description": "IEEE 57-Bus (7 PV-Knoten)",
        "n_pv": 6,
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Referenzlösungen generieren und anzeigen
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  TESTNETZE MIT PV-KNOTEN — Referenzlösungen (pandapower NR)")
    print("=" * 70)

    solver = PandapowerNRSolver(tol=1e-8, max_iter=100)

    for name, info in TEST_NETWORKS.items():
        print(f"\n{'─'*70}")
        print(f"  Netz: {name} — {info['description']}")
        print(f"{'─'*70}")

        net = info["constructor"]()
        result = solver.solve_from_net(net)

        print(f"  Konvergiert: {result.converged}")
        print(f"  Iterationen: {result.iterations}")
        print(f"  Zeit: {result.elapsed_time_s*1000:.2f} ms")

        # PPC-Bustypen anzeigen
        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        n_slack = np.sum(bus_types == 3)
        n_pv = np.sum(bus_types == 2)
        n_pq = np.sum(bus_types == 1)
        print(f"  Bustypen: Slack={n_slack}, PV={n_pv}, PQ={n_pq}")

        # PV-Ergebnisse
        if result.pv_indices is not None:
            print(f"\n  PV-Knoten Referenzergebnisse:")
            print(f"  {'Bus':<6} {'|V| soll':<10} {'|V| ist':<10} {'Q (p.u.)':<10}")
            print(f"  {'─'*40}")
            for i, idx in enumerate(result.pv_indices):
                v_actual = np.abs(result.voltages[idx])
                v_set = result.pv_v_setpoint_pu[i]
                q = result.pv_q_pu[i]
                print(f"  {idx:<6} {v_set:<10.4f} {v_actual:<10.6f} {q:<10.6f}")

            # Prüfe: |V_ist| == |V_soll|?
            v_error = np.abs(
                np.abs(result.voltages[result.pv_indices]) - result.pv_v_setpoint_pu
            )
            print(f"\n  Max |V|-Fehler an PV-Knoten: {np.max(v_error):.2e}")
            assert np.max(v_error) < 1e-6, "PV-Spannung nicht eingehalten!"
            print(f"    PV-Spannungen korrekt eingehalten")
        else:
            print(f"  (Keine PV-Knoten gefunden)")

        # Alle Spannungen (kompakt)
        print(f"\n  Spannungsbeträge (Min/Max):")
        v_mag = np.abs(result.voltages)
        print(f"    Min |V| = {np.min(v_mag):.6f} p.u. (Bus {np.argmin(v_mag)})")
        print(f"    Max |V| = {np.max(v_mag):.6f} p.u. (Bus {np.argmax(v_mag)})")

    print(f"\n{'='*70}")
    print("  ALLE REFERENZLÖSUNGEN ERFOLGREICH BERECHNET")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
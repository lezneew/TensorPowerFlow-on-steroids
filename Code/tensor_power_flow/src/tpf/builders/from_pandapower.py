# builders/from_pandapower.py — korrigierte Version
import numpy as np
import pandapower as pp
from tpf.core.network import NetworkData


def build_network_from_pandapower(net: pp.pandapowerNet) -> NetworkData:
    """
    Konvertiert ein pandapower-Netz ins interne TPF-Format.

    ACHTUNG: Aktuell werden nur Slack- und PQ-Knoten unterstützt.
    PV-Knoten im Netz werden ignoriert / müssen vorher entfernt werden.
    """
    # Vollständigen Power Flow laufen lassen, um _ppc korrekt aufzubauen
    pp.runpp(net, algorithm="nr", tolerance_mva=1e-8)

    ppc = net._ppc

    # Admittanzmatrix aus PYPOWER extrahieren
    from pandapower.pypower.makeYbus import makeYbus
    Y_bus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    Y_bus = Y_bus.toarray()  # dense für TPF

    # Knotentypen identifizieren (PYPOWER-Konvention)
    # 1=PQ, 2=PV, 3=Ref/Slack
    bus_types = ppc["bus"][:, 1].astype(int)
    slack_idx = np.where(bus_types == 3)[0]
    pq_idx = np.where(bus_types == 1)[0]

    if len(pq_idx) == 0:
        raise ValueError("Keine PQ-Knoten im Netz gefunden.")

    # Admittanzmatrix partitionieren
    Y_dd = Y_bus[np.ix_(pq_idx, pq_idx)]
    Y_ds = Y_bus[np.ix_(pq_idx, slack_idx)]

    # Slack-Spannung (Betrag × Phase)
    v_s = (ppc["bus"][slack_idx, 7]
           * np.exp(1j * np.deg2rad(ppc["bus"][slack_idx, 8])))

    # Knotenleistungen der PQ-Knoten (in p.u., Verbraucherkonvention)
    # ppc["bus"][:, 2] = Pd (MW), ppc["bus"][:, 3] = Qd (MVAr)
    base_mva = ppc["baseMVA"]
    s_nom = (ppc["bus"][pq_idx, 2] + 1j * ppc["bus"][pq_idx, 3]) / base_mva

    return NetworkData(
        Y_dd=Y_dd,
        Y_ds=Y_ds,
        v_s=v_s,
        s_nom=s_nom,
        alpha_p=np.ones(len(pq_idx)),
        alpha_i=np.zeros(len(pq_idx)),
        alpha_z=np.zeros(len(pq_idx)),
        n_buses=len(pq_idx),
        n_phases=1,
        bus_names=[f"bus_{i}" for i in pq_idx],
    )


def get_pq_indices_from_net(net: pp.pandapowerNet) -> np.ndarray:
    """Gibt die internen PPC-Indizes der PQ-Knoten zurück."""
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    return np.where(bus_types == 1)[0]
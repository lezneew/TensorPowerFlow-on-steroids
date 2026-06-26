import numpy as np
import pandapower as pp
from tpf.core.network import NetworkData


def build_network_from_pandapower(
    net: pp.pandapowerNet,
    include_pv: bool = False,
) -> NetworkData:
    """
    Konvertiert ein pandapower-Netz ins interne TPF-Format.

    Parameters
    ----------
    net : pandapowerNet
    include_pv : bool
        False (default): Nur PQ-Knoten (original-Verhalten)
        True: PQ + PV-Knoten im d-Block (für PV-Erweiterung)
    """
    pp.runpp(net, algorithm="nr", tolerance_mva=1e-8)
    ppc = net._ppc

    from pandapower.pypower.makeYbus import makeYbus
    Y_bus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    Y_bus = Y_bus.toarray()

    bus_types = ppc["bus"][:, 1].astype(int)
    slack_idx = np.where(bus_types == 3)[0]
    pq_idx = np.where(bus_types == 1)[0]
    pv_idx = np.where(bus_types == 2)[0]

    if include_pv:
        # d-Block = PQ ∪ PV (alles außer Slack)
        d_idx = np.sort(np.concatenate([pq_idx, pv_idx]))
    else:
        d_idx = pq_idx

    if len(d_idx) == 0:
        raise ValueError("Keine Lastknoten im Netz gefunden.")

    # Admittanzmatrix partitionieren
    Y_dd = Y_bus[np.ix_(d_idx, d_idx)]
    Y_ds = Y_bus[np.ix_(d_idx, slack_idx)]

    # Slack-Spannung
    v_s = (ppc["bus"][slack_idx, 7]
           * np.exp(1j * np.deg2rad(ppc["bus"][slack_idx, 8])))

    # Knotenleistungen (Netto = Pg - Pd, Qg - Qd)
    base_mva = ppc["baseMVA"]

    # Leistungsinjektion: P_inject = Pg - Pd (Erzeuger-Konvention)
    # Für TPF brauchen wir Verbraucherkonvention: s_nom = Pd - Pg + j(Qd - Qg)
    # ABER: s_nom im PPC ist bereits als "Last" angegeben (Pd, Qd positiv = Verbrauch)
    # Generatoren stehen separat in ppc["gen"]

    s_nom = np.zeros(len(d_idx), dtype=np.complex128)
    gen_buses = ppc["gen"][:, 0].astype(int)

    for i, bus in enumerate(d_idx):
        # Lastleistung (positiv = Verbrauch)
        p_load = ppc["bus"][bus, 2] / base_mva
        q_load = ppc["bus"][bus, 3] / base_mva

        # Generatorleistung an diesem Bus
        gen_mask = gen_buses == bus
        p_gen = np.sum(ppc["gen"][gen_mask, 1]) / base_mva
        q_gen = np.sum(ppc["gen"][gen_mask, 2]) / base_mva

        # Netto-Last (Verbraucherkonvention für TPF)
        # s_nom = P_load - P_gen + j*(Q_load - Q_gen)
        s_nom[i] = (p_load - p_gen) + 1j * (q_load - q_gen)

    # PV-Maske (innerhalb des d-Blocks)
    pv_mask = None
    pv_v_setpoint = None
    pv_p_setpoint = None

    if include_pv and len(pv_idx) > 0:
        # Position der PV-Knoten innerhalb von d_idx
        pv_mask = np.isin(d_idx, pv_idx)

        # Sollspannung aus PPC
        pv_v_setpoint = ppc["bus"][pv_idx, 7].copy()  # Vm setpoint

        # Soll-Wirkleistung (Netto: Pg - Pd)
        pv_p_setpoint = np.zeros(len(pv_idx))
        for i, bus in enumerate(pv_idx):
            gen_mask = gen_buses == bus
            p_gen = np.sum(ppc["gen"][gen_mask, 1]) / base_mva
            p_load = ppc["bus"][bus, 2] / base_mva
            pv_p_setpoint[i] = p_gen - p_load  # Netto-Einspeisung

    return NetworkData(
        Y_dd=Y_dd,
        Y_ds=Y_ds,
        v_s=v_s,
        s_nom=s_nom,
        alpha_p=np.ones(len(d_idx)),
        alpha_i=np.zeros(len(d_idx)),
        alpha_z=np.zeros(len(d_idx)),
        n_buses=len(d_idx),
        n_phases=1,
        bus_names=[f"bus_{i}" for i in d_idx],
        pv_mask=pv_mask,
        pv_v_setpoint=pv_v_setpoint,
        pv_p_setpoint=pv_p_setpoint,
    )


def get_pq_indices_from_net(net: pp.pandapowerNet) -> np.ndarray:
    """Gibt die internen PPC-Indizes der PQ-Knoten zurück."""
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    return np.where(bus_types == 1)[0]


def get_pv_indices_from_net(net: pp.pandapowerNet) -> np.ndarray:
    """Gibt die internen PPC-Indizes der PV-Knoten zurück."""
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    return np.where(bus_types == 2)[0]


def get_non_slack_indices_from_net(net: pp.pandapowerNet) -> np.ndarray:
    """Gibt die internen PPC-Indizes aller Nicht-Slack-Knoten zurück."""
    ppc = net._ppc
    bus_types = ppc["bus"][:, 1].astype(int)
    return np.where(bus_types != 3)[0]
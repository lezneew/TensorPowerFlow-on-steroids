# tensor_power_flow/src/tpf/generators/radial_network.py
"""
Generator für radiale Verteilnetze mit PV-Knoten
==================================================

Erzeugt parametrierbare pandapower-Netze mit:
- Baumtopologie (k-ary tree oder zufällige Radialstruktur)
- Konfigurierbarer Anzahl und Platzierung von PV-Knoten (DG)
- Einstellbarem R/X-Verhältnis, Lastprofil, Spannungsebene
- Konsistenter Verbraucherkonvention

Einsatzzweck: Testnetze für TPF mit PV-Knoten (Methode A & B)
in Szenarien mit η < 1 (garantierte FPI-Konvergenz).

Verwendung:
    from tpf.generators.radial_network import TEST_NETWORKS, create_radial_network

    # Einzelnes Netz:
    net = create_radial_network(n_buses=50, n_pv=5)

    # Alle Testnetze iterieren:
    for name, info in TEST_NETWORKS.items():
        net = info["constructor"]()
        print(f"{name}: {info['description']}")
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
from dataclasses import dataclass
from typing import Literal


# ══════════════════════════════════════════════════════════════════════
#  Konfiguration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RadialNetworkConfig:
    """Konfiguration für die Netzgenerierung."""

    # ── Topologie ──
    n_buses: int = 34
    branching_factor: int = 3
    topology: Literal["tree", "chain", "random_tree"] = "tree"

    # ── Spannungsebene ──
    vn_kv: float = 20.0
    v_slack_pu: float = 1.02

    # ── Leitungsparameter ──
    r_ohm_per_km: float = 0.40
    x_ohm_per_km: float = 0.30
    c_nf_per_km: float = 0.0
    line_length_km: float = 1.0
    line_length_std: float = 0.3

    # ── Lasten ──
    p_load_mean_kw: float = 50.0
    p_load_std_kw: float = 20.0
    cos_phi: float = 0.95
    load_at_every_bus: bool = True

    # ── PV-Knoten (DG-Einspeiser) ──
    n_pv: int = 3
    pv_placement: Literal["uniform", "end", "random", "custom"] = "uniform"
    pv_buses: list[int] | None = None
    pv_p_mw: float = 0.2
    pv_p_std_mw: float = 0.05
    pv_vm_pu: float = 1.02
    pv_vm_std_pu: float = 0.01
    pv_q_max_mvar: float = 0.15
    pv_q_min_mvar: float = -0.15

    # ── Reproduzierbarkeit ──
    seed: int | None = 42

    @property
    def rx_ratio(self) -> float:
        if self.x_ohm_per_km == 0:
            return np.inf
        return self.r_ohm_per_km / self.x_ohm_per_km

    @property
    def pv_penetration(self) -> float:
        p_load_total = (self.n_buses - 1) * self.p_load_mean_kw / 1000
        p_pv_total = self.n_pv * self.pv_p_mw
        return p_pv_total / max(p_load_total, 1e-6)


# ══════════════════════════════════════════════════════════════════════
#  Netzgenerator
# ══════════════════════════════════════════════════════════════════════

class RadialNetworkGenerator:
    """
    Erzeugt radiale Verteilnetze mit PV-Knoten.

    Verwendung:
        config = RadialNetworkConfig(n_buses=50, n_pv=5)
        gen = RadialNetworkGenerator(config)
        net = gen.generate()
    """

    def __init__(self, config: RadialNetworkConfig | None = None):
        self.config = config or RadialNetworkConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def generate(self) -> pp.pandapowerNet:
        """Erzeugt ein einzelnes radiales Verteilnetz."""
        graph = self._create_topology()
        net = self._build_pandapower_net(graph)
        self._validate(net)
        return net

    def generate_load_scenarios(
        self, n_steps: int = 100, load_range: tuple = (0.3, 1.5)
    ) -> tuple[pp.pandapowerNet, np.ndarray, np.ndarray]:
        """
        Erzeugt ein Netz + τ Lastszenarien (für Tensor-Betrieb).

        Returns
        -------
        net : pandapowerNet (Basisnetz)
        p_mw_matrix : (n_loads, n_steps) Wirkleistung in MW
        q_mvar_matrix : (n_loads, n_steps) Blindleistung in MVAr
        """
        net = self.generate()
        n_loads = len(net.load)

        lambdas = self._rng.uniform(
            load_range[0], load_range[1], size=(n_loads, n_steps)
        )

        p_base = net.load["p_mw"].values.reshape(-1, 1)
        q_base = net.load["q_mvar"].values.reshape(-1, 1)

        return net, p_base * lambdas, q_base * lambdas

    # ──────────────────────────────────────────────────────────────
    #  Private: Topologie
    # ──────────────────────────────────────────────────────────────

    def _create_topology(self) -> nx.Graph:
        cfg = self.config
        n = cfg.n_buses

        if cfg.topology == "tree":
            G = nx.full_rary_tree(cfg.branching_factor, n)
        elif cfg.topology == "chain":
            G = nx.path_graph(n)
        elif cfg.topology == "random_tree":
            G = nx.nx.random_labeled_tree(n, seed=cfg.seed)
        else:
            raise ValueError(f"Unbekannte Topologie: {cfg.topology}")

        actual_n = len(G.nodes)
        if actual_n < n:
            for i in range(actual_n, n):
                parent = self._rng.integers(0, actual_n)
                G.add_edge(parent, i)

        G = nx.Graph(nx.subgraph(G, range(n)))
        assert len(G.nodes) == n
        assert nx.is_tree(G)
        return G

    # ──────────────────────────────────────────────────────────────
    #  Private: pandapower-Netz aufbauen
    # ──────────────────────────────────────────────────────────────

    def _build_pandapower_net(self, graph: nx.Graph) -> pp.pandapowerNet:
        cfg = self.config
        net = pp.create_empty_network(
            name=f"radial_{cfg.n_buses}bus_{cfg.n_pv}pv"
        )
        n = cfg.n_buses

        # Busse
        bus_ids = []
        for i in range(n):
            bus = pp.create_bus(net, vn_kv=cfg.vn_kv, name=f"Bus {i}")
            bus_ids.append(bus)

        # Slack
        pp.create_ext_grid(net, bus=bus_ids[0], vm_pu=cfg.v_slack_pu, name="Slack")

        # Leitungen
        for u, v in graph.edges():
            length = max(
                0.1,
                cfg.line_length_km + self._rng.normal(0, cfg.line_length_std),
            )
            pp.create_line_from_parameters(
                net,
                from_bus=bus_ids[u],
                to_bus=bus_ids[v],
                length_km=length,
                r_ohm_per_km=cfg.r_ohm_per_km,
                x_ohm_per_km=cfg.x_ohm_per_km,
                c_nf_per_km=cfg.c_nf_per_km,
                max_i_ka=1.0,
                name=f"Line {u}-{v}",
            )

        # PV-Busse auswählen
        pv_buses = self._select_pv_buses(graph)

        # Lasten
        for bus_i in range(1, n):
            if bus_i in pv_buses and not cfg.load_at_every_bus:
                continue
            p_kw = max(1.0, self._rng.normal(cfg.p_load_mean_kw, cfg.p_load_std_kw))
            q_kvar = p_kw * np.tan(np.arccos(cfg.cos_phi))
            pp.create_load(
                net,
                bus=bus_ids[bus_i],
                p_mw=p_kw / 1000.0,
                q_mvar=q_kvar / 1000.0,
                name=f"Load Bus {bus_i}",
            )

        # Generatoren (PV-Knoten)
        for i, pv_bus in enumerate(pv_buses):
            p_mw = max(0.01, cfg.pv_p_mw + self._rng.normal(0, cfg.pv_p_std_mw))
            vm_pu = np.clip(
                cfg.pv_vm_pu + self._rng.normal(0, cfg.pv_vm_std_pu), 0.95, 1.1
            )
            pp.create_gen(
                net,
                bus=bus_ids[pv_bus],
                p_mw=p_mw,
                vm_pu=vm_pu,
                max_q_mvar=cfg.pv_q_max_mvar,
                min_q_mvar=cfg.pv_q_min_mvar,
                name=f"DG {i+1} (Bus {pv_bus})",
            )

        return net

    # ──────────────────────────────────────────────────────────────
    #  Private: PV-Platzierung
    # ──────────────────────────────────────────────────────────────

    def _select_pv_buses(self, graph: nx.Graph) -> list[int]:
        cfg = self.config
        n = cfg.n_buses
        n_pv = min(cfg.n_pv, n - 2)  # Mindestens 1 PQ-Knoten bleibt

        if n_pv == 0:
            return []

        if cfg.pv_placement == "custom" and cfg.pv_buses is not None:
            return cfg.pv_buses[:n_pv]

        candidates = list(range(1, n))

        if cfg.pv_placement == "end":
            leaves = [
                node for node in graph.nodes()
                if graph.degree(node) == 1 and node != 0
            ]
            if len(leaves) >= n_pv:
                selected = sorted(self._rng.choice(leaves, n_pv, replace=False))
            else:
                remaining = [c for c in candidates if c not in leaves]
                extra = self._rng.choice(remaining, n_pv - len(leaves), replace=False)
                selected = sorted(list(leaves) + list(extra))

        elif cfg.pv_placement == "uniform":
            distances = nx.single_source_shortest_path_length(graph, 0)
            sorted_by_dist = sorted(candidates, key=lambda x: distances.get(x, 0))
            step = max(1, len(sorted_by_dist) // (n_pv + 1))
            indices = [(i + 1) * step for i in range(n_pv)]
            indices = [min(idx, len(sorted_by_dist) - 1) for idx in indices]
            selected = [sorted_by_dist[idx] for idx in indices]

        elif cfg.pv_placement == "random":
            selected = sorted(self._rng.choice(candidates, n_pv, replace=False))

        else:
            raise ValueError(f"Unbekannte PV-Platzierung: {cfg.pv_placement}")

        return [int(s) for s in selected]

    # ──────────────────────────────────────────────────────────────
    #  Private: Validierung
    # ──────────────────────────────────────────────────────────────

    def _validate(self, net: pp.pandapowerNet) -> None:
        try:
            pp.runpp(net, algorithm="nr", tolerance_mva=1e-6, max_iteration=50)
            if not net.converged:
                raise RuntimeError("NR konvergiert nicht für generiertes Netz.")
        except Exception as e:
            raise RuntimeError(f"Netz-Validierung fehlgeschlagen: {e}")


# ══════════════════════════════════════════════════════════════════════
#  Convenience-Funktionen
# ══════════════════════════════════════════════════════════════════════

def create_radial_network(
    n_buses: int = 34,
    n_pv: int = 3,
    rx_ratio: float = 1.3,
    load_kw: float = 50.0,
    pv_p_mw: float = 0.2,
    pv_vm_pu: float = 1.02,
    pv_vm_std_pu: float = 0.01,
    seed: int = 42,
    topology: str = "tree",
    pv_placement: str = "uniform",
) -> pp.pandapowerNet:
    """
    Schnelle Erzeugung eines radialen Verteilnetzes mit PV-Knoten.

    Parameters
    ----------
    n_buses : int – Gesamtzahl Busse (inkl. Slack)
    n_pv : int – Anzahl PV-Knoten
    rx_ratio : float – R/X Verhältnis der Leitungen
    load_kw : float – Mittlere Last pro Bus [kW]
    pv_p_mw : float – Wirkleistung pro DG [MW]
    pv_vm_pu : float – Sollspannung der DG [p.u.]
    pv_vm_std_pu : float – Standardabweichung der Sollspannung [p.u.]
    seed : int – Zufallssamen
    topology : str – "tree", "chain", "random_tree"
    pv_placement : str – "uniform", "end", "random"

    Returns
    -------
    pandapowerNet
    """
    x_ohm = 0.30
    r_ohm = rx_ratio * x_ohm

    config = RadialNetworkConfig(
        n_buses=n_buses,
        n_pv=n_pv,
        r_ohm_per_km=r_ohm,
        x_ohm_per_km=x_ohm,
        p_load_mean_kw=load_kw,
        pv_p_mw=pv_p_mw,
        pv_vm_pu=pv_vm_pu,
        pv_vm_std_pu=pv_vm_std_pu,
        seed=seed,
        topology=topology,
        pv_placement=pv_placement,
    )
    gen = RadialNetworkGenerator(config)
    return gen.generate()


# ══════════════════════════════════════════════════════════════════════
#  IEEE-Standardnetze mit PV-Knoten
# ══════════════════════════════════════════════════════════════════════

def create_4bus_1pv() -> pp.pandapowerNet:
    """
    Minimales Testnetz: 4 Busse, 1 Slack, 1 PV, 2 PQ.

        Slack (Bus 0) ---Line--- PV (Bus 1) ---Line--- PQ (Bus 2)
                                     |
                                   Line
                                     |
                                 PQ (Bus 3)
    """
    net = pp.create_empty_network(name="4bus_1pv")

    b0 = pp.create_bus(net, vn_kv=20.0, name="Slack")
    b1 = pp.create_bus(net, vn_kv=20.0, name="PV-Bus")
    b2 = pp.create_bus(net, vn_kv=20.0, name="PQ-Bus 1")
    b3 = pp.create_bus(net, vn_kv=20.0, name="PQ-Bus 2")

    pp.create_ext_grid(net, bus=b0, vm_pu=1.02, name="Grid")
    pp.create_gen(net, bus=b1, p_mw=5.0, vm_pu=1.03, name="Gen PV")

    pp.create_load(net, bus=b1, p_mw=2.0, q_mvar=0.5, name="Load 1")
    pp.create_load(net, bus=b2, p_mw=4.0, q_mvar=1.5, name="Load 2")
    pp.create_load(net, bus=b3, p_mw=3.0, q_mvar=1.0, name="Load 3")

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


def create_33bus_2dg() -> pp.pandapowerNet:
    """IEEE 33-Bus + 2 DG-Einspeiser als PV-Knoten (Bus 12, Bus 24)."""
    net = pn.case33bw()
    pp.create_gen(net, bus=12, p_mw=0.5, vm_pu=1.02, name="DG1")
    pp.create_gen(net, bus=24, p_mw=0.4, vm_pu=1.01, name="DG2")
    return net


def create_33bus_5dg() -> pp.pandapowerNet:
    """IEEE 33-Bus + 5 DG-Einspeiser (hohe PV-Durchdringung)."""
    net = pn.case33bw()
    dg_config = [
        (6, 0.3, 1.02),
        (12, 0.5, 1.02),
        (18, 0.4, 1.01),
        (24, 0.4, 1.01),
        (30, 0.3, 1.02),
    ]
    for bus, p_mw, vm_pu in dg_config:
        pp.create_gen(net, bus=bus, p_mw=p_mw, vm_pu=vm_pu, name=f"DG Bus {bus}")
    return net


def create_ieee30() -> pp.pandapowerNet:
    """IEEE 30-Bus: Standardnetz mit 5 PV-Knoten."""
    return pn.case_ieee30()


def create_ieee57() -> pp.pandapowerNet:
    """IEEE 57-Bus: 6 PV-Knoten."""
    return pn.case57()

def pegase() -> pp.pandapowerNet:
    """IEEE 57-Bus: 6 PV-Knoten."""
    return pn.case9241pegase()


def create_radial_10bus_1pv() -> pp.pandapowerNet:
    """Generiertes 10-Bus Radialnetz mit 1 PV (klein, einfach)."""
    return create_radial_network(n_buses=10, n_pv=1, seed=100)


def create_radial_34bus_3pv() -> pp.pandapowerNet:
    """Generiertes 34-Bus Radialnetz mit 3 PV (Referenzgröße)."""
    return create_radial_network(n_buses=34, n_pv=3, seed=101)


def create_radial_50bus_5pv() -> pp.pandapowerNet:
    """Generiertes 50-Bus Radialnetz mit 5 PV."""
    return create_radial_network(n_buses=50, n_pv=5, seed=102)


def create_radial_100bus_10pv() -> pp.pandapowerNet:
    """Generiertes 100-Bus Radialnetz mit 10 PV."""
    return create_radial_network(n_buses=100, n_pv=10, seed=103)


def create_radial_200bus_20pv() -> pp.pandapowerNet:
    """Generiertes 200-Bus Radialnetz mit 20 PV (Stress-Test)."""
    return create_radial_network(n_buses=200, n_pv=20, seed=104)


def create_radial_chain_20bus_2pv() -> pp.pandapowerNet:
    """20-Bus Kette (Stichleitung) mit 2 PV (worst-case R/X-Pfad)."""
    return create_radial_network(
        n_buses=20, n_pv=2, topology="chain",
        pv_placement="end", rx_ratio=2.0, seed=200
    )


def create_radial_high_rx_34bus_3pv() -> pp.pandapowerNet:
    """34-Bus Baum mit hohem R/X=3.0 (stresst FPI-Konvergenz)."""
    return create_radial_network(
        n_buses=34, n_pv=3, rx_ratio=3.0, seed=201
    )


def create_radial_heavy_load_34bus_3pv() -> pp.pandapowerNet:
    """34-Bus Baum mit hoher Last (η nahe 1)."""
    return create_radial_network(
        n_buses=34, n_pv=3, load_kw=120.0, seed=202
    )


def create_radial_high_pv_penetration() -> pp.pandapowerNet:
    """34-Bus mit 8 PV (hohe Durchdringung, viel Q zu bestimmen)."""
    return create_radial_network(
        n_buses=34, n_pv=8, pv_p_mw=0.3, pv_vm_pu=1.03, seed=203
    )


# ══════════════════════════════════════════════════════════════════════
#  ERWEITERT: Viele Netzgrößen für umfassende Validierung
# ══════════════════════════════════════════════════════════════════════

# ── Größen-Sweep ──

def create_radial_5bus_1pv() -> pp.pandapowerNet:
    """5-Bus Kette mit 1 PV."""
    return create_radial_network(n_buses=5, n_pv=1, topology="chain", seed=300)


def create_radial_8bus_2pv() -> pp.pandapowerNet:
    """8-Bus Baum mit 2 PV."""
    return create_radial_network(n_buses=8, n_pv=2, seed=301)


def create_radial_15bus_2pv() -> pp.pandapowerNet:
    """15-Bus Baum mit 2 PV."""
    return create_radial_network(n_buses=15, n_pv=2, seed=302)


def create_radial_20bus_3pv() -> pp.pandapowerNet:
    """20-Bus Baum mit 3 PV."""
    return create_radial_network(n_buses=20, n_pv=3, seed=303)


def create_radial_25bus_4pv() -> pp.pandapowerNet:
    """25-Bus Baum mit 4 PV."""
    return create_radial_network(n_buses=25, n_pv=4, seed=304)


def create_radial_40bus_4pv() -> pp.pandapowerNet:
    """40-Bus Baum mit 4 PV."""
    return create_radial_network(n_buses=40, n_pv=4, seed=305)


def create_radial_60bus_6pv() -> pp.pandapowerNet:
    """60-Bus Baum mit 6 PV."""
    return create_radial_network(n_buses=60, n_pv=6, seed=306)


def create_radial_75bus_8pv() -> pp.pandapowerNet:
    """75-Bus Baum mit 8 PV."""
    return create_radial_network(n_buses=75, n_pv=8, seed=307)


def create_radial_150bus_15pv() -> pp.pandapowerNet:
    """150-Bus Baum mit 15 PV."""
    return create_radial_network(n_buses=150, n_pv=15, seed=308)


def create_radial_300bus_30pv() -> pp.pandapowerNet:
    """300-Bus Baum mit 30 PV."""
    return create_radial_network(n_buses=300, n_pv=30, seed=309)


def create_radial_500bus_50pv() -> pp.pandapowerNet:
    """500-Bus Baum mit 50 PV (Stress-Test)."""
    return create_radial_network(n_buses=500, n_pv=50, seed=310)


# ── R/X-Variationen (50-Bus Basis) ──

def create_radial_50bus_5pv_rx05() -> pp.pandapowerNet:
    """50-Bus, 5 PV, R/X=0.5 (ÜN-artig)."""
    return create_radial_network(n_buses=50, n_pv=5, rx_ratio=0.5, seed=400)


def create_radial_50bus_5pv_rx10() -> pp.pandapowerNet:
    """50-Bus, 5 PV, R/X=1.0."""
    return create_radial_network(n_buses=50, n_pv=5, rx_ratio=1.0, seed=401)


def create_radial_50bus_5pv_rx20() -> pp.pandapowerNet:
    """50-Bus, 5 PV, R/X=2.0 (NS)."""
    return create_radial_network(n_buses=50, n_pv=5, rx_ratio=2.0, seed=402)


def create_radial_50bus_5pv_rx40() -> pp.pandapowerNet:
    """50-Bus, 5 PV, R/X=4.0 (Stress)."""
    return create_radial_network(n_buses=50, n_pv=5, rx_ratio=4.0, seed=403)


# ── PV-Durchdringung (50-Bus Basis) ──

def create_radial_50bus_1pv() -> pp.pandapowerNet:
    """50-Bus, 1 PV (minimal)."""
    return create_radial_network(n_buses=50, n_pv=1, seed=500)


def create_radial_50bus_10pv() -> pp.pandapowerNet:
    """50-Bus, 10 PV (20%)."""
    return create_radial_network(n_buses=50, n_pv=10, seed=501)


def create_radial_50bus_20pv() -> pp.pandapowerNet:
    """50-Bus, 20 PV (40%)."""
    return create_radial_network(n_buses=50, n_pv=20, pv_p_mw=0.15, seed=502)


# ── Lastprofil (50-Bus Basis) ──

def create_radial_50bus_5pv_light_load() -> pp.pandapowerNet:
    """50-Bus, 5 PV, 20 kW/Bus (leicht)."""
    return create_radial_network(n_buses=50, n_pv=5, load_kw=20.0, seed=600)


def create_radial_50bus_5pv_heavy_load() -> pp.pandapowerNet:
    """50-Bus, 5 PV, 100 kW/Bus (schwer)."""
    return create_radial_network(n_buses=50, n_pv=5, load_kw=100.0, seed=601)


def create_radial_50bus_5pv_very_heavy() -> pp.pandapowerNet:
    """50-Bus, 5 PV, 150 kW/Bus (η nahe 1)."""
    return create_radial_network(n_buses=50, n_pv=5, load_kw=150.0, seed=602)


# ── Topologie-Variationen ──

def create_chain_50bus_5pv() -> pp.pandapowerNet:
    """50-Bus Kette mit 5 PV."""
    return create_radial_network(
        n_buses=50, n_pv=5, topology="chain", pv_placement="uniform", seed=700
    )


def create_chain_100bus_10pv() -> pp.pandapowerNet:
    """100-Bus Kette mit 10 PV."""
    return create_radial_network(
        n_buses=100, n_pv=10, topology="chain", pv_placement="uniform", seed=701
    )


def create_random_tree_50bus_5pv() -> pp.pandapowerNet:
    """50-Bus Zufallsbaum mit 5 PV."""
    return create_radial_network(
        n_buses=50, n_pv=5, topology="random_tree", seed=702
    )


def create_random_tree_100bus_10pv() -> pp.pandapowerNet:
    """100-Bus Zufallsbaum mit 10 PV."""
    return create_radial_network(
        n_buses=100, n_pv=10, topology="random_tree", seed=703
    )


# ── PV-Platzierung (50-Bus Basis) ──

def create_radial_50bus_5pv_end() -> pp.pandapowerNet:
    """50-Bus, 5 PV an Leitungsenden."""
    return create_radial_network(
        n_buses=50, n_pv=5, pv_placement="end", seed=800
    )


def create_radial_50bus_5pv_random() -> pp.pandapowerNet:
    """50-Bus, 5 PV zufällig platziert."""
    return create_radial_network(
        n_buses=50, n_pv=5, pv_placement="random", seed=801
    )


# ── PV Spannungs-Sollwerte ──

def create_radial_50bus_5pv_high_vm() -> pp.pandapowerNet:
    """50-Bus, 5 PV mit V_spec=1.05 p.u."""
    return create_radial_network(
        n_buses=50, n_pv=5, pv_vm_pu=1.05, pv_vm_std_pu=0.005, seed=900
    )


def create_radial_50bus_5pv_low_vm() -> pp.pandapowerNet:
    """50-Bus, 5 PV mit V_spec=0.98 p.u."""
    return create_radial_network(
        n_buses=50, n_pv=5, pv_vm_pu=0.98, pv_vm_std_pu=0.005, seed=901
    )


def create_radial_50bus_5pv_mixed_vm() -> pp.pandapowerNet:
    """50-Bus, 5 PV mit unterschiedlichen V_spec."""
    return create_radial_network(
        n_buses=50, n_pv=5, pv_vm_pu=1.02, pv_vm_std_pu=0.03, seed=902
    )


# ── IEEE-Netze mit zusätzlichen DG ──

def create_33bus_1dg() -> pp.pandapowerNet:
    """IEEE 33-Bus + 1 DG an Bus 17."""
    net = pn.case33bw()
    pp.create_gen(net, bus=17, p_mw=0.5, vm_pu=1.02, name="DG1")
    return net


def create_33bus_8dg() -> pp.pandapowerNet:
    """IEEE 33-Bus + 8 DG (extremes PV-Szenario)."""
    net = pn.case33bw()
    dg_buses = [3, 6, 9, 12, 15, 18, 24, 30]
    for i, bus in enumerate(dg_buses):
        pp.create_gen(net, bus=bus, p_mw=0.3, vm_pu=1.01 + 0.005 * (i % 3),
                      name=f"DG{i+1}")
    return net


def create_case9() -> pp.pandapowerNet:
    """IEEE 9-Bus (3 Gen, 2 PV)."""
    return pn.case9()


def create_case39() -> pp.pandapowerNet:
    """IEEE 39-Bus New England (10 Gen)."""
    return pn.case39()


def _try_case_ieee118() -> pp.pandapowerNet:
    """IEEE 118-Bus — mit Fallback."""
    try:
        return pn.case_ieee118()
    except AttributeError:
        return pn.case118()


# ══════════════════════════════════════════════════════════════════════
#  TEST_NETWORKS — Hauptexport für Validierungsskripte
# ══════════════════════════════════════════════════════════════════════

TEST_NETWORKS: dict[str, dict] = {
    # ── Minimale Netze (Debugging) ──
    "4bus_1pv": {
        "constructor": create_4bus_1pv,
        "description": "4 Busse, 1 PV, 2 PQ (minimal, handgebaut)",
        "n_pv": 1,
        "category": "minimal",
    },
    # ── IEEE-Standardnetze mit DG ──
    "33bus_2dg": {
        "constructor": create_33bus_2dg,
        "description": "IEEE 33-Bus radial + 2 DG (PV an Bus 12, 24)",
        "n_pv": 2,
        "category": "ieee_radial",
    },
    "33bus_5dg": {
        "constructor": create_33bus_5dg,
        "description": "IEEE 33-Bus radial + 5 DG (hohe Durchdringung)",
        "n_pv": 5,
        "category": "ieee_radial",
    },
    # ── IEEE vermaschte Netze (Härtetest) ──
    "ieee30": {
        "constructor": create_ieee30,
        "description": "IEEE 30-Bus vermascht (5 PV, Standard-ÜN)",
        "n_pv": 5,
        "category": "ieee_meshed",
    },
    "ieee57": {
        "constructor": create_ieee57,
        "description": "IEEE 57-Bus vermascht (6 PV)",
        "n_pv": 6,
        "category": "ieee_meshed",
    },
    "pegase9000": {
        "constructor": pegase,
        "description": "IEEE 57-Bus vermascht (6 PV)",
        "n_pv": 6,
        "category": "ieee_meshed",
    },
    # ── Generierte Radialnetze (skalierbar) ──
    "radial_10bus_1pv": {
        "constructor": create_radial_10bus_1pv,
        "description": "Generiert: 10-Bus Baum, 1 PV (trivial)",
        "n_pv": 1,
        "category": "generated_radial",
    },
    "radial_34bus_3pv": {
        "constructor": create_radial_34bus_3pv,
        "description": "Generiert: 34-Bus Baum, 3 PV (Referenz)",
        "n_pv": 3,
        "category": "generated_radial",
    },
    "radial_50bus_5pv": {
        "constructor": create_radial_50bus_5pv,
        "description": "Generiert: 50-Bus Baum, 5 PV",
        "n_pv": 5,
        "category": "generated_radial",
    },
    "radial_100bus_10pv": {
        "constructor": create_radial_100bus_10pv,
        "description": "Generiert: 100-Bus Baum, 10 PV",
        "n_pv": 10,
        "category": "generated_radial",
    },
    "radial_200bus_20pv": {
        "constructor": create_radial_200bus_20pv,
        "description": "Generiert: 200-Bus Baum, 20 PV (Stress)",
        "n_pv": 20,
        "category": "generated_radial",
    },
    # ── Spezialfälle (Konvergenz-Stress) ──
    "chain_20bus_2pv": {
        "constructor": create_radial_chain_20bus_2pv,
        "description": "20-Bus Kette, 2 PV an Enden, R/X=2.0",
        "n_pv": 2,
        "category": "stress",
    },
    "high_rx_34bus_3pv": {
        "constructor": create_radial_high_rx_34bus_3pv,
        "description": "34-Bus Baum, R/X=3.0 (stresst FPI)",
        "n_pv": 3,
        "category": "stress",
    },
    "heavy_load_34bus_3pv": {
        "constructor": create_radial_heavy_load_34bus_3pv,
        "description": "34-Bus Baum, 120 kW/Bus (η nahe 1)",
        "n_pv": 3,
        "category": "stress",
    },
    "high_pv_34bus_8pv": {
        "constructor": create_radial_high_pv_penetration,
        "description": "34-Bus Baum, 8 PV (hohe DG-Durchdringung)",
        "n_pv": 8,
        "category": "stress",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  TEST_NETWORKS_COMPREHENSIVE — Alle Größen und Variationen
# ══════════════════════════════════════════════════════════════════════

TEST_NETWORKS_COMPREHENSIVE: dict[str, dict] = {
    # ════════════════════════════════════════════════════════════
    #  GRÖSSE: Sehr klein (3-10 Busse)
    # ════════════════════════════════════════════════════════════
    "4bus_1pv": {
        "constructor": create_4bus_1pv,
        "description": "4 Busse, 1 PV, 2 PQ (minimal, handgebaut)",
        "n_pv": 1,
        "category": "size_tiny",
    },
    "5bus_1pv_chain": {
        "constructor": create_radial_5bus_1pv,
        "description": "5-Bus Kette, 1 PV",
        "n_pv": 1,
        "category": "size_tiny",
    },
    "8bus_2pv": {
        "constructor": create_radial_8bus_2pv,
        "description": "8-Bus Baum, 2 PV",
        "n_pv": 2,
        "category": "size_tiny",
    },
    "case9": {
        "constructor": create_case9,
        "description": "IEEE 9-Bus (3 Gen, 2 PV)",
        "n_pv": 2,
        "category": "size_tiny",
    },

    # ════════════════════════════════════════════════════════════
    #  GRÖSSE: Klein (10-25 Busse)
    # ════════════════════════════════════════════════════════════
    "radial_10bus_1pv": {
        "constructor": create_radial_10bus_1pv,
        "description": "10-Bus Baum, 1 PV",
        "n_pv": 1,
        "category": "size_small",
    },
    "radial_15bus_2pv": {
        "constructor": create_radial_15bus_2pv,
        "description": "15-Bus Baum, 2 PV",
        "n_pv": 2,
        "category": "size_small",
    },
    "radial_20bus_3pv": {
        "constructor": create_radial_20bus_3pv,
        "description": "20-Bus Baum, 3 PV",
        "n_pv": 3,
        "category": "size_small",
    },
    "chain_20bus_2pv": {
        "constructor": create_radial_chain_20bus_2pv,
        "description": "20-Bus Kette, 2 PV an Enden, R/X=2.0",
        "n_pv": 2,
        "category": "size_small",
    },
    "radial_25bus_4pv": {
        "constructor": create_radial_25bus_4pv,
        "description": "25-Bus Baum, 4 PV",
        "n_pv": 4,
        "category": "size_small",
    },

    # ════════════════════════════════════════════════════════════
    #  GRÖSSE: Mittel (30-75 Busse)
    # ════════════════════════════════════════════════════════════
    "ieee30": {
        "constructor": create_ieee30,
        "description": "IEEE 30-Bus vermascht (5 PV)",
        "n_pv": 5,
        "category": "size_medium",
    },
    "33bus_1dg": {
        "constructor": create_33bus_1dg,
        "description": "IEEE 33-Bus + 1 DG",
        "n_pv": 1,
        "category": "size_medium",
    },
    "33bus_2dg": {
        "constructor": create_33bus_2dg,
        "description": "IEEE 33-Bus + 2 DG",
        "n_pv": 2,
        "category": "size_medium",
    },
    "33bus_5dg": {
        "constructor": create_33bus_5dg,
        "description": "IEEE 33-Bus + 5 DG",
        "n_pv": 5,
        "category": "size_medium",
    },
    "33bus_8dg": {
        "constructor": create_33bus_8dg,
        "description": "IEEE 33-Bus + 8 DG (extrem)",
        "n_pv": 8,
        "category": "size_medium",
    },
    "radial_34bus_3pv": {
        "constructor": create_radial_34bus_3pv,
        "description": "34-Bus Baum, 3 PV (Referenz)",
        "n_pv": 3,
        "category": "size_medium",
    },
    "high_rx_34bus_3pv": {
        "constructor": create_radial_high_rx_34bus_3pv,
        "description": "34-Bus, R/X=3.0 (Stress FPI)",
        "n_pv": 3,
        "category": "size_medium",
    },
    "heavy_load_34bus_3pv": {
        "constructor": create_radial_heavy_load_34bus_3pv,
        "description": "34-Bus, 120 kW/Bus (η nahe 1)",
        "n_pv": 3,
        "category": "size_medium",
    },
    "high_pv_34bus_8pv": {
        "constructor": create_radial_high_pv_penetration,
        "description": "34-Bus, 8 PV (hohe Durchdringung)",
        "n_pv": 8,
        "category": "size_medium",
    },
    "case39": {
        "constructor": create_case39,
        "description": "IEEE 39-Bus New England (10 Gen)",
        "n_pv": 9,
        "category": "size_medium",
    },
    "radial_40bus_4pv": {
        "constructor": create_radial_40bus_4pv,
        "description": "40-Bus Baum, 4 PV",
        "n_pv": 4,
        "category": "size_medium",
    },
    "radial_50bus_5pv": {
        "constructor": create_radial_50bus_5pv,
        "description": "50-Bus Baum, 5 PV",
        "n_pv": 5,
        "category": "size_medium",
    },
    "chain_50bus_5pv": {
        "constructor": create_chain_50bus_5pv,
        "description": "50-Bus Kette, 5 PV",
        "n_pv": 5,
        "category": "size_medium",
    },
    "random_tree_50bus_5pv": {
        "constructor": create_random_tree_50bus_5pv,
        "description": "50-Bus Zufallsbaum, 5 PV",
        "n_pv": 5,
        "category": "size_medium",
    },
    "ieee57": {
        "constructor": create_ieee57,
        "description": "IEEE 57-Bus vermascht (6 PV)",
        "n_pv": 6,
        "category": "size_medium",
    },
    "radial_60bus_6pv": {
        "constructor": create_radial_60bus_6pv,
        "description": "60-Bus Baum, 6 PV",
        "n_pv": 6,
        "category": "size_medium",
    },
    "radial_75bus_8pv": {
        "constructor": create_radial_75bus_8pv,
        "description": "75-Bus Baum, 8 PV",
        "n_pv": 8,
        "category": "size_medium",
    },

    # ════════════════════════════════════════════════════════════
    #  GRÖSSE: Groß (100-300 Busse)
    # ════════════════════════════════════════════════════════════
    "radial_100bus_10pv": {
        "constructor": create_radial_100bus_10pv,
        "description": "100-Bus Baum, 10 PV",
        "n_pv": 10,
        "category": "size_large",
    },
    "chain_100bus_10pv": {
        "constructor": create_chain_100bus_10pv,
        "description": "100-Bus Kette, 10 PV",
        "n_pv": 10,
        "category": "size_large",
    },
    "random_tree_100bus_10pv": {
        "constructor": create_random_tree_100bus_10pv,
        "description": "100-Bus Zufallsbaum, 10 PV",
        "n_pv": 10,
        "category": "size_large",
    },
    "radial_150bus_15pv": {
        "constructor": create_radial_150bus_15pv,
        "description": "150-Bus Baum, 15 PV",
        "n_pv": 15,
        "category": "size_large",
    },
    "radial_200bus_20pv": {
        "constructor": create_radial_200bus_20pv,
        "description": "200-Bus Baum, 20 PV",
        "n_pv": 20,
        "category": "size_large",
    },
    "radial_300bus_30pv": {
        "constructor": create_radial_300bus_30pv,
        "description": "300-Bus Baum, 30 PV",
        "n_pv": 30,
        "category": "size_large",
    },

    # ════════════════════════════════════════════════════════════
    #  GRÖSSE: Sehr groß (500+ Busse)
    # ════════════════════════════════════════════════════════════
    "radial_500bus_50pv": {
        "constructor": create_radial_500bus_50pv,
        "description": "500-Bus Baum, 50 PV (Stress)",
        "n_pv": 50,
        "category": "size_xlarge",
    },

    # ════════════════════════════════════════════════════════════
    #  VARIATION: R/X-Verhältnis
    # ════════════════════════════════════════════════════════════
    "50bus_5pv_rx05": {
        "constructor": create_radial_50bus_5pv_rx05,
        "description": "50-Bus, R/X=0.5 (ÜN-artig)",
        "n_pv": 5,
        "category": "variation_rx",
    },
    "50bus_5pv_rx10": {
        "constructor": create_radial_50bus_5pv_rx10,
        "description": "50-Bus, R/X=1.0",
        "n_pv": 5,
        "category": "variation_rx",
    },
    "50bus_5pv_rx20": {
        "constructor": create_radial_50bus_5pv_rx20,
        "description": "50-Bus, R/X=2.0 (NS)",
        "n_pv": 5,
        "category": "variation_rx",
    },
    "50bus_5pv_rx40": {
        "constructor": create_radial_50bus_5pv_rx40,
        "description": "50-Bus, R/X=4.0 (Stress)",
        "n_pv": 5,
        "category": "variation_rx",
    },

    # ════════════════════════════════════════════════════════════
    #  VARIATION: PV-Durchdringung
    # ════════════════════════════════════════════════════════════
    "50bus_1pv": {
        "constructor": create_radial_50bus_1pv,
        "description": "50-Bus, 1 PV (minimal)",
        "n_pv": 1,
        "category": "variation_pv_count",
    },
    "50bus_10pv": {
        "constructor": create_radial_50bus_10pv,
        "description": "50-Bus, 10 PV (20%)",
        "n_pv": 10,
        "category": "variation_pv_count",
    },
    "50bus_20pv": {
        "constructor": create_radial_50bus_20pv,
        "description": "50-Bus, 20 PV (40%)",
        "n_pv": 20,
        "category": "variation_pv_count",
    },

    # ════════════════════════════════════════════════════════════
    #  VARIATION: Lastprofil
    # ════════════════════════════════════════════════════════════
    "50bus_5pv_light": {
        "constructor": create_radial_50bus_5pv_light_load,
        "description": "50-Bus, 5PV, 20kW/Bus (leicht)",
        "n_pv": 5,
        "category": "variation_load",
    },
    "50bus_5pv_heavy": {
        "constructor": create_radial_50bus_5pv_heavy_load,
        "description": "50-Bus, 5PV, 100kW/Bus (schwer)",
        "n_pv": 5,
        "category": "variation_load",
    },
    "50bus_5pv_very_heavy": {
        "constructor": create_radial_50bus_5pv_very_heavy,
        "description": "50-Bus, 5PV, 150kW/Bus (η nahe 1)",
        "n_pv": 5,
        "category": "variation_load",
    },

    # ════════════════════════════════════════════════════════════
    #  VARIATION: PV-Platzierung
    # ════════════════════════════════════════════════════════════
    "50bus_5pv_uniform": {
        "constructor": create_radial_50bus_5pv,
        "description": "50-Bus, 5PV gleichmäßig",
        "n_pv": 5,
        "category": "variation_placement",
    },
    "50bus_5pv_end": {
        "constructor": create_radial_50bus_5pv_end,
        "description": "50-Bus, 5PV an Enden",
        "n_pv": 5,
        "category": "variation_placement",
    },
    "50bus_5pv_random": {
        "constructor": create_radial_50bus_5pv_random,
        "description": "50-Bus, 5PV zufällig",
        "n_pv": 5,
        "category": "variation_placement",
    },

    # ════════════════════════════════════════════════════════════
    #  VARIATION: PV-Spannungssollwert
    # ════════════════════════════════════════════════════════════
    "50bus_5pv_vm105": {
        "constructor": create_radial_50bus_5pv_high_vm,
        "description": "50-Bus, 5PV, V_spec=1.05",
        "n_pv": 5,
        "category": "variation_vm",
    },
    "50bus_5pv_vm098": {
        "constructor": create_radial_50bus_5pv_low_vm,
        "description": "50-Bus, 5PV, V_spec=0.98",
        "n_pv": 5,
        "category": "variation_vm",
    },
    "50bus_5pv_vm_mixed": {
        "constructor": create_radial_50bus_5pv_mixed_vm,
        "description": "50-Bus, 5PV, V_spec gestreut",
        "n_pv": 5,
        "category": "variation_vm",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Filter-Funktionen für Testauswahl
# ══════════════════════════════════════════════════════════════════════

def get_networks_by_category(category: str) -> dict[str, dict]:
    """Filtert TEST_NETWORKS nach Kategorie."""
    return {k: v for k, v in TEST_NETWORKS.items() if v["category"] == category}


def get_quick_test_networks() -> dict[str, dict]:
    """Nur die schnellen Netze (< 50 Busse) für schnelle Iteration."""
    quick = ["4bus_1pv", "33bus_2dg", "radial_10bus_1pv", "radial_34bus_3pv"]
    return {k: v for k, v in TEST_NETWORKS.items() if k in quick}


def get_radial_only_networks() -> dict[str, dict]:
    """Nur radiale Netze (η < 1 garantiert) — Primärziel des TPF."""
    return {
        k: v for k, v in TEST_NETWORKS.items()
        if v["category"] in ("minimal", "ieee_radial", "generated_radial", "stress")
    }


def get_full_test_suite() -> dict[str, dict]:
    """Alle Netze inkl. IEEE vermaschter Netze."""
    return TEST_NETWORKS


def get_comprehensive_networks() -> dict[str, dict]:
    """Alle Netze aus dem erweiterten Katalog."""
    return TEST_NETWORKS_COMPREHENSIVE


def get_size_sweep_networks() -> dict[str, dict]:
    """Netze aufsteigend nach Größe (für Skalierungstests)."""
    size_cats = ("size_tiny", "size_small", "size_medium", "size_large", "size_xlarge")
    return {k: v for k, v in TEST_NETWORKS_COMPREHENSIVE.items()
            if v["category"] in size_cats}


def get_variation_networks(variation: str) -> dict[str, dict]:
    """Filtert nach Variationstyp: 'rx', 'pv_count', 'load', 'placement', 'vm'."""
    cat = f"variation_{variation}"
    return {k: v for k, v in TEST_NETWORKS_COMPREHENSIVE.items()
            if v["category"] == cat}


# ══════════════════════════════════════════════════════════════════════
#  Batch-Erzeugung für Zeitreihen-Tests
# ══════════════════════════════════════════════════════════════════════

def create_batch_scenarios(
    n_buses: int = 34,
    n_pv: int = 3,
    n_steps: int = 100,
    load_range: tuple = (0.5, 1.5),
    seed: int = 42,
) -> tuple[pp.pandapowerNet, np.ndarray, np.ndarray]:
    """
    Erzeugt ein Netz + Zeitreihen-Lastszenarien für den Tensor-Betrieb.

    Returns
    -------
    net : pandapowerNet
    p_matrix : (n_loads, n_steps) in MW
    q_matrix : (n_loads, n_steps) in MVAr
    """
    config = RadialNetworkConfig(n_buses=n_buses, n_pv=n_pv, seed=seed)
    gen = RadialNetworkGenerator(config)
    return gen.generate_load_scenarios(n_steps=n_steps, load_range=load_range)


# ══════════════════════════════════════════════════════════════════════
#  Standalone-Aufruf: Zeigt Übersicht aller Testnetze
# ══════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  TEST-NETZE FÜR PV-KNOTEN VALIDIERUNG (radial_network.py)       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    print(f"\n  TEST_NETWORKS: {len(TEST_NETWORKS)} Netze")
    print(f"  TEST_NETWORKS_COMPREHENSIVE: {len(TEST_NETWORKS_COMPREHENSIVE)} Netze")
    print(f"{'─'*70}")
    print(f"  {'Name':<28} {'Kat.':<20} {'#PV':<5} {'Beschreibung'}")
    print(f"{'─'*70}")

    for name, info in TEST_NETWORKS_COMPREHENSIVE.items():
        print(
            f"  {name:<28} {info['category']:<20} "
            f"{info['n_pv']:<5} {info['description']}"
        )

    # Alle erzeugen und prüfen
    print(f"\n{'═'*70}")
    print(f"  ERZEUGUNG & VALIDIERUNG (Comprehensive)")
    print(f"{'═'*70}")

    n_pass = 0
    n_fail = 0

    for name, info in TEST_NETWORKS_COMPREHENSIVE.items():
        try:
            net = info["constructor"]()
            pp.runpp(net, algorithm="nr", tolerance_mva=1e-8)
            ppc = net._ppc
            bus_types = ppc["bus"][:, 1].astype(int)
            n_bus = len(bus_types)
            n_slack = int(np.sum(bus_types == 3))
            n_pv = int(np.sum(bus_types == 2))
            n_pq = int(np.sum(bus_types == 1))

            status = "✓"
            n_pass += 1
            print(
                f"  {status} {name:<28} "
                f"Busse={n_bus:<4} Sl={n_slack} PV={n_pv:<3} PQ={n_pq:<4} "
                f"NR={ppc.get('iterations', '?')} iter"
            )
        except Exception as e:
            status = "✗"
            n_fail += 1
            print(f"  {status} {name:<28} FEHLER: {e}")

    print(f"\n{'─'*70}")
    print(f"  Ergebnis: {n_pass} PASS, {n_fail} FAIL")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
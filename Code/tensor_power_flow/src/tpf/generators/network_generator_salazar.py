# tensor_power_flow/src/tpf/generators/network_generator_salazar.py
"""
Netzgenerator nach Salazar et al. (2024) — Erweitert um PV-Knoten
====================================================================

Repliziert die Netzgenerierung aus dem Paper:
    "Tensor Power Flow Formulations for Multidimensional Analyses
     in Distribution Systems" (arXiv:2403.04578v1)

Original-Ansatz (Paper):
    - Topologie: nx.full_rary_tree(child, nodes)
    - Leitungen: Alle identisch (R=0.3144/line_factor, X=0.054/line_factor)
    - Lasten: Normalverteilt (μ=50*load_factor kW, σ=50 kW)
    - ZIP: Nur constant power (α_P=1, α_I=0, α_Z=0)
    - Spannungsebene: 11 kV
    - s_base: 1000 kVA
    - Keine PV-Knoten

Erweiterung (diese Datei):
    - PV-Knoten (DG-Einspeiser) mit konfigurierbarer Platzierung
    - Ausgabe als pandapower-Netz (statt DataFrames)
    - Kompatibel mit tpf.builders.from_pandapower

HINWEIS: Das Salazar-Netz hat R/X ≈ 5.82 (sehr hoch).
    → Die Kopplung ∂V/∂Q ist SEHR schwach
    → PV-Spannungssollwerte müssen nahe am natürlichen Profil liegen
    → vm_pu wird automatisch aus dem lastflussfreien Profil abgeleitet,
      sofern pv_vm_mode="natural" gesetzt ist (default)

Verwendung:
    from tpf.generators.network_generator_salazar import (
        create_salazar_network,
        SALAZAR_TEST_NETWORKS,
    )

    # Einzelnes Netz (wie im Paper, ohne PV):
    net = create_salazar_network(nodes=100, child=3, load_factor=2, line_factor=3)

    # Mit PV-Knoten:
    net = create_salazar_network(nodes=100, child=3, n_pv=5, pv_p_mw=0.1)
"""

import numpy as np
import pandapower as pp
import networkx as nx
from dataclasses import dataclass
from typing import Literal


# ══════════════════════════════════════════════════════════════════════
#  Konfiguration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SalazarNetworkConfig:
    """
    Konfiguration für die Netzgenerierung nach Salazar et al.

    Die Defaultwerte reproduzieren exakt das Verhalten des Papers.
    """

    # ── Topologie (Paper: nx.full_rary_tree) ──
    nodes: int = 34
    child: int = 3

    # ── Spannungsebene (Paper: 11 kV, s_base=1000 kVA) ──
    vn_kv: float = 11.0
    s_base_kva: float = 1000.0
    v_slack_pu: float = 1.00

    # ── Leitungsparameter (Paper: R=0.3144/line_factor, X=0.054/line_factor) ──
    r_ohm_per_km: float = 0.3144
    x_ohm_per_km: float = 0.054
    line_factor: float = 3.0
    line_length_km: float = 1.0

    # ── Lasten (Paper: N(50*load_factor, 50) kW, Q = 0.1*P) ──
    load_factor: float = 2.0
    load_std_kw: float = 50.0
    q_over_p_ratio: float = 0.1
    clip_loads_positive: bool = True  # NEU: Negative Lasten vermeiden

    # ── PV-Knoten (NEU — nicht im Paper) ──
    n_pv: int = 0
    pv_placement: Literal["uniform", "end", "random", "mid"] = "uniform"
    pv_p_mw: float = 0.05  # Konservativ (50 kW statt 100 kW)
    pv_p_std_mw: float = 0.01
    # PV-Spannungsmodus:
    #   "natural": vm_pu aus PQ-Lastfluss + kleiner Offset (EMPFOHLEN für R/X>>1)
    #   "fixed": Fester Wert (Standard: 1.00)
    pv_vm_mode: Literal["natural", "fixed"] = "natural"
    pv_vm_pu: float = 1.00  # Wird bei "natural" überschrieben
    pv_vm_offset_pu: float = 0.005  # Offset über natürlichem Profil (bei "natural")
    pv_vm_std_pu: float = 0.002  # Streuung der Sollspannung
    pv_q_max_mvar: float = 0.5  # Großzügig wegen schwacher Q↔V-Kopplung
    pv_q_min_mvar: float = -0.5

    # ── Reproduzierbarkeit ──
    seed: int | None = 42

    @property
    def r_effective(self) -> float:
        return self.r_ohm_per_km / self.line_factor

    @property
    def x_effective(self) -> float:
        return self.x_ohm_per_km / self.line_factor

    @property
    def rx_ratio(self) -> float:
        return self.r_ohm_per_km / max(self.x_ohm_per_km, 1e-12)

    @property
    def load_mean_kw(self) -> float:
        return 50.0 * self.load_factor


# ══════════════════════════════════════════════════════════════════════
#  Netzgenerator
# ══════════════════════════════════════════════════════════════════════

class SalazarNetworkGenerator:
    """
    Erzeugt Netze nach dem Verfahren aus Salazar et al. (2024),
    erweitert um PV-Knoten.
    """

    def __init__(self, config: SalazarNetworkConfig | None = None):
        self.config = config or SalazarNetworkConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def generate(self) -> pp.pandapowerNet:
        """Erzeugt ein pandapower-Netz nach Salazar-Konfiguration."""
        cfg = self.config

        # ── 1. Topologie: k-ary tree ──
        G = nx.full_rary_tree(cfg.child, cfg.nodes)
        assert len(G.nodes) == cfg.nodes
        assert nx.is_tree(G)

        # ── 2. pandapower-Netz aufbauen ──
        net = pp.create_empty_network(
            name=f"salazar_{cfg.nodes}bus_c{cfg.child}_lf{cfg.load_factor}_pv{cfg.n_pv}"
        )

        # Busse
        bus_ids = []
        for i in range(cfg.nodes):
            bus = pp.create_bus(net, vn_kv=cfg.vn_kv, name=f"Bus {i}")
            bus_ids.append(bus)

        # Slack
        pp.create_ext_grid(net, bus=bus_ids[0], vm_pu=cfg.v_slack_pu, name="Slack")

        # ── 3. Leitungen ──
        r_eff = cfg.r_effective
        x_eff = cfg.x_effective

        for u, v in G.edges():
            pp.create_line_from_parameters(
                net,
                from_bus=bus_ids[u],
                to_bus=bus_ids[v],
                length_km=cfg.line_length_km,
                r_ohm_per_km=r_eff,
                x_ohm_per_km=x_eff,
                c_nf_per_km=0.0,
                max_i_ka=10.0,
                name=f"Line {u}-{v}",
            )

        # ── 4. Lasten ──
        active_kw = self._rng.normal(
            cfg.load_mean_kw, scale=cfg.load_std_kw, size=cfg.nodes
        )

        # Negative Lasten clippen (Normalverteilung kann neg. werden!)
        if cfg.clip_loads_positive:
            active_kw = np.maximum(active_kw, 1.0)  # Mindestens 1 kW

        active_kw = np.round(active_kw, 3)
        reactive_kw = np.round(active_kw * cfg.q_over_p_ratio, 3)

        # Slack: keine Last
        active_kw[0] = 0.0
        reactive_kw[0] = 0.0

        # PV-Busse auswählen
        pv_buses = self._select_pv_buses(G) if cfg.n_pv > 0 else []

        for i in range(1, cfg.nodes):
            p_mw = active_kw[i] / 1000.0
            q_mvar = reactive_kw[i] / 1000.0
            pp.create_load(
                net,
                bus=bus_ids[i],
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=f"Load Bus {i}",
            )

        # ── 5. PV-Knoten ──
        if cfg.n_pv > 0:
            # Bestimme vm_pu für jeden PV-Knoten
            vm_values = self._determine_pv_voltages(net, bus_ids, pv_buses)

            for idx, pv_bus in enumerate(pv_buses):
                p_mw = max(0.005, cfg.pv_p_mw + self._rng.normal(0, cfg.pv_p_std_mw))
                vm_pu = float(vm_values[idx])

                pp.create_gen(
                    net,
                    bus=bus_ids[pv_bus],
                    p_mw=p_mw,
                    vm_pu=vm_pu,
                    max_q_mvar=cfg.pv_q_max_mvar,
                    min_q_mvar=cfg.pv_q_min_mvar,
                    name=f"DG {idx+1} (Bus {pv_bus})",
                )

        # ── 6. Validierung ──
        self._validate(net)

        return net

    # ──────────────────────────────────────────────────────────────
    #  PV-Spannungssollwerte bestimmen
    # ──────────────────────────────────────────────────────────────

    def _determine_pv_voltages(
        self, net: pp.pandapowerNet, bus_ids: list, pv_buses: list
    ) -> np.ndarray:
        """
        Bestimmt realistische vm_pu-Werte für PV-Knoten.

        Bei R/X >> 1 ist die Q↔V-Kopplung schwach. Daher dürfen die
        Sollspannungen nur minimal vom natürlichen Profil abweichen,
        sonst konvergiert NR nicht.

        Strategie "natural":
            1. Löse Netz ohne PV (alle PQ) → natürliches Spannungsprofil
            2. Setze vm_pu = V_natural[bus] + kleiner Offset

        Strategie "fixed":
            Verwende festen Wert (cfg.pv_vm_pu)
        """
        cfg = self.config
        n_pv = len(pv_buses)

        if cfg.pv_vm_mode == "fixed":
            # Fester Wert mit kleiner Streuung
            vm_values = np.array([
                float(np.clip(
                    cfg.pv_vm_pu + self._rng.normal(0, cfg.pv_vm_std_pu),
                    0.95, 1.10
                ))
                for _ in range(n_pv)
            ])
            return vm_values

        # ── "natural" Modus: Erst PQ-Lastfluss, dann Offset ──
        try:
            pp.runpp(net, algorithm="nr", tolerance_mva=1e-6, max_iteration=50)
            if net.converged:
                # Natürliche Spannungen an PV-Bussen ablesen
                vm_natural = net.res_bus["vm_pu"].values
                vm_values = np.array([
                    float(np.clip(
                        vm_natural[bus_ids[pv_bus]]
                        + cfg.pv_vm_offset_pu
                        + self._rng.normal(0, cfg.pv_vm_std_pu),
                        0.95, 1.08
                    ))
                    for pv_bus in pv_buses
                ])
                return vm_values
        except Exception:
            pass

        # Fallback: Slack-Spannung mit minimalem Offset
        vm_values = np.array([
            float(np.clip(
                cfg.v_slack_pu + cfg.pv_vm_offset_pu
                + self._rng.normal(0, cfg.pv_vm_std_pu),
                0.95, 1.08
            ))
            for _ in range(n_pv)
        ])
        return vm_values

    # ──────────────────────────────────────────────────────────────
    #  PV-Platzierung
    # ──────────────────────────────────────────────────────────────

    def _select_pv_buses(self, graph: nx.Graph) -> list[int]:
        """Wählt PV-Bus-Indizes aus (Bus 0 = Slack ist ausgeschlossen)."""
        cfg = self.config
        n = cfg.nodes
        n_pv = min(cfg.n_pv, n - 2)

        if n_pv == 0:
            return []

        candidates = list(range(1, n))

        if cfg.pv_placement == "uniform":
            distances = nx.single_source_shortest_path_length(graph, 0)
            sorted_by_dist = sorted(candidates, key=lambda x: distances.get(x, 0))
            step = max(1, len(sorted_by_dist) // (n_pv + 1))
            indices = [(i + 1) * step for i in range(n_pv)]
            indices = [min(idx, len(sorted_by_dist) - 1) for idx in indices]
            selected = [sorted_by_dist[idx] for idx in indices]

        elif cfg.pv_placement == "end":
            leaves = [
                node for node in graph.nodes()
                if graph.degree(node) == 1 and node != 0
            ]
            if len(leaves) >= n_pv:
                selected = sorted(self._rng.choice(leaves, n_pv, replace=False))
            else:
                remaining = [c for c in candidates if c not in leaves]
                extra = list(self._rng.choice(remaining, n_pv - len(leaves), replace=False))
                selected = sorted(list(leaves) + extra)

        elif cfg.pv_placement == "mid":
            distances = nx.single_source_shortest_path_length(graph, 0)
            max_dist = max(distances.values())
            mid_range = (max_dist * 0.3, max_dist * 0.7)
            mid_candidates = [
                c for c in candidates
                if mid_range[0] <= distances.get(c, 0) <= mid_range[1]
            ]
            if len(mid_candidates) >= n_pv:
                selected = sorted(self._rng.choice(mid_candidates, n_pv, replace=False))
            else:
                selected = sorted(self._rng.choice(candidates, n_pv, replace=False))

        elif cfg.pv_placement == "random":
            selected = sorted(self._rng.choice(candidates, n_pv, replace=False))

        else:
            raise ValueError(f"Unbekannte PV-Platzierung: {cfg.pv_placement}")

        return [int(s) for s in selected]

    # ──────────────────────────────────────────────────────────────
    #  Validierung
    # ──────────────────────────────────────────────────────────────

    def _validate(self, net: pp.pandapowerNet) -> None:
        """Prüft ob NR konvergiert. Versucht mehrere Strategien."""
        strategies = [
            {"algorithm": "nr", "tolerance_mva": 1e-6, "max_iteration": 100},
            {"algorithm": "nr", "tolerance_mva": 1e-5, "max_iteration": 200},
            {"algorithm": "nr", "tolerance_mva": 1e-4, "max_iteration": 300,
             "init": "dc"},
        ]

        for strat in strategies:
            try:
                pp.runpp(net, **strat)
                if net.converged:
                    return
            except Exception:
                continue

        raise RuntimeError(
            f"NR konvergiert nicht für Salazar-Netz "
            f"(nodes={self.config.nodes}, child={self.config.child}, "
            f"n_pv={self.config.n_pv}). "
            f"R/X={self.config.rx_ratio:.1f} → Q↔V-Kopplung zu schwach "
            f"für gewählte PV-Parameter."
        )


# ══════════════════════════════════════════════════════════════════════
#  Convenience-Funktion
# ══════════════════════════════════════════════════════════════════════

def create_salazar_network(
    nodes: int = 34,
    child: int = 3,
    load_factor: float = 2.0,
    line_factor: float = 3.0,
    n_pv: int = 0,
    pv_placement: str = "uniform",
    pv_p_mw: float = 0.05,
    pv_vm_mode: str = "natural",
    pv_vm_pu: float = 1.00,
    pv_vm_offset_pu: float = 0.005,
    pv_vm_std_pu: float = 0.002,
    vn_kv: float = 11.0,
    v_slack_pu: float = 1.00,
    seed: int | None = 42,
) -> pp.pandapowerNet:
    """
    Erzeugt ein Netz nach Salazar et al. (2024) mit optionalen PV-Knoten.

    Parameters
    ----------
    nodes : int
        Gesamtzahl Busse (inkl. Slack). Paper testet 9 bis 5000.
    child : int
        Verzweigungsfaktor des k-ary tree.
    load_factor : float
        Lastskalierung. μ_P = 50 * load_factor [kW]. Paper default: 2.
    line_factor : float
        Impedanzskalierung. R_eff = 0.3144 / line_factor. Paper default: 3.
    n_pv : int
        Anzahl PV-Knoten (Paper: 0).
    pv_placement : str
        "uniform", "end", "mid", "random".
    pv_p_mw : float
        Wirkleistung pro DG [MW]. Default: 0.05 (50 kW).
    pv_vm_mode : str
        "natural": vm_pu aus PQ-Lastfluss + Offset (empfohlen bei R/X>>1)
        "fixed": Fester Wert.
    pv_vm_pu : float
        Fester Sollspannungswert (nur bei pv_vm_mode="fixed").
    pv_vm_offset_pu : float
        Spannungsoffset über natürlichem Profil (bei pv_vm_mode="natural").
    pv_vm_std_pu : float
        Streuung der Sollspannung.
    vn_kv : float
        Nennspannung [kV]. Paper: 11 kV.
    v_slack_pu : float
        Slack-Spannung [p.u.]. Paper: 1.00.
    seed : int | None
        Zufallssamen.

    Returns
    -------
    pandapowerNet
    """
    config = SalazarNetworkConfig(
        nodes=nodes,
        child=child,
        load_factor=load_factor,
        line_factor=line_factor,
        n_pv=n_pv,
        pv_placement=pv_placement,
        pv_p_mw=pv_p_mw,
        pv_vm_mode=pv_vm_mode,
        pv_vm_pu=pv_vm_pu,
        pv_vm_offset_pu=pv_vm_offset_pu,
        pv_vm_std_pu=pv_vm_std_pu,
        vn_kv=vn_kv,
        v_slack_pu=v_slack_pu,
        seed=seed,
    )
    gen = SalazarNetworkGenerator(config)
    return gen.generate()


# ══════════════════════════════════════════════════════════════════════
#  Vordefinierte Netze — Paper-Reproduktion (OHNE PV)
# ══════════════════════════════════════════════════════════════════════

def create_salazar_34bus() -> pp.pandapowerNet:
    """Paper: 34-Bus Basis-Testnetz."""
    return create_salazar_network(nodes=34, child=3, seed=42)


def create_salazar_100bus() -> pp.pandapowerNet:
    """Paper: 100-Bus k-ary tree."""
    return create_salazar_network(nodes=100, child=3, seed=100)


def create_salazar_500bus() -> pp.pandapowerNet:
    """Paper: 500-Bus k-ary tree."""
    return create_salazar_network(nodes=500, child=3, seed=500)


def create_salazar_1000bus() -> pp.pandapowerNet:
    """Paper: 1000-Bus k-ary tree."""
    return create_salazar_network(nodes=1000, child=3, seed=1000)


def create_salazar_2000bus() -> pp.pandapowerNet:
    """Paper: 2000-Bus k-ary tree."""
    return create_salazar_network(nodes=2000, child=3, seed=2000)


def create_salazar_5000bus() -> pp.pandapowerNet:
    """Paper: 5000-Bus k-ary tree."""
    return create_salazar_network(nodes=5000, child=3, seed=5000)


# ══════════════════════════════════════════════════════════════════════
#  Vordefinierte Netze — Erweitert mit PV-Knoten
# ══════════════════════════════════════════════════════════════════════

def create_salazar_34bus_3pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=34, child=3, n_pv=3, seed=42)


def create_salazar_100bus_5pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, n_pv=5, seed=100)


def create_salazar_100bus_10pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, n_pv=10, seed=101)


def create_salazar_200bus_5pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=200, child=3, n_pv=5, seed=200)


def create_salazar_200bus_10pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=200, child=3, n_pv=10, seed=201)


def create_salazar_200bus_20pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=200, child=3, n_pv=20, seed=202)


def create_salazar_500bus_5pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=500, child=3, n_pv=5, seed=500)


def create_salazar_500bus_10pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=500, child=3, n_pv=10, seed=501)


def create_salazar_500bus_25pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=500, child=3, n_pv=25, seed=502)


def create_salazar_1000bus_10pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=1000, child=3, n_pv=10, seed=1000)


def create_salazar_1000bus_25pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=1000, child=3, n_pv=25, seed=1001)


def create_salazar_1000bus_50pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=1000, child=3, n_pv=50, seed=1002)


# ── Variationen: Verzweigungsfaktor ──

def create_salazar_100bus_c2_5pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=2, n_pv=5, seed=110)


def create_salazar_100bus_c5_5pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=5, n_pv=5, seed=111)


# ── Variationen: line_factor ──

def create_salazar_100bus_lf1_5pv() -> pp.pandapowerNet:
    """line_factor=1 → hohe Impedanz, R/X bleibt 5.82."""
    return create_salazar_network(nodes=100, child=3, line_factor=1.0, n_pv=5, seed=120)


def create_salazar_100bus_lf5_5pv() -> pp.pandapowerNet:
    """line_factor=5 → niedrige Impedanz."""
    return create_salazar_network(nodes=100, child=3, line_factor=5.0, n_pv=5, seed=121)


# ── Variationen: load_factor ──

def create_salazar_100bus_loadhigh_5pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, load_factor=4.0, n_pv=5, seed=130)


def create_salazar_100bus_loadlow_5pv() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, load_factor=1.0, n_pv=5, seed=131)


# ── Variationen: PV-Platzierung ──

def create_salazar_100bus_5pv_end() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, n_pv=5, pv_placement="end", seed=140)


def create_salazar_100bus_5pv_random() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, n_pv=5, pv_placement="random", seed=141)


def create_salazar_100bus_5pv_mid() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, n_pv=5, pv_placement="mid", seed=142)


# ── Variation: PV mit fixem vm_pu (zeigt Konvergenzproblem bei R/X>>1) ──

def create_salazar_100bus_5pv_fixed_vm() -> pp.pandapowerNet:
    """PV mit festem vm_pu=1.01 — kann bei R/X=5.82 problematisch sein."""
    return create_salazar_network(
        nodes=100, child=3, n_pv=5,
        pv_vm_mode="fixed", pv_vm_pu=1.01, seed=150
    )


# ══════════════════════════════════════════════════════════════════════
#  TEST_NETWORKS Dictionary
# ══════════════════════════════════════════════════════════════════════

SALAZAR_TEST_NETWORKS: dict[str, dict] = {
    # ════════════════════════════════════════════════════════════
    #  Paper-Reproduktion (OHNE PV)
    # ════════════════════════════════════════════════════════════
    "salazar_34bus": {
        "constructor": create_salazar_34bus,
        "description": "Salazar Paper: 34-Bus, child=3 (default)",
        "n_pv": 0,
        "category": "salazar_paper",
    },
    "salazar_100bus": {
        "constructor": create_salazar_100bus,
        "description": "Salazar Paper: 100-Bus, child=3",
        "n_pv": 0,
        "category": "salazar_paper",
    },
    "salazar_500bus": {
        "constructor": create_salazar_500bus,
        "description": "Salazar Paper: 500-Bus, child=3",
        "n_pv": 0,
        "category": "salazar_paper",
    },
    "salazar_1000bus": {
        "constructor": create_salazar_1000bus,
        "description": "Salazar Paper: 1000-Bus, child=3",
        "n_pv": 0,
        "category": "salazar_paper",
    },

    # ════════════════════════════════════════════════════════════
    #  MIT PV — Kleine Netze
    # ════════════════════════════════════════════════════════════
    "salazar_34bus_3pv": {
        "constructor": create_salazar_34bus_3pv,
        "description": "Salazar 34-Bus + 3 PV (natural vm)",
        "n_pv": 3,
        "category": "salazar_pv_small",
    },
    "salazar_100bus_5pv": {
        "constructor": create_salazar_100bus_5pv,
        "description": "Salazar 100-Bus + 5 PV",
        "n_pv": 5,
        "category": "salazar_pv_small",
    },
    "salazar_100bus_10pv": {
        "constructor": create_salazar_100bus_10pv,
        "description": "Salazar 100-Bus + 10 PV",
        "n_pv": 10,
        "category": "salazar_pv_small",
    },

    # ════════════════════════════════════════════════════════════
    #  MIT PV — Mittlere Netze
    # ════════════════════════════════════════════════════════════
    "salazar_200bus_5pv": {
        "constructor": create_salazar_200bus_5pv,
        "description": "Salazar 200-Bus + 5 PV",
        "n_pv": 5,
        "category": "salazar_pv_medium",
    },
    "salazar_200bus_10pv": {
        "constructor": create_salazar_200bus_10pv,
        "description": "Salazar 200-Bus + 10 PV",
        "n_pv": 10,
        "category": "salazar_pv_medium",
    },
    "salazar_200bus_20pv": {
        "constructor": create_salazar_200bus_20pv,
        "description": "Salazar 200-Bus + 20 PV",
        "n_pv": 20,
        "category": "salazar_pv_medium",
    },

    # ════════════════════════════════════════════════════════════
    #  MIT PV — Große Netze
    # ════════════════════════════════════════════════════════════
    "salazar_500bus_5pv": {
        "constructor": create_salazar_500bus_5pv,
        "description": "Salazar 500-Bus + 5 PV",
        "n_pv": 5,
        "category": "salazar_pv_large",
    },
    "salazar_500bus_10pv": {
        "constructor": create_salazar_500bus_10pv,
        "description": "Salazar 500-Bus + 10 PV",
        "n_pv": 10,
        "category": "salazar_pv_large",
    },
    "salazar_500bus_25pv": {
        "constructor": create_salazar_500bus_25pv,
        "description": "Salazar 500-Bus + 25 PV",
        "n_pv": 25,
        "category": "salazar_pv_large",
    },
    "salazar_1000bus_10pv": {
        "constructor": create_salazar_1000bus_10pv,
        "description": "Salazar 1000-Bus + 10 PV",
        "n_pv": 10,
        "category": "salazar_pv_large",
    },
    "salazar_1000bus_25pv": {
        "constructor": create_salazar_1000bus_25pv,
        "description": "Salazar 1000-Bus + 25 PV",
        "n_pv": 25,
        "category": "salazar_pv_large",
    },
    "salazar_1000bus_50pv": {
        "constructor": create_salazar_1000bus_50pv,
        "description": "Salazar 1000-Bus + 50 PV",
        "n_pv": 50,
        "category": "salazar_pv_large",
    },

    # ════════════════════════════════════════════════════════════
    #  Variationen
    # ════════════════════════════════════════════════════════════
    "salazar_100bus_c2_5pv": {
        "constructor": create_salazar_100bus_c2_5pv,
        "description": "Salazar 100-Bus, Binärbaum + 5 PV",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_c5_5pv": {
        "constructor": create_salazar_100bus_c5_5pv,
        "description": "Salazar 100-Bus, child=5 + 5 PV",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_lf1_5pv": {
        "constructor": create_salazar_100bus_lf1_5pv,
        "description": "Salazar 100-Bus, line_factor=1 (hohe Z) + 5 PV",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_lf5_5pv": {
        "constructor": create_salazar_100bus_lf5_5pv,
        "description": "Salazar 100-Bus, line_factor=5 (niedrige Z) + 5 PV",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_loadhigh_5pv": {
        "constructor": create_salazar_100bus_loadhigh_5pv,
        "description": "Salazar 100-Bus, load_factor=4 + 5 PV",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_loadlow_5pv": {
        "constructor": create_salazar_100bus_loadlow_5pv,
        "description": "Salazar 100-Bus, load_factor=1 + 5 PV",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_5pv_end": {
        "constructor": create_salazar_100bus_5pv_end,
        "description": "Salazar 100-Bus + 5 PV an Blattknoten",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_5pv_random": {
        "constructor": create_salazar_100bus_5pv_random,
        "description": "Salazar 100-Bus + 5 PV zufällig",
        "n_pv": 5,
        "category": "salazar_variation",
    },
    "salazar_100bus_5pv_mid": {
        "constructor": create_salazar_100bus_5pv_mid,
        "description": "Salazar 100-Bus + 5 PV Netzmitte",
        "n_pv": 5,
        "category": "salazar_variation",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Filter-Funktionen
# ══════════════════════════════════════════════════════════════════════

def get_salazar_paper_networks() -> dict[str, dict]:
    return {k: v for k, v in SALAZAR_TEST_NETWORKS.items()
            if v["category"] == "salazar_paper"}


def get_salazar_pv_networks() -> dict[str, dict]:
    return {k: v for k, v in SALAZAR_TEST_NETWORKS.items()
            if v["n_pv"] > 0}


def get_salazar_small_pv_networks() -> dict[str, dict]:
    return {k: v for k, v in SALAZAR_TEST_NETWORKS.items()
            if v["category"] == "salazar_pv_small"}


def get_salazar_large_pv_networks() -> dict[str, dict]:
    return {k: v for k, v in SALAZAR_TEST_NETWORKS.items()
            if v["category"] == "salazar_pv_large"}


def get_salazar_all_networks() -> dict[str, dict]:
    return SALAZAR_TEST_NETWORKS


# ══════════════════════════════════════════════════════════════════════
#  SYSTEMATISCHE TESTMATRIX: 10 Größen × verschiedene PV-Ratios
# ══════════════════════════════════════════════════════════════════════

# Netzgrößen (10 Stufen)
_SIZES = [20, 40, 75, 120, 200, 350, 500, 750, 1000, 1500]

# PV-Ratio-Stufen (PV/Total)
_PV_RATIOS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def _create_salazar_size_ratio(nodes: int, pv_ratio: float, seed: int):
    """Erzeugt ein Salazar-Netz mit gegebener Größe und PV-Ratio."""
    n_pv = max(1, int(round(nodes * pv_ratio)))
    # Sicherstellen: mindestens 2 PQ-Knoten bleiben
    n_pv = min(n_pv, nodes - 3)
    return create_salazar_network(nodes=nodes, child=3, n_pv=n_pv, seed=seed)


# ── Generierte Konstruktoren für die Testmatrix ──

def create_salazar_20bus_r000():
    return _create_salazar_size_ratio(20, 0.0, 2000)

def create_salazar_20bus_r005():
    return _create_salazar_size_ratio(20, 0.05, 2000)

def create_salazar_20bus_r010():
    return _create_salazar_size_ratio(20, 0.10, 2001)

def create_salazar_20bus_r020():
    return _create_salazar_size_ratio(20, 0.20, 2002)

def create_salazar_40bus_r000():
    return _create_salazar_size_ratio(40, 0.0, 2010)

def create_salazar_40bus_r005():
    return _create_salazar_size_ratio(40, 0.05, 2010)

def create_salazar_40bus_r010():
    return _create_salazar_size_ratio(40, 0.10, 2011)

def create_salazar_40bus_r020():
    return _create_salazar_size_ratio(40, 0.20, 2012)

def create_salazar_75bus_r000():
    return _create_salazar_size_ratio(75, 0.0, 2020)

def create_salazar_75bus_r005():
    return _create_salazar_size_ratio(75, 0.05, 2020)

def create_salazar_75bus_r010():
    return _create_salazar_size_ratio(75, 0.10, 2021)

def create_salazar_75bus_r020():
    return _create_salazar_size_ratio(75, 0.20, 2022)

def create_salazar_75bus_r030():
    return _create_salazar_size_ratio(75, 0.30, 2023)

def create_salazar_120bus_r000():
    return _create_salazar_size_ratio(120, 0.0, 2030)

def create_salazar_120bus_r005():
    return _create_salazar_size_ratio(120, 0.05, 2030)

def create_salazar_120bus_r010():
    return _create_salazar_size_ratio(120, 0.10, 2031)

def create_salazar_120bus_r020():
    return _create_salazar_size_ratio(120, 0.20, 2032)

def create_salazar_120bus_r030():
    return _create_salazar_size_ratio(120, 0.30, 2033)

def create_salazar_200bus_r000():
    return _create_salazar_size_ratio(200, 0.0, 2040)

def create_salazar_200bus_r002():
    return _create_salazar_size_ratio(200, 0.02, 2040)

def create_salazar_200bus_r005():
    return _create_salazar_size_ratio(200, 0.05, 2041)

def create_salazar_200bus_r010():
    return _create_salazar_size_ratio(200, 0.10, 2042)

def create_salazar_200bus_r020():
    return _create_salazar_size_ratio(200, 0.20, 2043)

def create_salazar_200bus_r030():
    return _create_salazar_size_ratio(200, 0.30, 2044)

def create_salazar_350bus_r000():
    return _create_salazar_size_ratio(350, 0.0, 2050)

def create_salazar_350bus_r002():
    return _create_salazar_size_ratio(350, 0.02, 2050)

def create_salazar_350bus_r005():
    return _create_salazar_size_ratio(350, 0.05, 2051)

def create_salazar_350bus_r010():
    return _create_salazar_size_ratio(350, 0.10, 2052)

def create_salazar_350bus_r020():
    return _create_salazar_size_ratio(350, 0.20, 2053)

def create_salazar_500bus_r000():
    return _create_salazar_size_ratio(500, 0.0, 2060)

def create_salazar_500bus_r002():
    return _create_salazar_size_ratio(500, 0.02, 2060)

def create_salazar_500bus_r005():
    return _create_salazar_size_ratio(500, 0.05, 2061)

def create_salazar_500bus_r010():
    return _create_salazar_size_ratio(500, 0.10, 2062)

def create_salazar_500bus_r020():
    return _create_salazar_size_ratio(500, 0.20, 2063)

def create_salazar_750bus_r000():
    return _create_salazar_size_ratio(750, 0.0, 2070)

def create_salazar_750bus_r002():
    return _create_salazar_size_ratio(750, 0.02, 2070)

def create_salazar_750bus_r005():
    return _create_salazar_size_ratio(750, 0.05, 2071)

def create_salazar_750bus_r010():
    return _create_salazar_size_ratio(750, 0.10, 2072)

def create_salazar_1000bus_r000():
    return _create_salazar_size_ratio(1000, 0.0, 2080)

def create_salazar_1000bus_r002():
    return _create_salazar_size_ratio(1000, 0.02, 2080)

def create_salazar_1000bus_r005():
    return _create_salazar_size_ratio(1000, 0.05, 2081)

def create_salazar_1000bus_r010():
    return _create_salazar_size_ratio(1000, 0.10, 2082)

def create_salazar_1500bus_r000():
    return _create_salazar_size_ratio(1500, 0.0, 2090)

def create_salazar_1500bus_r002():
    return _create_salazar_size_ratio(1500, 0.02, 2090)

def create_salazar_1500bus_r005():
    return _create_salazar_size_ratio(1500, 0.05, 2091)

def create_salazar_1500bus_r010():
    return _create_salazar_size_ratio(1500, 0.10, 2092)


# ══════════════════════════════════════════════════════════════════════
#  SALAZAR_SCALING_NETWORKS — Systematische Testmatrix
# ══════════════════════════════════════════════════════════════════════

SALAZAR_SCALING_NETWORKS: dict[str, dict] = {
    # ── 20 Bus ──
    "sz_20_r000": {
        "constructor": create_salazar_20bus_r000,
        "description": "Salazar 20-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 20, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_20_r005": {
        "constructor": create_salazar_20bus_r005,
        "description": "Salazar 20-Bus, PV/Total=5%",
        "n_pv": 1, "n_bus_total": 20, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_20_r010": {
        "constructor": create_salazar_20bus_r010,
        "description": "Salazar 20-Bus, PV/Total=10%",
        "n_pv": 2, "n_bus_total": 20, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    "sz_20_r020": {
        "constructor": create_salazar_20bus_r020,
        "description": "Salazar 20-Bus, PV/Total=20%",
        "n_pv": 4, "n_bus_total": 20, "pv_ratio": 0.20,
        "category": "salazar_scaling",
    },
    # ── 40 Bus ──
    "sz_40_r000": {
        "constructor": create_salazar_40bus_r000,
        "description": "Salazar 40-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 40, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_40_r005": {
        "constructor": create_salazar_40bus_r005,
        "description": "Salazar 40-Bus, PV/Total=5%",
        "n_pv": 2, "n_bus_total": 40, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_40_r010": {
        "constructor": create_salazar_40bus_r010,
        "description": "Salazar 40-Bus, PV/Total=10%",
        "n_pv": 4, "n_bus_total": 40, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    "sz_40_r020": {
        "constructor": create_salazar_40bus_r020,
        "description": "Salazar 40-Bus, PV/Total=20%",
        "n_pv": 8, "n_bus_total": 40, "pv_ratio": 0.20,
        "category": "salazar_scaling",
    },
    # ── 75 Bus ──
    "sz_75_r000": {
        "constructor": create_salazar_75bus_r000,
        "description": "Salazar 75-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 75, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_75_r005": {
        "constructor": create_salazar_75bus_r005,
        "description": "Salazar 75-Bus, PV/Total=5%",
        "n_pv": 4, "n_bus_total": 75, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_75_r010": {
        "constructor": create_salazar_75bus_r010,
        "description": "Salazar 75-Bus, PV/Total=10%",
        "n_pv": 8, "n_bus_total": 75, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    "sz_75_r020": {
        "constructor": create_salazar_75bus_r020,
        "description": "Salazar 75-Bus, PV/Total=20%",
        "n_pv": 15, "n_bus_total": 75, "pv_ratio": 0.20,
        "category": "salazar_scaling",
    },
    "sz_75_r030": {
        "constructor": create_salazar_75bus_r030,
        "description": "Salazar 75-Bus, PV/Total=30%",
        "n_pv": 23, "n_bus_total": 75, "pv_ratio": 0.30,
        "category": "salazar_scaling",
    },
    # ── 120 Bus ──
    "sz_120_r000": {
        "constructor": create_salazar_120bus_r000,
        "description": "Salazar 120-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 120, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_120_r005": {
        "constructor": create_salazar_120bus_r005,
        "description": "Salazar 120-Bus, PV/Total=5%",
        "n_pv": 6, "n_bus_total": 120, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_120_r010": {
        "constructor": create_salazar_120bus_r010,
        "description": "Salazar 120-Bus, PV/Total=10%",
        "n_pv": 12, "n_bus_total": 120, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    "sz_120_r020": {
        "constructor": create_salazar_120bus_r020,
        "description": "Salazar 120-Bus, PV/Total=20%",
        "n_pv": 24, "n_bus_total": 120, "pv_ratio": 0.20,
        "category": "salazar_scaling",
    },
    "sz_120_r030": {
        "constructor": create_salazar_120bus_r030,
        "description": "Salazar 120-Bus, PV/Total=30%",
        "n_pv": 36, "n_bus_total": 120, "pv_ratio": 0.30,
        "category": "salazar_scaling",
    },
    # ── 200 Bus ──
    "sz_200_r000": {
        "constructor": create_salazar_200bus_r000,
        "description": "Salazar 200-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 200, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_200_r002": {
        "constructor": create_salazar_200bus_r002,
        "description": "Salazar 200-Bus, PV/Total=2%",
        "n_pv": 4, "n_bus_total": 200, "pv_ratio": 0.02,
        "category": "salazar_scaling",
    },
    "sz_200_r005": {
        "constructor": create_salazar_200bus_r005,
        "description": "Salazar 200-Bus, PV/Total=5%",
        "n_pv": 10, "n_bus_total": 200, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_200_r010": {
        "constructor": create_salazar_200bus_r010,
        "description": "Salazar 200-Bus, PV/Total=10%",
        "n_pv": 20, "n_bus_total": 200, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    "sz_200_r020": {
        "constructor": create_salazar_200bus_r020,
        "description": "Salazar 200-Bus, PV/Total=20%",
        "n_pv": 40, "n_bus_total": 200, "pv_ratio": 0.20,
        "category": "salazar_scaling",
    },
    "sz_200_r030": {
        "constructor": create_salazar_200bus_r030,
        "description": "Salazar 200-Bus, PV/Total=30%",
        "n_pv": 60, "n_bus_total": 200, "pv_ratio": 0.30,
        "category": "salazar_scaling",
    },
    # ── 350 Bus ──
    "sz_350_r000": {
        "constructor": create_salazar_350bus_r000,
        "description": "Salazar 350-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 350, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_350_r002": {
        "constructor": create_salazar_350bus_r002,
        "description": "Salazar 350-Bus, PV/Total=2%",
        "n_pv": 7, "n_bus_total": 350, "pv_ratio": 0.02,
        "category": "salazar_scaling",
    },
    "sz_350_r005": {
        "constructor": create_salazar_350bus_r005,
        "description": "Salazar 350-Bus, PV/Total=5%",
        "n_pv": 18, "n_bus_total": 350, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_350_r010": {
        "constructor": create_salazar_350bus_r010,
        "description": "Salazar 350-Bus, PV/Total=10%",
        "n_pv": 35, "n_bus_total": 350, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    "sz_350_r020": {
        "constructor": create_salazar_350bus_r020,
        "description": "Salazar 350-Bus, PV/Total=20%",
        "n_pv": 70, "n_bus_total": 350, "pv_ratio": 0.20,
        "category": "salazar_scaling",
    },
    # ── 500 Bus ──
    "sz_500_r000": {
        "constructor": create_salazar_500bus_r000,
        "description": "Salazar 500-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 500, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_500_r002": {
        "constructor": create_salazar_500bus_r002,
        "description": "Salazar 500-Bus, PV/Total=2%",
        "n_pv": 10, "n_bus_total": 500, "pv_ratio": 0.02,
        "category": "salazar_scaling",
    },
    "sz_500_r005": {
        "constructor": create_salazar_500bus_r005,
        "description": "Salazar 500-Bus, PV/Total=5%",
        "n_pv": 25, "n_bus_total": 500, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_500_r010": {
        "constructor": create_salazar_500bus_r010,
        "description": "Salazar 500-Bus, PV/Total=10%",
        "n_pv": 50, "n_bus_total": 500, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    "sz_500_r020": {
        "constructor": create_salazar_500bus_r020,
        "description": "Salazar 500-Bus, PV/Total=20%",
        "n_pv": 100, "n_bus_total": 500, "pv_ratio": 0.20,
        "category": "salazar_scaling",
    },
    # ── 750 Bus ──
    "sz_750_r000": {
        "constructor": create_salazar_750bus_r000,
        "description": "Salazar 750-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 750, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_750_r002": {
        "constructor": create_salazar_750bus_r002,
        "description": "Salazar 750-Bus, PV/Total=2%",
        "n_pv": 15, "n_bus_total": 750, "pv_ratio": 0.02,
        "category": "salazar_scaling",
    },
    "sz_750_r005": {
        "constructor": create_salazar_750bus_r005,
        "description": "Salazar 750-Bus, PV/Total=5%",
        "n_pv": 38, "n_bus_total": 750, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_750_r010": {
        "constructor": create_salazar_750bus_r010,
        "description": "Salazar 750-Bus, PV/Total=10%",
        "n_pv": 75, "n_bus_total": 750, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    # ── 1000 Bus ──
    "sz_1000_r000": {
        "constructor": create_salazar_1000bus_r000,
        "description": "Salazar 1000-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 1000, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_1000_r002": {
        "constructor": create_salazar_1000bus_r002,
        "description": "Salazar 1000-Bus, PV/Total=2%",
        "n_pv": 20, "n_bus_total": 1000, "pv_ratio": 0.02,
        "category": "salazar_scaling",
    },
    "sz_1000_r005": {
        "constructor": create_salazar_1000bus_r005,
        "description": "Salazar 1000-Bus, PV/Total=5%",
        "n_pv": 50, "n_bus_total": 1000, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_1000_r010": {
        "constructor": create_salazar_1000bus_r010,
        "description": "Salazar 1000-Bus, PV/Total=10%",
        "n_pv": 100, "n_bus_total": 1000, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
    # ── 1500 Bus ──
    "sz_1500_r000": {
        "constructor": create_salazar_1500bus_r000,
        "description": "Salazar 1500-Bus, PV/Total=0%",
        "n_pv": 0, "n_bus_total": 1500, "pv_ratio": 0.0,
        "category": "salazar_scaling",
    },
    "sz_1500_r002": {
        "constructor": create_salazar_1500bus_r002,
        "description": "Salazar 1500-Bus, PV/Total=2%",
        "n_pv": 30, "n_bus_total": 1500, "pv_ratio": 0.02,
        "category": "salazar_scaling",
    },
    "sz_1500_r005": {
        "constructor": create_salazar_1500bus_r005,
        "description": "Salazar 1500-Bus, PV/Total=5%",
        "n_pv": 75, "n_bus_total": 1500, "pv_ratio": 0.05,
        "category": "salazar_scaling",
    },
    "sz_1500_r010": {
        "constructor": create_salazar_1500bus_r010,
        "description": "Salazar 1500-Bus, PV/Total=10%",
        "n_pv": 150, "n_bus_total": 1500, "pv_ratio": 0.10,
        "category": "salazar_scaling",
    },
}


def get_salazar_scaling_networks() -> dict[str, dict]:
    """Systematische Testmatrix: 10 Größen × verschiedene PV-Ratios."""
    return SALAZAR_SCALING_NETWORKS


# ══════════════════════════════════════════════════════════════════════
#  PQ-ONLY NETZE — Paper-Reproduktion für Baseline-Performance
#  Keine PV-Knoten, nur Slack + PQ. Für reinen TPF vs. NR Vergleich.
#  Größen: 9 bis 5000 (wie im Paper, Fig. 5)
# ══════════════════════════════════════════════════════════════════════

_PQ_SIZES = [9, 20, 50, 100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000]


def create_salazar_pq_9bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=9, child=3, n_pv=0, seed=3009)

def create_salazar_pq_20bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=20, child=3, n_pv=0, seed=3020)

def create_salazar_pq_50bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=50, child=3, n_pv=0, seed=3050)

def create_salazar_pq_100bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=100, child=3, n_pv=0, seed=3100)

def create_salazar_pq_200bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=200, child=3, n_pv=0, seed=3200)

def create_salazar_pq_500bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=500, child=3, n_pv=0, seed=3500)

def create_salazar_pq_750bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=750, child=3, n_pv=0, seed=3750)

def create_salazar_pq_1000bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=1000, child=3, n_pv=0, seed=4000)

def create_salazar_pq_1500bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=1500, child=3, n_pv=0, seed=4500)

def create_salazar_pq_2000bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=2000, child=3, n_pv=0, seed=5000)

def create_salazar_pq_3000bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=3000, child=3, n_pv=0, seed=6000)

def create_salazar_pq_5000bus() -> pp.pandapowerNet:
    return create_salazar_network(nodes=5000, child=3, n_pv=0, seed=7000)


# ── Variationen: Verschiedene load_factor (bei 500 Bus) ──

def create_salazar_pq_500bus_lf1() -> pp.pandapowerNet:
    """500-Bus PQ-only, load_factor=1 (leichte Last)."""
    return create_salazar_network(nodes=500, child=3, load_factor=1.0, n_pv=0, seed=3501)

def create_salazar_pq_500bus_lf3() -> pp.pandapowerNet:
    """500-Bus PQ-only, load_factor=3 (mittlere Last)."""
    return create_salazar_network(nodes=500, child=3, load_factor=3.0, n_pv=0, seed=3502)

def create_salazar_pq_500bus_lf5() -> pp.pandapowerNet:
    """500-Bus PQ-only, load_factor=5 (hohe Last, η nahe 1)."""
    return create_salazar_network(nodes=500, child=3, load_factor=5.0, n_pv=0, seed=3503)


# ── Variationen: Verschiedene line_factor (bei 500 Bus) ──

def create_salazar_pq_500bus_linef1() -> pp.pandapowerNet:
    """500-Bus PQ-only, line_factor=1 (hohe Impedanz)."""
    return create_salazar_network(nodes=500, child=3, line_factor=1.0, n_pv=0, seed=3504)

def create_salazar_pq_500bus_linef5() -> pp.pandapowerNet:
    """500-Bus PQ-only, line_factor=5 (niedrige Impedanz)."""
    return create_salazar_network(nodes=500, child=3, line_factor=5.0, n_pv=0, seed=3505)

def create_salazar_pq_500bus_linef10() -> pp.pandapowerNet:
    """500-Bus PQ-only, line_factor=10 (sehr niedrige Impedanz)."""
    return create_salazar_network(nodes=500, child=3, line_factor=10.0, n_pv=0, seed=3506)


# ── Variationen: Verschiedene child (Verzweigung) ──

def create_salazar_pq_500bus_c2() -> pp.pandapowerNet:
    """500-Bus PQ-only, Binärbaum (child=2, tief)."""
    return create_salazar_network(nodes=500, child=2, n_pv=0, seed=3507)

def create_salazar_pq_500bus_c5() -> pp.pandapowerNet:
    """500-Bus PQ-only, child=5 (breit, flach)."""
    return create_salazar_network(nodes=500, child=5, n_pv=0, seed=3508)

def create_salazar_pq_500bus_c10() -> pp.pandapowerNet:
    """500-Bus PQ-only, child=10 (sehr breit)."""
    return create_salazar_network(nodes=500, child=10, n_pv=0, seed=3509)


# ══════════════════════════════════════════════════════════════════════
#  SALAZAR_PQ_NETWORKS — Dictionary
# ══════════════════════════════════════════════════════════════════════

SALAZAR_PQ_NETWORKS: dict[str, dict] = {
    # ════════════════════════════════════════════════════════════
    #  Größenstaffel (wie Paper Fig. 5): 9 → 5000 Busse
    # ════════════════════════════════════════════════════════════
    "pq_9bus": {
        "constructor": create_salazar_pq_9bus,
        "description": "Salazar PQ-only: 9-Bus, child=3",
        "n_pv": 0, "n_bus_total": 9,
        "category": "salazar_pq_size",
    },
    "pq_20bus": {
        "constructor": create_salazar_pq_20bus,
        "description": "Salazar PQ-only: 20-Bus, child=3",
        "n_pv": 0, "n_bus_total": 20,
        "category": "salazar_pq_size",
    },
    "pq_50bus": {
        "constructor": create_salazar_pq_50bus,
        "description": "Salazar PQ-only: 50-Bus, child=3",
        "n_pv": 0, "n_bus_total": 50,
        "category": "salazar_pq_size",
    },
    "pq_100bus": {
        "constructor": create_salazar_pq_100bus,
        "description": "Salazar PQ-only: 100-Bus, child=3",
        "n_pv": 0, "n_bus_total": 100,
        "category": "salazar_pq_size",
    },
    "pq_200bus": {
        "constructor": create_salazar_pq_200bus,
        "description": "Salazar PQ-only: 200-Bus, child=3",
        "n_pv": 0, "n_bus_total": 200,
        "category": "salazar_pq_size",
    },
    "pq_500bus": {
        "constructor": create_salazar_pq_500bus,
        "description": "Salazar PQ-only: 500-Bus, child=3",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_size",
    },
    "pq_750bus": {
        "constructor": create_salazar_pq_750bus,
        "description": "Salazar PQ-only: 750-Bus, child=3",
        "n_pv": 0, "n_bus_total": 750,
        "category": "salazar_pq_size",
    },
    "pq_1000bus": {
        "constructor": create_salazar_pq_1000bus,
        "description": "Salazar PQ-only: 1000-Bus, child=3",
        "n_pv": 0, "n_bus_total": 1000,
        "category": "salazar_pq_size",
    },
    "pq_1500bus": {
        "constructor": create_salazar_pq_1500bus,
        "description": "Salazar PQ-only: 1500-Bus, child=3",
        "n_pv": 0, "n_bus_total": 1500,
        "category": "salazar_pq_size",
    },
    "pq_2000bus": {
        "constructor": create_salazar_pq_2000bus,
        "description": "Salazar PQ-only: 2000-Bus, child=3",
        "n_pv": 0, "n_bus_total": 2000,
        "category": "salazar_pq_size",
    },
    "pq_3000bus": {
        "constructor": create_salazar_pq_3000bus,
        "description": "Salazar PQ-only: 3000-Bus, child=3",
        "n_pv": 0, "n_bus_total": 3000,
        "category": "salazar_pq_size",
    },
    "pq_5000bus": {
        "constructor": create_salazar_pq_5000bus,
        "description": "Salazar PQ-only: 5000-Bus, child=3",
        "n_pv": 0, "n_bus_total": 5000,
        "category": "salazar_pq_size",
    },

    # ════════════════════════════════════════════════════════════
    #  Variationen bei 500 Bus: load_factor
    # ════════════════════════════════════════════════════════════
    "pq_500bus_lf1": {
        "constructor": create_salazar_pq_500bus_lf1,
        "description": "PQ-only 500-Bus, load_factor=1 (leicht)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_load",
    },
    "pq_500bus_lf3": {
        "constructor": create_salazar_pq_500bus_lf3,
        "description": "PQ-only 500-Bus, load_factor=3 (mittel)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_load",
    },
    "pq_500bus_lf5": {
        "constructor": create_salazar_pq_500bus_lf5,
        "description": "PQ-only 500-Bus, load_factor=5 (hoch)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_load",
    },

    # ════════════════════════════════════════════════════════════
    #  Variationen bei 500 Bus: line_factor (Impedanz)
    # ════════════════════════════════════════════════════════════
    "pq_500bus_linef1": {
        "constructor": create_salazar_pq_500bus_linef1,
        "description": "PQ-only 500-Bus, line_factor=1 (hohe Z)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_impedance",
    },
    "pq_500bus_linef5": {
        "constructor": create_salazar_pq_500bus_linef5,
        "description": "PQ-only 500-Bus, line_factor=5 (niedrige Z)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_impedance",
    },
    "pq_500bus_linef10": {
        "constructor": create_salazar_pq_500bus_linef10,
        "description": "PQ-only 500-Bus, line_factor=10 (sehr niedrig)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_impedance",
    },

    # ════════════════════════════════════════════════════════════
    #  Variationen bei 500 Bus: child (Topologie)
    # ════════════════════════════════════════════════════════════
    "pq_500bus_c2": {
        "constructor": create_salazar_pq_500bus_c2,
        "description": "PQ-only 500-Bus, Binärbaum (tief)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_topology",
    },
    "pq_500bus_c5": {
        "constructor": create_salazar_pq_500bus_c5,
        "description": "PQ-only 500-Bus, child=5 (breit)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_topology",
    },
    "pq_500bus_c10": {
        "constructor": create_salazar_pq_500bus_c10,
        "description": "PQ-only 500-Bus, child=10 (sehr breit)",
        "n_pv": 0, "n_bus_total": 500,
        "category": "salazar_pq_topology",
    },
}


# ══════════════════════════════════════════════════════════════════════
#  Filter-Funktionen für PQ-Netze
# ══════════════════════════════════════════════════════════════════════

def get_salazar_pq_networks() -> dict[str, dict]:
    """Alle PQ-only Netze (für Baseline-Performance ohne PV)."""
    return SALAZAR_PQ_NETWORKS


def get_salazar_pq_size_sweep() -> dict[str, dict]:
    """Nur die Größenstaffel 9→5000 (wie Paper Fig. 5)."""
    return {k: v for k, v in SALAZAR_PQ_NETWORKS.items()
            if v["category"] == "salazar_pq_size"}


def get_salazar_pq_variations() -> dict[str, dict]:
    """Alle Variationen (Last, Impedanz, Topologie) bei 500 Bus."""
    return {k: v for k, v in SALAZAR_PQ_NETWORKS.items()
            if v["category"] in ("salazar_pq_load", "salazar_pq_impedance", "salazar_pq_topology")}


# ══════════════════════════════════════════════════════════════════════
#  Standalone
# ══════════════════════════════════════════════════════════════════════

def main():
    import time

    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║  NETZGENERATOR NACH SALAZAR ET AL. (2024) — Erweitert um PV         ║")
    print("║  FIX: vm_pu aus natürlichem Spannungsprofil (R/X=5.82 Problem)      ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")

    print(f"\n  SALAZAR_TEST_NETWORKS: {len(SALAZAR_TEST_NETWORKS)} Netze")
    print(f"  R/X = {0.3144/0.054:.2f} (hoch → schwache Q↔V-Kopplung)")
    print(f"  pv_vm_mode = 'natural' (vm_pu aus PQ-Profil + Offset)")
    print(f"{'─'*75}")

    n_pass = 0
    n_fail = 0

    for name, info in SALAZAR_TEST_NETWORKS.items():
        t0 = time.perf_counter()
        try:
            net = info["constructor"]()
            t_gen = (time.perf_counter() - t0) * 1000

            ppc = net._ppc
            bus_types = ppc["bus"][:, 1].astype(int)
            n_bus = len(bus_types)
            n_pv = int(np.sum(bus_types == 2))
            n_pq = int(np.sum(bus_types == 1))
            nr_iter = ppc.get("iterations", "?")

            # Spannungsbereich
            vm = net.res_bus["vm_pu"].values
            vm_min = vm[1:].min()  # Ohne Slack
            vm_max = vm[1:].max()

            n_pass += 1
            print(f"  ✓ {name:<30} Busse={n_bus:<5} PV={n_pv:<3} PQ={n_pq:<5} "
                  f"NR={nr_iter} iter  V=[{vm_min:.4f},{vm_max:.4f}]  "
                  f"({t_gen:.0f} ms)")
        except Exception as e:
            n_fail += 1
            t_gen = (time.perf_counter() - t0) * 1000
            print(f"  ✗ {name:<30} FEHLER: {str(e)[:60]}  ({t_gen:.0f} ms)")

    print(f"\n{'─'*75}")
    print(f"  Ergebnis: {n_pass} PASS, {n_fail} FAIL")

    if n_fail == 0:
        print(f"\n  ✓ ALLE NETZE ERFOLGREICH GENERIERT UND VALIDIERT!")
    else:
        print(f"\n  ⚠ {n_fail} Netze fehlgeschlagen")

    print(f"{'═'*75}")


if __name__ == "__main__":
    main()
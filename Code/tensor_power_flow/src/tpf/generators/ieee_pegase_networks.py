# tensor_power_flow/src/tpf/generators/ieee_pegase_networks.py
"""
IEEE, PEGASE und RTE Standard-Testnetze
========================================

Sammlung von Standard-Power-Flow-Testnetzen aus pandapower.
Verwendet die Netze unverändert (as-is).

Verwendung:
    from tpf.generators.ieee_pegase_networks import (
        IEE_PEGASE_NETWORKS,
        get_ieee_networks,
        get_pegase_networks,
        get_rte_networks,
        get_large_networks,
        get_all_standard_networks,
    )

    # Einzelnes Netz:
    net = IEE_PEGASE_NETWORKS["case14"]["constructor"]()

    # Alle Netze iterieren:
    for name, info in IEE_PEGASE_NETWORKS.items():
        net = info["constructor"]()
"""

import pandapower as pp
import pandapower.networks as pn


# ══════════════════════════════════════════════════════════════════════
#  Constructor Functions — IEEE, PEGASE, RTE networks (as-is)
# ══════════════════════════════════════════════════════════════════════

def create_case4gs() -> pp.pandapowerNet:
    """4 Bus Gene/Load system."""
    return pn.case4gs()


def create_case5() -> pp.pandapowerNet:
    """Case 5 — 5 Bus system."""
    return pn.case5()


def create_case6ww() -> pp.pandapowerNet:
    """6 Bus Ward & Ward — 6 Bus system with special elements."""
    return pn.case6ww()


def create_case9() -> pp.pandapowerNet:
    """IEEE 9-Bus (3 Gen, 2 PV)."""
    return pn.case9()


def create_case14() -> pp.pandapowerNet:
    """IEEE 14-Bus standard test case."""
    return pn.case14()


def create_case_ieee30() -> pp.pandapowerNet:
    """IEEE 30-Bus standard test case (5 PV)."""
    return pn.case_ieee30()


def create_case33bw() -> pp.pandapowerNet:
    """IEEE 33-Bus Baran & Wu (radial distribution)."""
    return pn.case33bw()


def create_case39() -> pp.pandapowerNet:
    """IEEE 39-Bus New England (10 Gen)."""
    return pn.case39()


def create_case57() -> pp.pandapowerNet:
    """IEEE 57-Bus standard test case (6 PV)."""
    return pn.case57()


def create_case_ieee118() -> pp.pandapowerNet:
    """IEEE 118-Bus standard test case."""
    try:
        return pn.case_ieee118()
    except AttributeError:
        return pn.case118()


def create_case300() -> pp.pandapowerNet:
    """IEEE 300-Bus standard test case."""
    return pn.case300()


def create_case1354pegase() -> pp.pandapowerNet:
    """PEGASE 1354-Bus network."""
    return pn.case1354pegase()


def create_case1888rte() -> pp.pandapowerNet:
    """RTE French 1888-Bus network."""
    return pn.case1888rte()


def create_case2848rte() -> pp.pandapowerNet:
    """RTE French 2848-Bus network."""
    return pn.case2848rte()


def create_case2869pegase() -> pp.pandapowerNet:
    """PEGASE 2869-Bus network."""
    return pn.case2869pegase()


def create_case9241pegase() -> pp.pandapowerNet:
    """PEGASE 9241-Bus network (very large)."""
    return pn.case9241pegase()


# ══════════════════════════════════════════════════════════════════════
#  Helper to count PV nodes from a network
# ══════════════════════════════════════════════════════════════════════

def count_pv_nodes(net: pp.pandapowerNet) -> int:
    """Count PV nodes from pandapower network bus types."""
    try:
        pp.runpp(net, algorithm="nr", tolerance_mva=1e-6, max_iteration=10)
        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        return int(np.sum(bus_types == 2))
    except Exception:
        return 0


import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  IEE_PEGASE_NETWORKS — Complete catalog
# ══════════════════════════════════════════════════════════════════════

IEE_PEGASE_NETWORKS: dict[str, dict] = {
    # ═══════════════════════════════════════════════════════════════
    #  Very Small (≤10 buses)
    # ═══════════════════════════════════════════════════════════════
    "case4gs": {
        "constructor": create_case4gs,
        "description": "4 Bus Gene/Load system",
        "category": "ieee_tiny",
        "n_pv": None,
    },
    "case5": {
        "constructor": create_case5,
        "description": "Case 5 — 5 Bus system",
        "category": "ieee_tiny",
        "n_pv": None,
    },
    "case6ww": {
        "constructor": create_case6ww,
        "description": "6 Bus Ward & Ward",
        "category": "ieee_tiny",
        "n_pv": None,
    },
    "case9": {
        "constructor": create_case9,
        "description": "IEEE 9-Bus (3 Gen, 2 PV)",
        "category": "ieee_tiny",
        "n_pv": None,
    },

    # ═══════════════════════════════════════════════════════════════
    #  Small (11-50 buses)
    # ═══════════════════════════════════════════════════════════════
    "case14": {
        "constructor": create_case14,
        "description": "IEEE 14-Bus standard",
        "category": "ieee_small",
        "n_pv": None,
    },
    "case_ieee30": {
        "constructor": create_case_ieee30,
        "description": "IEEE 30-Bus (5 PV)",
        "category": "ieee_medium",
        "n_pv": None,
    },
    "case33bw": {
        "constructor": create_case33bw,
        "description": "IEEE 33-Bus Baran & Wu (radial)",
        "category": "ieee_radial",
        "n_pv": None,
    },

    # ═══════════════════════════════════════════════════════════════
    #  Medium (51-100 buses)
    # ═══════════════════════════════════════════════════════════════
    "case39": {
        "constructor": create_case39,
        "description": "IEEE 39-Bus New England (10 Gen)",
        "category": "ieee_medium",
        "n_pv": None,
    },
    "case57": {
        "constructor": create_case57,
        "description": "IEEE 57-Bus (6 PV)",
        "category": "ieee_medium",
        "n_pv": None,
    },

    # ═══════════════════════════════════════════════════════════════
    #  Large (101-500 buses)
    # ═══════════════════════════════════════════════════════════════
    "case_ieee118": {
        "constructor": create_case_ieee118,
        "description": "IEEE 118-Bus",
        "category": "ieee_large",
        "n_pv": None,
    },
    "case300": {
        "constructor": create_case300,
        "description": "IEEE 300-Bus",
        "category": "ieee_xlarge",
        "n_pv": None,
    },

    # ═══════════════════════════════════════════════════════════════
    #  Very Large (>1000 buses) — PEGASE & RTE
    # ═══════════════════════════════════════════════════════════════
    "case1354pegase": {
        "constructor": create_case1354pegase,
        "description": "PEGASE 1354-Bus",
        "category": "pegase",
        "n_pv": None,
    },
    "case1888rte": {
        "constructor": create_case1888rte,
        "description": "RTE French 1888-Bus",
        "category": "rte",
        "n_pv": None,
    },
    "case2848rte": {
        "constructor": create_case2848rte,
        "description": "RTE French 2848-Bus",
        "category": "rte",
        "n_pv": None,
    },
    "case2869pegase": {
        "constructor": create_case2869pegase,
        "description": "PEGASE 2869-Bus",
        "category": "pegase",
        "n_pv": None,
    },
    "case9241pegase": {
        "constructor": create_case9241pegase,
        "description": "PEGASE 9241-Bus (very large)",
        "category": "pegase",
        "n_pv": None,
    },
}


def _resolve_n_pv():
    """Resolve n_pv for all networks by actually loading them."""
    for name, info in IEE_PEGASE_NETWORKS.items():
        try:
            net = info["constructor"]()
            info["n_pv"] = count_pv_nodes(net)
        except Exception:
            info["n_pv"] = 0


_resolve_n_pv()


# ══════════════════════════════════════════════════════════════════════
#  Filter Functions
# ══════════════════════════════════════════════════════════════════════

def get_ieee_networks() -> dict[str, dict]:
    """All IEEE standard test cases (9, 14, 30, 33, 39, 57, 118, 300)."""
    ieee_names = [
        "case9", "case14", "case_ieee30", "case33bw",
        "case39", "case57", "case_ieee118", "case300"
    ]
    return {k: v for k, v in IEE_PEGASE_NETWORKS.items() if k in ieee_names}


def get_pegase_networks() -> dict[str, dict]:
    """All PEGASE test cases (1354, 2869, 9241)."""
    return {k: v for k, v in IEE_PEGASE_NETWORKS.items() if "pegase" in k}


def get_rte_networks() -> dict[str, dict]:
    """All RTE French test cases (1888, 2848)."""
    return {k: v for k, v in IEE_PEGASE_NETWORKS.items() if "rte" in k}


def get_large_networks() -> dict[str, dict]:
    """Networks with > 100 buses."""
    large_cats = ("ieee_large", "ieee_xlarge", "pegase", "rte")
    return {k: v for k, v in IEE_PEGASE_NETWORKS.items() if v["category"] in large_cats}


def get_all_standard_networks() -> dict[str, dict]:
    """All IEEE, PEGASE, and RTE networks."""
    return IEE_PEGASE_NETWORKS.copy()


def get_networks_by_category(category: str) -> dict[str, dict]:
    """Filter networks by category."""
    return {k: v for k, v in IEE_PEGASE_NETWORKS.items() if v["category"] == category}


# ══════════════════════════════════════════════════════════════════════
#  Standalone: Show overview
# ══════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  IEEE/PEGASE/RTE STANDARD NETWORKS (ieee_pegase_networks.py)    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    print(f"\n  Total networks: {len(IEE_PEGASE_NETWORKS)}")
    print(f"{'─'*70}")
    print(f"  {'Name':<22} {'Kat.':<15} {'#PV':<5} {'Beschreibung'}")
    print(f"{'─'*70}")

    for name, info in IEE_PEGASE_NETWORKS.items():
        print(
            f"  {name:<22} {info['category']:<15} "
            f"{info['n_pv']:<5} {info['description']}"
        )

    print(f"\n{'═'*70}")
    print(f"  IEEE networks:  {len(get_ieee_networks())}")
    print(f"  PEGASE networks: {len(get_pegase_networks())}")
    print(f"  RTE networks:    {len(get_rte_networks())}")
    print(f"  Large (>100):    {len(get_large_networks())}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
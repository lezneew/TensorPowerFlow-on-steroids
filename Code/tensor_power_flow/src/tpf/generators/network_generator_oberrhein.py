# tensor_power_flow/src/tpf/generators/network_generator_oberrhein.py
"""
MV Oberrhein German Distribution Network
=========================================

Medium Voltage Oberrhein network from pandapower.
Real-world German distribution network with 179 buses, 153 DG units.

Verwendung:
    from tpf.generators.network_generator_oberrhein import (
        MV_OBERHEIN_NETWORKS,
        create_mv_oberrhein,
        get_oberrhein_networks,
    )

    # Einzelnes Netz:
    net = create_mv_oberrhein()

    # Alle Netze iterieren:
    for name, info in MV_OBERHEIN_NETWORKS.items():
        net = info["constructor"]()
"""

import pandapower as pp
import pandapower.networks as pn
import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  Constructor Functions — MV Oberrhein network (as-is from pandapower)
# ══════════════════════════════════════════════════════════════════════

def create_mv_oberrhein() -> pp.pandapowerNet:
    """
    German MV Oberrhein distribution network.
    
    Returns 179-bus network with:
    - 2 external grids (slack buses at 58, 318)
    - 147 loads
    - 153 static generators (DG/PV)
    - 2 transformers (110/20 kV)
    - 181 lines (R/X ≈ 1.22)
    """
    return pn.mv_oberrhein()


# ══════════════════════════════════════════════════════════════════════
#  Helper Functions — PV identification and network statistics
# ══════════════════════════════════════════════════════════════════════

def count_pv_nodes(net: pp.pandapowerNet) -> int:
    """Count PV nodes from static generators (sgen)."""
    return int(len(net.sgen))


def identify_pv_buses(net: pp.pandapowerNet) -> list[int]:
    """
    Identify PV buses from static generators.
    
    Returns list of bus indices where sgen elements are connected.
    All 153 static generators in Oberrhein are treated as PV nodes.
    """
    return list(net.sgen.bus.values)


def get_voltage_stats(net: pp.pandapowerNet) -> tuple[float, float, float]:
    """
    Get voltage statistics from power flow results.
    
    Returns (min, max, mean) voltage in p.u.
    """
    pp.runpp(net, algorithm="nr", tolerance_mva=1e-8, max_iteration=100, numba=False)
    vm_pu = net.res_bus.vm_pu.values
    return float(vm_pu.min()), float(vm_pu.max()), float(vm_pu.mean())


# ══════════════════════════════════════════════════════════════════════
#  MV_OBERHEIN_NETWORKS — Complete catalog
# ══════════════════════════════════════════════════════════════════════

MV_OBERHEIN_NETWORKS: dict[str, dict] = {
    "mv_oberrhein": {
        "constructor": create_mv_oberrhein,
        "description": "German MV Oberrhein (179 buses, 153 DG, real-world)",
        "n_bus": 179,
        "n_pv": 153,
        "n_load": 147,
        "n_trafo": 2,
        "n_line": 181,
        "vn_kv": 20.0,
        "r_x_ratio": 1.22,
        "vm_pu_range": (0.976, 1.029),
        "category": "real_world",
    }
}


# ══════════════════════════════════════════════════════════════════════
#  Filter Functions
# ══════════════════════════════════════════════════════════════════════

def get_oberrhein_networks() -> dict[str, dict]:
    """All Oberrhein networks."""
    return MV_OBERHEIN_NETWORKS.copy()


def get_oberrhein_real_world() -> dict[str, dict]:
    """Real-world distribution networks (Oberrhein)."""
    return {k: v for k, v in MV_OBERHEIN_NETWORKS.items()
            if v["category"] == "real_world"}


# ══════════════════════════════════════════════════════════════════════
#  Standalone: Show overview
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MV OBERHEIN NETWORK (network_generator_oberrhein.py)")
    print("=" * 70)

    print(f"\n  Total networks: {len(MV_OBERHEIN_NETWORKS)}")
    print("-" * 70)
    print(f"  {'Name':<22} {'Category':<15} {'#PV':<5} {'Description'}")
    print("-" * 70)

    for name, info in MV_OBERHEIN_NETWORKS.items():
        print(
            f"  {name:<22} {info['category']:<15} "
            f"{info['n_pv']:<5} {info['description']}"
        )

# Validate network
    print(f"\n{'=' * 70}")
    print("  VALIDATION")
    print("=" * 70)
    
    try:
        net = create_mv_oberrhein()
        vm_min, vm_max, vm_mean = get_voltage_stats(net)
        
        print(f"  Network loaded successfully")
        print(f"  Buses: {len(net.bus)}")
        print(f"  Lines: {len(net.line)}")
        print(f"  Transformers: {len(net.trafo)}")
        print(f"  Loads: {len(net.load)}")
        print(f"  DG (sgen): {len(net.sgen)}")
        print(f"  External grids: {len(net.ext_grid)}")
        print(f"  Voltage: {vm_min:.4f} - {vm_max:.4f} p.u. (mean: {vm_mean:.4f})")
        print(f"  Converged: {net.converged}")
        print(f"  R/X ratio: ~{net.line.r_ohm_per_km.mean() / net.line.x_ohm_per_km.mean():.2f}")
    except Exception as e:
        print(f"  Validation failed: {e}")


if __name__ == "__main__":
    main()
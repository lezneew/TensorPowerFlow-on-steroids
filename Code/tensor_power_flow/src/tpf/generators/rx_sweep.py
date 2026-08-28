# tensor_power_flow/src/tpf/generators/rx_sweep.py
"""
R/X-Sweep-Generator
===================
Reparametrisiert die Leitungsimpedanzen eines Salazar-Netzes auf ein
vorgegebenes R/X-Verhältnis und tuned anschließend die PV-Sollspannungen
auf das neue natürliche Spannungsprofil.

Modi:
    const_z : |z| = const, nur Impedanzwinkel variiert  (Kontrollexperiment)
    const_x : X = const, R skaliert                     (praxisnah, HS->NS)
    const_r : R = const, X skaliert
"""

from __future__ import annotations
import copy
from dataclasses import dataclass

import numpy as np
import pandapower as pp

from tpf.generators.network_generator_salazar import create_salazar_network

# Referenz-Leitungsparameter (Ohm/km) des unskalierten Salazar-Netzes
R_REF = 0.3144
X_REF = 1.954
Z_REF = float(np.hypot(R_REF, X_REF))


# ══════════════════════════════════════════════════════════════════
#  Impedanz-Reparametrisierung
# ══════════════════════════════════════════════════════════════════

def rx_to_impedance(
    rx: float,
    mode: str = "const_z",
    z_abs: float = Z_REF,
    r_ref: float = R_REF,
    x_ref: float = X_REF,
) -> tuple[float, float]:
    """Liefert (r_ohm_per_km, x_ohm_per_km) fuer gegebenes R/X."""
    if rx <= 0:
        raise ValueError("rx muss > 0 sein")

    if mode == "const_z":
        denom = np.sqrt(1.0 + rx**2)
        x = z_abs / denom
        r = z_abs * rx / denom
    elif mode == "const_x":
        x = x_ref
        r = rx * x_ref
    elif mode == "const_r":
        r = r_ref
        x = r_ref / rx
    else:
        raise ValueError(f"unbekannter mode: {mode}")

    return float(r), float(x)


def set_line_impedances(
    net: pp.pandapowerNet,
    r_ohm_per_km: float,
    x_ohm_per_km: float,
    line_factor: float = 3.0,
) -> None:
    """Setzt alle Leitungen auf identische Impedanz (in-place)."""
    net.line.loc[:, "r_ohm_per_km"] = r_ohm_per_km / line_factor
    net.line.loc[:, "x_ohm_per_km"] = x_ohm_per_km / line_factor
    net.line.loc[:, "c_nf_per_km"] = 0.0


# ══════════════════════════════════════════════════════════════════
#  PV-Sollspannungen an neues Profil anpassen
# ══════════════════════════════════════════════════════════════════

def natural_vm_profile(net: pp.pandapowerNet) -> np.ndarray:
    """
    Spannungsprofil des Netzes OHNE Spannungsregelung:
    alle gen werden temporaer als PQ-Einspeiser (sgen, Q=0) behandelt.
    """
    n = copy.deepcopy(net)
    if len(n.gen) > 0:
        for _, row in n.gen.iterrows():
            pp.create_sgen(n, bus=int(row["bus"]), p_mw=float(row["p_mw"]),
                           q_mvar=0.0)
        n.gen.loc[:, "in_service"] = False

    for init in ("auto", "dc", "flat"):
        try:
            pp.runpp(n, algorithm="nr", tolerance_mva=1e-8,
                     max_iteration=200, init=init)
            if n.converged:
                return n.res_bus["vm_pu"].values.copy()
        except Exception:
            continue
    raise RuntimeError("natuerliches Profil nicht berechenbar")


def retune_pv_setpoints(
    net: pp.pandapowerNet,
    offset_pu: float = 0.005,
    std_pu: float = 0.0,
    seed: int | None = 0,
    clip: tuple[float, float] = (0.94, 1.08),
) -> np.ndarray:
    """
    Setzt gen.vm_pu = v_natural(bus) + offset (+ Rauschen).
    Muss nach jeder Impedanzaenderung aufgerufen werden, sonst liegen die
    Sollwerte relativ zum neuen Profil willkuerlich weit entfernt und der
    Vergleich ueber R/X misst den Sollwertabstand statt der Sensitivitaet.
    """
    if len(net.gen) == 0:
        return np.array([])

    rng = np.random.default_rng(seed)
    vm_nat = natural_vm_profile(net)
    buses = net.gen["bus"].values.astype(int)
    vm_new = np.clip(vm_nat[buses] + offset_pu
                     + rng.normal(0.0, std_pu, size=len(buses)),
                     clip[0], clip[1])
    net.gen.loc[:, "vm_pu"] = vm_new
    return vm_new


# ══════════════════════════════════════════════════════════════════
#  Netzbau
# ══════════════════════════════════════════════════════════════════

@dataclass
class RXCase:
    net: pp.pandapowerNet
    rx: float
    mode: str
    r_ohm_per_km: float
    x_ohm_per_km: float
    z_abs_ohm_per_km: float
    nodes: int
    n_pv: int
    pv_ratio: float
    load_factor: float
    vm_offset_pu: float
    line_factor: float = 3.0
    z_rel: float = 1.0


def build_rx_case(
    nodes: int = 40,
    pv_ratio: float = 0.10,
    rx: float = 1.0,
    mode: str = "const_z",
    z_abs: float = Z_REF,
    load_factor: float = 2.0,
    line_factor: float = 3.0,
    child: int = 3,
    vm_offset_pu: float = 0.005,
    vm_std_pu: float = 0.0,
    seed: int = 2000,
) -> RXCase:
    """
    Erzeugt ein Salazar-Netz mit vorgegebenem R/X.

    Ablauf:
      1. Netz mit pv_vm_mode='fixed' bauen (kein interner Lastfluss)
      2. Leitungsimpedanzen ueberschreiben
      3. PV-Sollwerte auf neues natuerliches Profil retunen
    """
    n_pv = min(max(0, int(round(nodes * pv_ratio))), nodes - 3)

    net = create_salazar_network(
        nodes=nodes, child=child, n_pv=n_pv,
        load_factor=load_factor, line_factor=line_factor,
        pv_vm_mode="fixed", pv_vm_pu=1.0, pv_vm_std_pu=0.0,
        seed=seed, skip_validation=True,
    )

    r, x = rx_to_impedance(rx, mode=mode, z_abs=z_abs)
    set_line_impedances(net, r, x, line_factor=line_factor)

    if n_pv > 0:
        retune_pv_setpoints(net, offset_pu=vm_offset_pu,
                            std_pu=vm_std_pu, seed=seed)
    refresh_ppc(net)

    z_abs_line = float(np.hypot(r, x) / line_factor)
    z_rel = float(np.hypot(r, x) / z_abs)

    return RXCase(
        net=net, rx=rx, mode=mode,
        r_ohm_per_km=r / line_factor, x_ohm_per_km=x / line_factor,
        z_abs_ohm_per_km=z_abs_line,
        nodes=nodes, n_pv=n_pv, pv_ratio=pv_ratio,
        load_factor=load_factor, vm_offset_pu=vm_offset_pu,
        line_factor=line_factor, z_rel=z_rel,
    )


def rx_grid(n_points: int = 13, rx_min: float = 0.1, rx_max: float = 10.0):
    """Logarithmisches R/X-Gitter, enthaelt exakt 1.0."""
    grid = np.geomspace(rx_min, rx_max, n_points)
    return np.unique(np.round(np.append(grid, 1.0), 4))



def refresh_ppc(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Erzwingt ein aktuelles net._ppc (PV-Bustypen, neue Impedanzen)."""
    for kwargs in ({"init": "auto"}, {"init": "dc"}, {"init": "flat"}):
        try:
            pp.runpp(net, algorithm="nr", tolerance_mva=1e-8,
                     max_iteration=200, **kwargs)
            if net.converged:
                return net
        except Exception:
            continue
    pp.rundcpp(net)          # letzter Ausweg: nur ppc-Struktur, kein AC-Ergebnis
    return net

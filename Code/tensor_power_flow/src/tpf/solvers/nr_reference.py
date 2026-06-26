import numpy as np
import pandapower as pp
import time

from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver
from tpf.core.network import NetworkData
from numpy.typing import NDArray


class PandapowerNRSolver(BaseSolver):
    """Wrapper um pandapower Newton-Raphson als Referenzlösung (mit PV-Support)."""

    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        super().__init__(tol, max_iter)

    def solve(self, network: NetworkData) -> PowerFlowResult:
        raise NotImplementedError(
            "Verwende solve_from_net(net) mit einem pandapower-Netz."
        )

    def solve_batch(self, network: NetworkData, s_batch: NDArray) -> PowerFlowResult:
        raise NotImplementedError(
            "Batch-Lösung über pandapower nicht unterstützt."
        )

    def solve_from_net(self, net: pp.pandapowerNet) -> PowerFlowResult:
        """
        Löst den Lastfluss auf einem pandapower-Netz.
        Extrahiert zusätzlich PV-Knoten-Ergebnisse (Q, V_setpoint).
        """
        t_start = time.perf_counter()

        pp.runpp(
            net,
            algorithm="nr",
            tolerance_mva=self.tol,
            max_iteration=self.max_iter,
            enforce_q_lims=False,
        )

        elapsed = time.perf_counter() - t_start

        # --- Spannungen aller Busse ---
        vm_pu = net.res_bus["vm_pu"].values
        va_deg = net.res_bus["va_degree"].values
        v_complex = vm_pu * np.exp(1j * np.deg2rad(va_deg))

        # --- Iterationen ---
        iterations = -1
        if hasattr(net, "_ppc") and net._ppc is not None:
            iterations = net._ppc.get("iterations", -1)

        # --- PV-Knoten-Ergebnisse extrahieren ---
        pv_indices, pv_q_pu, pv_v_setpoint = self._extract_pv_results(net)

        return PowerFlowResult(
            voltages=v_complex,
            iterations=iterations,
            converged=net.converged,
            elapsed_time_s=elapsed,
            max_mismatch=0.0,
            pv_indices=pv_indices,
            pv_q_pu=pv_q_pu,
            pv_v_setpoint_pu=pv_v_setpoint,
        )

    def _extract_pv_results(self, net: pp.pandapowerNet):
        """
        Extrahiert PV-Knoten-Daten aus dem gelösten pandapower-Netz.

        Returns
        -------
        pv_indices : ndarray (PPC-Busindizes der PV-Knoten)
        pv_q_pu : ndarray (Blindleistung in p.u. an jedem PV-Knoten)
        pv_v_setpoint : ndarray (Sollspannungsbetrag in p.u.)
        """
        ppc = net._ppc
        bus_types = ppc["bus"][:, 1].astype(int)
        pv_idx = np.where(bus_types == 2)[0]

        if len(pv_idx) == 0:
            return None, None, None

        base_mva = ppc["baseMVA"]

        # Sollspannung der PV-Knoten (aus PPC bus-Daten, Spalte 7 = Vm)
        pv_v_setpoint = ppc["bus"][pv_idx, 7]  # p.u.

        # Q an PV-Knoten: Summe aus gen-Einspeisung - Last
        # Generatoren am PV-Bus
        gen_buses = ppc["gen"][:, 0].astype(int)
        pv_q_pu = np.zeros(len(pv_idx))

        for i, bus in enumerate(pv_idx):
            # Generator-Q an diesem Bus (in MW → p.u.)
            gen_mask = gen_buses == bus
            q_gen = np.sum(ppc["gen"][gen_mask, 2]) / base_mva  # Qg in p.u.

            # Last-Q an diesem Bus
            q_load = ppc["bus"][bus, 3] / base_mva  # Qd in p.u.

            # Netto-Einspeisung (aus Sicht der Netzgleichung)
            pv_q_pu[i] = q_gen - q_load

        return pv_idx, pv_q_pu, pv_v_setpoint
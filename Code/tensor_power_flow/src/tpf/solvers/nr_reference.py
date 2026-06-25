# solvers/nr_reference.py — vollständige korrigierte Version
import numpy as np
import pandapower as pp
import time

from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver
from tpf.core.network import NetworkData
from numpy.typing import NDArray


class PandapowerNRSolver(BaseSolver):
    """Wrapper um pandapower Newton-Raphson als Referenzlösung."""

    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        super().__init__(tol, max_iter)

    def solve(self, network: NetworkData) -> PowerFlowResult:
        """Nicht direkt nutzbar — verwende solve_from_net()."""
        raise NotImplementedError(
            "Verwende solve_from_net(net) mit einem pandapower-Netz."
        )

    def solve_batch(self, network: NetworkData, s_batch: NDArray) -> PowerFlowResult:
        """Nicht direkt nutzbar — verwende solve_from_net()."""
        raise NotImplementedError(
            "Batch-Lösung über pandapower nicht unterstützt."
        )

    def solve_from_net(self, net: pp.pandapowerNet) -> PowerFlowResult:
        """Löst den Lastfluss direkt auf einem pandapower-Netz."""
        t_start = time.perf_counter()

        pp.runpp(
            net,
            algorithm="nr",
            tolerance_mva=self.tol,
            max_iteration=self.max_iter,
            enforce_q_lims=False,
        )

        elapsed = time.perf_counter() - t_start

        vm_pu = net.res_bus["vm_pu"].values
        va_deg = net.res_bus["va_degree"].values
        v_complex = vm_pu * np.exp(1j * np.deg2rad(va_deg))

        # Iterationszahl aus internem PPC extrahieren
        iterations = -1
        if hasattr(net, "_ppc") and net._ppc is not None:
            iterations = net._ppc.get("iterations", -1)

        return PowerFlowResult(
            voltages=v_complex,
            iterations=iterations,
            converged=net.converged,
            elapsed_time_s=elapsed,
            max_mismatch=0.0,
        )
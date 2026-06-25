import numpy as np
from numpy.typing import NDArray
import time

from tpf.core.network import NetworkData
from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver


class TPFDenseSolver(BaseSolver):
    """
    Tensor Power Flow - Dense Formulation.
    Implementiert Algorithm 1 aus Salazar Duque et al. (2024).
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        super().__init__(tol, max_iter)
        self._Z_B: NDArray | None = None
        self._w: NDArray | None = None

    def _precompute(self, network: NetworkData) -> None:
        """Berechnet Z_B und w einmalig (konstant über alle Iterationen)."""
        # Z_B = Y_dd^{-1}  (Bus-Impedanzmatrix)
        self._Z_B = np.linalg.inv(network.Y_dd)
        # w = -Z_B · Y_ds · v_s
        self._w = -self._Z_B @ network.Y_ds @ network.v_s

    def solve(self, network: NetworkData) -> PowerFlowResult:
        """Löst einen einzelnen Lastfluss."""
        return self.solve_batch(network, network.s_nom.reshape(-1, 1))

    def solve_batch(
            self, network: NetworkData, s_batch: NDArray
    ) -> PowerFlowResult:
        """
        Löst τ Lastflüsse parallel (Dense-Formulierung).

        Parameters
        ----------
        network : NetworkData
        s_batch : NDArray, shape (b·φ, τ)
            Leistungsmatrix: jede Spalte ist ein Lastfall.

        Returns
        -------
        PowerFlowResult
        """
        t_start = time.perf_counter()

        # Vorberechnung (einmalig)
        self._precompute(network)

        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        # Initialisierung: Flat Start
        V = np.ones((bphi, tau), dtype=np.complex128)
        W = np.tile(self._w.reshape(-1, 1), (1, tau))  # (b·φ × τ)

        # Konjugierte Leistung (konstant, da nur PQ-Knoten)
        S_conj = np.conj(s_batch)  # (b·φ × τ)

        converged = False
        n_iter = 0
        max_mismatch = np.inf

        for n in range(self.max_iter):
            # Kern-Iteration: V_{n+1} = Z_B · (S* ⊙ V*^{-1}) + W
            V_conj_inv = 1.0 / np.conj(V)  # element-weise
            I_conj = S_conj * V_conj_inv  # (b·φ × τ)
            V_new = -(self._Z_B @ I_conj) + W  # (b·φ × τ)

            # Konvergenzprüfung (Leistungsmismatch)
            S_calc = -V_new * np.conj(
                network.Y_dd @ V_new
                + (network.Y_ds @ network.v_s).reshape(-1, 1)
            )
            max_mismatch = np.max(np.abs(s_batch - S_calc))

            n_iter = n + 1
            V = V_new

            if max_mismatch < self.tol:
                converged = True
                break

        elapsed = time.perf_counter() - t_start

        return PowerFlowResult(
            voltages=V,
            iterations=n_iter,
            converged=converged,
            elapsed_time_s=elapsed,
            max_mismatch=max_mismatch,
        )
# tensor_power_flow/src/tpf/solvers/tpf_dense.py

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

    Unterstützt:
    - Constant Power Only (α_P=1, α_I=0, α_Z=0): Optimierter Pfad mit K, L
    - Volles ZIP-Modell (beliebige α_P, α_I, α_Z): Per-Zeitschritt F-Tensor

    Konvergenzkriterium: max(||V_new| - |V_old||)  (günstig, keine extra
    Matrixmultiplikation pro Iteration).
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        super().__init__(tol, max_iter)

    # ══════════════════════════════════════════════════════════════════════
    #  Hilfsmethoden
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _is_constant_power_only(network: NetworkData) -> bool:
        """Prüft ob ausschließlich constant-power Lasten vorliegen."""
        return (
            np.all(network.alpha_p == 1.0)
            and not np.any(network.alpha_i)
            and not np.any(network.alpha_z)
        )

    @staticmethod
    def _precompute_constant_power(network: NetworkData):
        """
        Optimierter Pfad für reine constant-power Lasten.

        Berechnet:
            K = -Y_dd^{-1}           (bφ × bφ)  — konstant über alle τ
            L = K @ Y_ds @ v_s       (bφ,)      — konstant über alle τ

        Iteration: V_{n+1} = K @ (S* ⊙ V*^{-1}) + L
        """
        Z_B = np.linalg.inv(network.Y_dd)
        K = -Z_B  # (bφ × bφ)
        L = K @ network.Y_ds @ network.v_s  # (bφ,)
        return K, L

    @staticmethod
    def _precompute_zip(network: NetworkData, s_batch: NDArray):
        """
        Volles ZIP-Modell: Berechnet F-Tensor und W-Matrix.

        Optimiert für zwei Fälle:
        - ZIP-A (αZ=0): K konstant, kein F-Tensor nötig
        - ZIP-B (αZ≠0): F-Tensor mit batched MatMul

        Parameters
        ----------
        network : NetworkData
        s_batch : (bφ, τ) Leistungsmatrix

        Returns
        -------
        zip_type : str
            'zip_a' für αZ=0 (optimiert), 'zip_b' für αZ≠0
        F : (τ, bφ, bφ) Tensor oder None
        W : (bφ, τ) Matrix
        K : (bφ, bφ) Matrix (nur für ZIP-A)
        alpha_p_s_conj : (bφ, τ) Matrix (nur für ZIP-A)
        """
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        alpha_p = network.alpha_p  # (bφ,)
        alpha_i = network.alpha_i  # (bφ,)
        alpha_z = network.alpha_z  # (bφ,)

        all_alpha_z_zero = not np.any(alpha_z)
        all_alpha_i_zero = not np.any(alpha_i)

        c_base = (network.Y_ds @ network.v_s).flatten()  # (bφ,)

        if all_alpha_z_zero:
            K = -np.linalg.inv(network.Y_dd)  # (bφ, bφ)

            alpha_p_s_conj = alpha_p.reshape(-1, 1) * np.conj(s_batch)  # (bφ, τ)

            if all_alpha_i_zero:
                W = np.tile(K @ c_base, (tau, 1)).T  # (bφ, τ)
            else:
                alpha_i_s_conj = alpha_i.reshape(-1, 1) * np.conj(s_batch)  # (bφ, τ)
                c_with_i = c_base.reshape(-1, 1) + alpha_i_s_conj  # (bφ, τ)
                W = K @ c_with_i  # (bφ, τ)

            return 'zip_a', None, W, K, alpha_p_s_conj

        F = np.zeros((tau, bphi, bphi), dtype=np.complex128)
        W = np.zeros((bphi, tau), dtype=np.complex128)

        for i in range(tau):
            s_conj_i = np.conj(s_batch[:, i])  # (bφ,)

            B = np.diag(alpha_z * s_conj_i) + network.Y_dd
            B_inv = np.linalg.inv(B)

            alpha_p_s_conj = alpha_p * s_conj_i  # (bφ,)
            F[i] = -B_inv * alpha_p_s_conj.reshape(1, -1)

            if all_alpha_i_zero:
                c_i = c_base
            else:
                c_i = c_base + alpha_i * s_conj_i

            W[:, i] = -B_inv @ c_i

        return 'zip_b', F, W, None, None

    # ══════════════════════════════════════════════════════════════════════
    #  Solver-Methoden
    # ══════════════════════════════════════════════════════════════════════

    def solve(self, network: NetworkData) -> PowerFlowResult:
        """Löst einen einzelnen Lastfluss (τ=1)."""
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

        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        constant_power = self._is_constant_power_only(network)

        if constant_power:
            V, n_iter, converged, tol_final = self._solve_constant_power(
                network, s_batch, bphi, tau
            )
        else:
            V, n_iter, converged, tol_final = self._solve_zip(
                network, s_batch, bphi, tau
            )

        elapsed = time.perf_counter() - t_start

        s_slack = None
        if network.Y_ss is not None and network.Y_sd is not None:
            V_mat = V if V.ndim == 2 else V.reshape(-1, 1)
            v_s = network.v_s.reshape(-1, 1)
            I_s = network.Y_ss @ v_s + network.Y_sd @ V_mat
            s_slack = v_s * np.conj(I_s)

        return PowerFlowResult(
            voltages=V,
            iterations=n_iter,
            converged=converged,
            elapsed_time_s=elapsed,
            max_mismatch=tol_final,
            s_slack=s_slack,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Constant Power Only (optimierter Pfad)
    # ──────────────────────────────────────────────────────────────────────

    def _solve_constant_power(
        self, network: NetworkData, s_batch: NDArray, bphi: int, tau: int
    ):
        """
        FPI für reine constant-power Lasten.

        Iteration:
            Λ = S* ⊙ (1/V*)         (Hadamard, bφ × τ)
            V_{n+1} = K @ Λ + L     (Matrix-Mult + Broadcast)

        Konvergenz: max(||V_{n+1}| - |V_n||)
        """
        K, L = self._precompute_constant_power(network)

        # Initialisierung: Flat Start
        V = np.ones((bphi, tau), dtype=np.complex128)
        S_conj = np.conj(s_batch)  # (bφ × τ), konstant
        L_col = L.reshape(-1, 1)  # (bφ × 1) für Broadcasting über τ

        converged = False
        n_iter = 0
        tol_val = np.inf

        for n in range(self.max_iter):
            # Λ = S* / V*  (elementweise)
            LAMBDA = S_conj * (1.0 / np.conj(V))  # (bφ × τ)

            # V_{n+1} = K @ Λ + L
            V_new = K @ LAMBDA + L_col  # (bφ × τ)

            # Konvergenzkriterium: Spannungsänderung (günstig!)
            tol_val = np.max(np.abs(np.abs(V_new) - np.abs(V)))

            n_iter = n + 1
            V = V_new

            if tol_val < self.tol:
                converged = True
                break

        return V, n_iter, converged, tol_val

    # ──────────────────────────────────────────────────────────────────────
    #  Volles ZIP-Modell (optimiert)
    # ──────────────────────────────────────────────────────────────────────

    def _solve_zip(
        self, network: NetworkData, s_batch: NDArray, bphi: int, tau: int
    ):
        """
        FPI für das volle ZIP-Lastmodell.

        Zwei optimierte Pfade:
        - ZIP-A (αZ=0): Single-GEMM + Skalierung (wie CP)
        - ZIP-B (αZ≠0): Batched matmul

        Vorberechnung (einmalig):
            ZIP-A: K, W, alpha_p_s_conj
            ZIP-B: F[i] = -B[i]^{-1} · diag(α_P ⊙ s*[i])     (τ × bφ × bφ)
                   W[i] = -B[i]^{-1} · c[i]                    (bφ × τ)

        Iteration:
            ZIP-A: V_{n+1} = K @ (alpha_p_s_conj ⊙ v_recp_conj) + W
            ZIP-B: V_{n+1}[:, i] = F[i] @ v_recp_conj[:, i] + W[:, i]

        Konvergenz: max(||V_{n+1}| - |V_n||)
        """
        zip_type, F, W, K, alpha_p_s_conj = self._precompute_zip(network, s_batch)

        V = np.ones((bphi, tau), dtype=np.complex128)
        converged = False
        n_iter = 0
        tol_val = np.inf

        if zip_type == 'zip_a':
            for n in range(self.max_iter):
                v_recp_conj = 1.0 / np.conj(V)  # (bφ, τ)
                LAMBDA = alpha_p_s_conj * v_recp_conj  # (bφ, τ) elementwise
                V_new = K @ LAMBDA + W  # Single GEMM: (bφ, bφ) @ (bφ, τ) -> (bφ, τ)

                tol_val = np.max(np.abs(np.abs(V_new) - np.abs(V)))
                n_iter = n + 1
                V = V_new

                if tol_val < self.tol:
                    converged = True
                    break
        else:
            for n in range(self.max_iter):
                v_recp_conj = 1.0 / np.conj(V)  # (bφ, τ)

                rhs = np.ascontiguousarray(v_recp_conj.T[:, :, None])  # (τ, bφ, 1)
                V_new = np.matmul(F, rhs).squeeze(-1).T + W  # Batched matmul

                tol_val = np.max(np.abs(np.abs(V_new) - np.abs(V)))
                n_iter = n + 1
                V = V_new

                if tol_val < self.tol:
                    converged = True
                    break

        return V, n_iter, converged, tol_val

    # # tensor_power_flow/src/tpf/solvers/tpf_dense.py
    #
    # import numpy as np
    # from numpy.typing import NDArray
    # import time
    #
    # from tpf.core.network import NetworkData
    # from tpf.core.results import PowerFlowResult
    # from tpf.solvers.base_solver import BaseSolver
    #
    # class TPFDenseSolver(BaseSolver):
    #     """
    #     Tensor Power Flow - Dense Formulation.
    #     Implementiert Algorithm 1 aus Salazar Duque et al. (2024).
    #
    #     Unterstützt:
    #     - Constant Power Only (α_P=1, α_I=0, α_Z=0): Optimierter Pfad mit K, L
    #     - Volles ZIP-Modell (beliebige α_P, α_I, α_Z): Per-Zeitschritt F-Tensor
    #
    #     Konvergenzkriterium: max(||V_new| - |V_old||)  (günstig, keine extra
    #     Matrixmultiplikation pro Iteration).
    #     """
    #
    #     def __init__(self, tol: float = 1e-6, max_iter: int = 100):
    #         super().__init__(tol, max_iter)
    #
    #     # ══════════════════════════════════════════════════════════════════════
    #     #  Hilfsmethoden
    #     # ══════════════════════════════════════════════════════════════════════
    #
    #     @staticmethod
    #     def _is_constant_power_only(network: NetworkData) -> bool:
    #         """Prüft ob ausschließlich constant-power Lasten vorliegen."""
    #         return (
    #                 np.all(network.alpha_p == 1.0)
    #                 and not np.any(network.alpha_i)
    #                 and not np.any(network.alpha_z)
    #         )
    #
    #     @staticmethod
    #     def _precompute_constant_power(network: NetworkData):
    #         """
    #         Optimierter Pfad für reine constant-power Lasten.
    #
    #         Berechnet:
    #             K = -Y_dd^{-1}           (bφ × bφ)  — konstant über alle τ
    #             L = K @ Y_ds @ v_s       (bφ,)      — konstant über alle τ
    #
    #         Iteration: V_{n+1} = K @ (S* ⊙ V*^{-1}) + L
    #         """
    #         Z_B = np.linalg.inv(network.Y_dd)
    #         K = -Z_B  # (bφ × bφ)
    #         L = K @ network.Y_ds @ network.v_s  # (bφ,)
    #         return K, L
    #
    #     @staticmethod
    #     def _precompute_zip(network: NetworkData, s_batch: NDArray):
    #         """
    #         Volles ZIP-Modell: Berechnet F-Tensor und W-Matrix.
    #
    #         Optimiert für zwei Fälle:
    #         - ZIP-A (αZ=0): K konstant, kein F-Tensor nötig
    #         - ZIP-B (αZ≠0): F-Tensor mit batched MatMul
    #
    #         Parameters
    #         ----------
    #         network : NetworkData
    #         s_batch : (bφ, τ) Leistungsmatrix
    #
    #         Returns
    #         -------
    #         zip_type : str
    #             'zip_a' für αZ=0 (optimiert), 'zip_b' für αZ≠0
    #         F : (τ, bφ, bφ) Tensor oder None
    #         W : (bφ, τ) Matrix
    #         K : (bφ, bφ) Matrix (nur für ZIP-A)
    #         alpha_p_s_conj : (bφ, τ) Matrix (nur für ZIP-A)
    #         """
    #         bphi = network.n_bus_phases
    #         tau = s_batch.shape[1]
    #
    #         alpha_p = network.alpha_p  # (bφ,)
    #         alpha_i = network.alpha_i  # (bφ,)
    #         alpha_z = network.alpha_z  # (bφ,)
    #
    #         all_alpha_z_zero = not np.any(alpha_z)
    #         all_alpha_i_zero = not np.any(alpha_i)
    #
    #         c_base = (network.Y_ds @ network.v_s).flatten()  # (bφ,)
    #
    #         if all_alpha_z_zero:
    #             K = -np.linalg.inv(network.Y_dd)  # (bφ, bφ)
    #
    #             alpha_p_s_conj = alpha_p.reshape(-1, 1) * np.conj(s_batch)  # (bφ, τ)
    #
    #             if all_alpha_i_zero:
    #                 W = np.tile(K @ c_base, (tau, 1)).T  # (bφ, τ)
    #             else:
    #                 alpha_i_s_conj = alpha_i.reshape(-1, 1) * np.conj(s_batch)  # (bφ, τ)
    #                 c_with_i = c_base.reshape(-1, 1) + alpha_i_s_conj  # (bφ, τ)
    #                 W = K @ c_with_i  # (bφ, τ)
    #
    #             return 'zip_a', None, W, K, alpha_p_s_conj
    #
    #         F = np.zeros((tau, bphi, bphi), dtype=np.complex128)
    #         W = np.zeros((bphi, tau), dtype=np.complex128)
    #
    #         for i in range(tau):
    #             s_conj_i = np.conj(s_batch[:, i])  # (bφ,)
    #
    #             B = np.diag(alpha_z * s_conj_i) + network.Y_dd
    #             B_inv = np.linalg.inv(B)
    #
    #             alpha_p_s_conj = alpha_p * s_conj_i  # (bφ,)
    #             F[i] = -B_inv * alpha_p_s_conj.reshape(1, -1)
    #
    #             if all_alpha_i_zero:
    #                 c_i = c_base
    #             else:
    #                 c_i = c_base + alpha_i * s_conj_i
    #
    #             W[:, i] = -B_inv @ c_i
    #
    #         return 'zip_b', F, W, None, None
    #
    #     # ══════════════════════════════════════════════════════════════════════
    #     #  Solver-Methoden
    #     # ══════════════════════════════════════════════════════════════════════
    #
    #     def solve(self, network: NetworkData) -> PowerFlowResult:
    #         """Löst einen einzelnen Lastfluss (τ=1)."""
    #         return self.solve_batch(network, network.s_nom.reshape(-1, 1))
    #
    #     def solve_batch(
    #             self, network: NetworkData, s_batch: NDArray
    #     ) -> PowerFlowResult:
    #         """
    #         Löst τ Lastflüsse parallel (Dense-Formulierung).
    #
    #         Parameters
    #         ----------
    #         network : NetworkData
    #         s_batch : NDArray, shape (b·φ, τ)
    #             Leistungsmatrix: jede Spalte ist ein Lastfall.
    #
    #         Returns
    #         -------
    #         PowerFlowResult
    #         """
    #         t_start = time.perf_counter()
    #
    #         bphi = network.n_bus_phases
    #         tau = s_batch.shape[1]
    #
    #         constant_power = self._is_constant_power_only(network)
    #
    #         if constant_power:
    #             V, n_iter, converged, tol_final = self._solve_constant_power(
    #                 network, s_batch, bphi, tau
    #             )
    #         else:
    #             V, n_iter, converged, tol_final = self._solve_zip(
    #                 network, s_batch, bphi, tau
    #             )
    #
    #         elapsed = time.perf_counter() - t_start
    #
    #         s_slack = None
    #         if network.Y_ss is not None and network.Y_sd is not None:
    #             V_mat = V if V.ndim == 2 else V.reshape(-1, 1)
    #             v_s = network.v_s.reshape(-1, 1)
    #             I_s = network.Y_ss @ v_s + network.Y_sd @ V_mat
    #             s_slack = v_s * np.conj(I_s)
    #
    #         return PowerFlowResult(
    #             voltages=V,
    #             iterations=n_iter,
    #             converged=converged,
    #             elapsed_time_s=elapsed,
    #             max_mismatch=tol_final,
    #             s_slack=s_slack,
    #         )
    #
    #     # ──────────────────────────────────────────────────────────────────────
    #     #  Constant Power Only (optimierter Pfad)
    #     # ──────────────────────────────────────────────────────────────────────
    #
    #     def _solve_constant_power(
    #             self, network: NetworkData, s_batch: NDArray, bphi: int, tau: int
    #     ):
    #         """
    #         FPI für reine constant-power Lasten.
    #
    #         Iteration:
    #             Λ = S* ⊙ (1/V*)         (Hadamard, bφ × τ)
    #             V_{n+1} = K @ Λ + L     (Matrix-Mult + Broadcast)
    #
    #         Konvergenz: max(||V_{n+1}| - |V_n||)
    #         """
    #         K, L = self._precompute_constant_power(network)
    #
    #         # Initialisierung: Flat Start
    #         V = np.ones((bphi, tau), dtype=np.complex128)
    #         S_conj = np.conj(s_batch)  # (bφ × τ), konstant
    #         L_col = L.reshape(-1, 1)  # (bφ × 1) für Broadcasting über τ
    #
    #         converged = False
    #         n_iter = 0
    #         tol_val = np.inf
    #
    #         for n in range(self.max_iter):
    #             # Λ = S* / V*  (elementweise)
    #             LAMBDA = S_conj * (1.0 / np.conj(V))  # (bφ × τ)
    #
    #             # V_{n+1} = K @ Λ + L
    #             V_new = K @ LAMBDA + L_col  # (bφ × τ)
    #
    #             # Konvergenzkriterium: Spannungsänderung (günstig!)
    #             tol_val = np.max(np.abs(np.abs(V_new) - np.abs(V)))
    #
    #             n_iter = n + 1
    #             V = V_new
    #
    #             if tol_val < self.tol:
    #                 converged = True
    #                 break
    #
    #         return V, n_iter, converged, tol_val
    #
    #     # ──────────────────────────────────────────────────────────────────────
    #     #  Volles ZIP-Modell (optimiert)
    #     # ──────────────────────────────────────────────────────────────────────
    #
    #     def _solve_zip(
    #             self, network: NetworkData, s_batch: NDArray, bphi: int, tau: int
    #     ):
    #         """
    #         FPI für das volle ZIP-Lastmodell.
    #
    #         Zwei optimierte Pfade:
    #         - ZIP-A (αZ=0): Single-GEMM + Skalierung (wie CP)
    #         - ZIP-B (αZ≠0): Batched matmul
    #
    #         Vorberechnung (einmalig):
    #             ZIP-A: K, W, alpha_p_s_conj
    #             ZIP-B: F[i] = -B[i]^{-1} · diag(α_P ⊙ s*[i])     (τ × bφ × bφ)
    #                    W[i] = -B[i]^{-1} · c[i]                    (bφ × τ)
    #
    #         Iteration:
    #             ZIP-A: V_{n+1} = K @ (alpha_p_s_conj ⊙ v_recp_conj) + W
    #             ZIP-B: V_{n+1}[:, i] = F[i] @ v_recp_conj[:, i] + W[:, i]
    #
    #         Konvergenz: max(||V_{n+1}| - |V_n||)
    #         """
    #         zip_type, F, W, K, alpha_p_s_conj = self._precompute_zip(network, s_batch)
    #
    #         V = np.ones((bphi, tau), dtype=np.complex128)
    #         converged = False
    #         n_iter = 0
    #         tol_val = np.inf
    #
    #         if zip_type == 'zip_a':
    #             for n in range(self.max_iter):
    #                 v_recp_conj = 1.0 / np.conj(V)  # (bφ, τ)
    #                 LAMBDA = alpha_p_s_conj * v_recp_conj  # (bφ, τ) elementwise
    #                 V_new = K @ LAMBDA + W  # Single GEMM: (bφ, bφ) @ (bφ, τ) -> (bφ, τ)
    #
    #                 tol_val = np.max(np.abs(np.abs(V_new) - np.abs(V)))
    #                 n_iter = n + 1
    #                 V = V_new
    #
    #                 if tol_val < self.tol:
    #                     converged = True
    #                     break
    #         else:
    #             for n in range(self.max_iter):
    #                 v_recp_conj = 1.0 / np.conj(V)  # (bφ, τ)
    #
    #                 rhs = np.ascontiguousarray(v_recp_conj.T[:, :, None])  # (τ, bφ, 1)
    #                 V_new = np.matmul(F, rhs).squeeze(-1).T + W  # Batched matmul
    #
    #                 tol_val = np.max(np.abs(np.abs(V_new) - np.abs(V)))
    #                 n_iter = n + 1
    #                 V = V_new
    #
    #                 if tol_val < self.tol:
    #                     converged = True
    #                     break
    #
    #         return V, n_iter, converged, tol_val
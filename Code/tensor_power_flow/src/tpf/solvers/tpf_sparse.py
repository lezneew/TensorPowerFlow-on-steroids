"""Sparse Tensor Power Flow solvers.

Implements sparse formulation to avoid storing dense Z_B = Y_dd^-1 matrix.
Based on Section 3.5 of Salazar Duque et al. (2024).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, block_diag, hstack, vstack
from scipy.sparse.linalg import spsolve
import time

from tpf.core.network import NetworkData
from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver


class TPFSparseConstantPower(BaseSolver):
    """Sparse Tensor Power Flow for constant power loads (α_P=1, α_I=α_Z=0).
    
    Implements sparse formulation:
        M · V = V⁽*⁾⁻¹ + H
    
    where M = -diag(s*)⁻¹ @ Y_dd (constant across iterations)
          H = diag(s*)⁻¹ @ Y_ds @ v_s
    
    Key advantage: M is constant → factorize once, reuse across all τ scenarios and iterations.
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        super().__init__(tol, max_iter)
        self._M_factorized = None
        self._H = None

    # ══════════════════════════════════════════════════════════════════════
    #  Core Methods
    # ══════════════════════════════════════════════════════════════════════

    def solve(self, network: NetworkData) -> PowerFlowResult:
        """Solve single power flow (τ=1)."""
        return self.solve_batch(network, network.s_nom.reshape(-1, 1))

    def solve_batch(self, network: NetworkData, s_batch: NDArray) -> PowerFlowResult:
        """Solve τ power flows simultaneously."""
        t_start = time.perf_counter()

        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        # Build M and H (once, constant across iterations)
        M, H = self._build_system_matrices(network, s_batch)

        # Factorize M (symbolic + numeric)
        self._M_factorized = M

        # Initialize voltages (flat start)
        V = np.ones((bphi, tau), dtype=np.complex128)

        # Iteration loop
        converged = False
        n_iter = 0
        tol_val = np.inf

        for n in range(self.max_iter):
            # Build RHS: V_old⁽*⁾⁻¹ + H
            rhs = 1.0 / np.conj(V) + H

            # Solve M · V_new = rhs for each scenario
            if tau == 1:
                V_new = spsolve(M, rhs.flatten()).reshape(-1, 1)
            else:
                V_new = np.zeros((bphi, tau), dtype=np.complex128)
                for i in range(tau):
                    V_new[:, i] = spsolve(M, rhs[:, i])

            # Convergence check: max ||V_new| - |V||
            tol_val = float(np.max(np.abs(np.abs(V_new) - np.abs(V))))
            n_iter = n + 1
            V = V_new

            if tol_val < self.tol:
                converged = True
                break

        elapsed = time.perf_counter() - t_start

        # Compute slack power
        s_slack = self._compute_slack_power(network, V)

        return PowerFlowResult(
            voltages=V,
            iterations=n_iter,
            converged=converged,
            elapsed_time_s=elapsed,
            max_mismatch=tol_val,
            s_slack=s_slack,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Helper Methods
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_system_matrices(network: NetworkData, s_batch: NDArray):
        """Build block-diagonal system matrices M and H."""
        Y_dd_sparse = csr_matrix(network.Y_dd)
        Y_ds_sparse = csr_matrix(network.Y_ds)
        v_s = network.v_s.reshape(-1, 1)

        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        if tau == 1:
            # Single scenario: M is (bφ × bφ), H is (bφ × 1)
            s_conj = np.conj(s_batch[:, 0])
            inv_s_conj = 1.0 / s_conj
            M = -diags(inv_s_conj) @ Y_dd_sparse
            H = diags(inv_s_conj) * (Y_ds_sparse @ v_s)
        else:
            # Block-diagonal for τ scenarios
            M_blocks = []
            H_blocks = []

            for i in range(tau):
                s_conj_i = np.conj(s_batch[:, i])
                inv_s_conj_i = 1.0 / s_conj_i
                M_i = -diags(inv_s_conj_i) @ Y_dd_sparse
                H_i = diags(inv_s_conj_i) * (Y_ds_sparse @ v_s)
                M_blocks.append(M_i)
                H_blocks.append(H_i)

            M = block_diag(*M_blocks)
            H = vstack(H_blocks)

        return M, H

    @staticmethod
    def _compute_slack_power(network: NetworkData, V: NDArray):
        """Compute slack bus power injection."""
        if not network.has_slack_blocks:
            return None

        V_mat = V if V.ndim == 2 else V.reshape(-1, 1)
        v_s = network.v_s.reshape(-1, 1)
        I_s = network.Y_ss @ v_s + network.Y_sd @ V_mat
        return v_s * np.conj(I_s)


class TPFSparsePVMethodA(BaseSolver):
    """Sparse TPF with PV buses (outer Q-loop, Method A equivalent).
    
    Same algorithmic structure as TPFDensePVMethodA, but inner FPI
    uses sparse solver instead of dense matmul Z_B @ Λ.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter_inner: int = 50,
        max_iter_outer: int = 30,
        tol_pv: float = 1e-6,
        omega: float = 1.0,
        enforce_q_lims: bool = False,
        cold_start: bool = False,
        max_iter_inner_per_outer: int | None = None,
        adaptive_inner: bool = False,
    ):
        super().__init__(tol, max_iter_inner)
        self.max_iter_outer = max_iter_outer
        self.tol_pv = tol_pv
        self.omega = omega
        self.enforce_q_lims = enforce_q_lims
        self.cold_start = cold_start
        self.max_iter_inner_per_outer = max_iter_inner_per_outer
        self.adaptive_inner = adaptive_inner
        self.pv_info = None

    # ══════════════════════════════════════════════════════════════════════
    #  Core Methods
    # ══════════════════════════════════════════════════════════════════════

    def solve(self, network: NetworkData) -> PowerFlowResult:
        return self.solve_batch(network, network.s_nom.reshape(-1, 1))

    def solve_batch(self, network: NetworkData, s_batch: NDArray) -> PowerFlowResult:
        t_start = time.perf_counter()
        if not network.has_pv:
            return self._solve_pq_only(network, s_batch, t_start)
        return self._solve_with_pv(network, s_batch, t_start)

    # ──────────────────────────────────────────────────────────────────────
    #  PQ-only fallback
    # ──────────────────────────────────────────────────────────────────────

    def _solve_pq_only(self, network, s_batch, t_start):
        """Use TPFSparseConstantPower for PQ-only case."""
        solver = TPFSparseConstantPower(self.tol, self.max_iter)
        return solver.solve_batch(network, s_batch)

    # ──────────────────────────────────────────────────────────────────────
    #  PV support with outer Q-loop
    # ──────────────────────────────────────────────────────────────────────

    def _solve_with_pv(self, network, s_batch, t_start):
        """Solve with PV buses using outer Q-loop."""
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        pv_idx = network.pv_indices
        n_pv = len(pv_idx)
        v_spec = network.pv_v_setpoint
        v_spec_2d = v_spec.reshape(-1, 1)

        s_work = s_batch.copy()
        p_pv_fixed = s_work[pv_idx, :].real.copy()
        q_pv = np.zeros((n_pv, tau))
        s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        # Build sparse system matrices once (rebuild if α_Z ≠ 0)
        M, H = self._build_sparse_system(network, s_work)

        V = np.ones((bphi, tau), dtype=np.complex128)

        # Per-scenario tracking
        converged_mask = np.zeros(tau, dtype=bool)
        outer_iters = np.zeros(tau, dtype=np.int32)
        inner_iters = np.zeros(tau, dtype=np.int32)
        pv_v_error = np.full(tau, np.inf)

        # Q-limits
        q_min_col = None
        q_max_col = None
        if self.enforce_q_lims:
            if network.pv_q_min is not None:
                q_min_col = network.pv_q_min.reshape(-1, 1)
            if network.pv_q_max is not None:
                q_max_col = network.pv_q_max.reshape(-1, 1)

        # Precompute A_pv for Q-update (dense, small n_pv × n_pv)
        Z_B = np.linalg.inv(network.Y_dd)
        X_pp = np.imag(Z_B[np.ix_(pv_idx, pv_idx)])
        A_pv = 2.0 * X_pp + 1e-12 * np.eye(n_pv)
        A_pv_inv = np.linalg.inv(A_pv)

        for ell in range(self.max_iter_outer):
            # Rebuild M, H for current q_pv (if α_Z ≠ 0, M changes)
            s_work_active = s_work[:, ~converged_mask] if np.any(~converged_mask) else s_work
            M, H = self._build_sparse_system(network, s_work_active)

            # Inner FPI with sparse solver
            V_active = V[:, ~converged_mask] if np.any(~converged_mask) else V
            S_conj_active = np.conj(s_work_active)

            V_new, n_inner, _ = self._sparse_inner_fpi(
                M, H, S_conj_active, bphi, np.sum(~converged_mask), V_init=V_active
            )

            V[:, ~converged_mask] = V_new
            inner_iters[~converged_mask] += n_inner

            # Check PV convergence
            v_mag_pv = np.abs(V[pv_idx, :])
            err_per_col = np.max(np.abs(v_mag_pv - v_spec_2d), axis=0)
            newly_conv = (err_per_col < self.tol_pv) & (~converged_mask)
            outer_iters[newly_conv] = ell + 1
            pv_v_error[newly_conv] = err_per_col[newly_conv]
            converged_mask |= newly_conv

            if converged_mask.all():
                break

            # Q-Newton update for non-converged scenarios
            v_spec_sq_2d = (v_spec ** 2).reshape(-1, 1)
            v_mag_pv_sq = v_mag_pv ** 2
            delta_v_sq = v_spec_sq_2d - v_mag_pv_sq
            delta_q = A_pv_inv @ delta_v_sq

            # Frozen for converged scenarios
            delta_q[:, converged_mask] = 0.0

            q_pv = q_pv - self.omega * delta_q

            # Q-limits
            if q_min_col is not None:
                q_pv = np.maximum(q_pv, q_min_col)
            if q_max_col is not None:
                q_pv = np.minimum(q_pv, q_max_col)

            s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        # Final inner FPI for all scenarios
        M, H = self._build_sparse_system(network, s_work)
        V, n_inner, _ = self._sparse_inner_fpi(M, H, np.conj(s_work), bphi, tau, V_init=V)
        inner_iters += n_inner

        # Compute slack power
        s_slack = self._compute_slack_power(network, V)

        return PowerFlowResult(
            voltages=V,
            iterations=int(np.max(inner_iters)),
            converged=bool(converged_mask.all()),
            elapsed_time_s=time.perf_counter() - t_start,
            max_mismatch=float(np.max(pv_v_error[np.isfinite(pv_v_error)])) if np.any(np.isfinite(pv_v_error)) else np.inf,
            pv_indices=pv_idx,
            pv_q_pu=q_pv,
            pv_v_setpoint_pu=v_spec,
            s_slack=s_slack,
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Sparse system building
    # ──────────────────────────────────────────────────────────────────────

    def _build_sparse_system(self, network, s_work):
        """Build sparse M and H for given s_work."""
        Y_dd_sparse = csr_matrix(network.Y_dd)
        Y_ds_sparse = csr_matrix(network.Y_ds)
        v_s = network.v_s.reshape(-1, 1)

        bphi = network.n_bus_phases
        tau = s_work.shape[1]

        if tau == 1:
            s_conj = np.conj(s_work[:, 0])
            inv_s_conj = 1.0 / s_conj
            M = -diags(inv_s_conj) @ Y_dd_sparse
            H = diags(inv_s_conj) * (Y_ds_sparse @ v_s)
        else:
            M_blocks = []
            H_blocks = []
            for i in range(tau):
                s_conj_i = np.conj(s_work[:, i])
                inv_s_conj_i = 1.0 / s_conj_i
                M_i = -diags(inv_s_conj_i) @ Y_dd_sparse
                H_i = diags(inv_s_conj_i) * (Y_ds_sparse @ v_s)
                M_blocks.append(M_i)
                H_blocks.append(H_i)
            M = block_diag(*M_blocks)
            H = vstack(H_blocks)

        return M, H

    # ──────────────────────────────────────────────────────────────────────
    #  Inner FPI with sparse solver
    # ──────────────────────────────────────────────────────────────────────

    def _sparse_inner_fpi(self, M, H, S_conj, bphi, tau, V_init=None):
        """Inner FPI using sparse solver."""
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.ones((bphi, tau), dtype=np.complex128)

        converged = False
        n_iter = 0
        tol_val = np.inf

        for n in range(self.max_iter):
            # RHS: V_old⁽*⁾⁻¹ + H
            rhs = 1.0 / np.conj(V) + H

            # Solve M · V_new = rhs
            if tau == 1:
                V_new = spsolve(M, rhs.flatten()).reshape(-1, 1)
            else:
                V_new = np.zeros((bphi, tau), dtype=np.complex128)
                for i in range(tau):
                    V_new[:, i] = spsolve(M, rhs[:, i])

            tol_val = float(np.max(np.abs(np.abs(V_new) - np.abs(V))))
            n_iter = n + 1
            V = V_new

            if tol_val < self.tol:
                converged = True
                break

        return V, n_iter, converged

    # ──────────────────────────────────────────────────────────────────────
    #  Slack power computation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_slack_power(network: NetworkData, V: NDArray):
        """Compute slack bus power injection."""
        if not network.has_slack_blocks:
            return None

        V_mat = V if V.ndim == 2 else V.reshape(-1, 1)
        v_s = network.v_s.reshape(-1, 1)
        I_s = network.Y_ss @ v_s + network.Y_sd @ V_mat
        return v_s * np.conj(I_s)
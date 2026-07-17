# tensor_power_flow/src/tpf/solvers/tpf_pv_method_b.py
"""
TPF mit PV-Knoten: Methode B (Eingebettete Q-Korrektur)
=========================================================

Algorithmus (Single-Pass mit eingebetteter Korrektur):
┌─────────────────────────────────────────────────────────────┐
│  VORBERECHNUNG (einmalig):                                  │
│    A = diag(α_P ⊙ s*)                                       │
│    B = diag(α_Z ⊙ s*) + Y_dd                                │
│    c = Y_ds · v_s + α_I ⊙ s*                                │
│    F = -B⁻¹ · A                                             │
│    w = -B⁻¹ · c                                             │
│                                                             │
│  ITERATION n = 0, 1, 2, ...:                                │
│    1. FPI-Schritt:                                          │
│         v' = F @ v*^-1 + w                                  │
│                                                             │
│    2. Spannungsbetrags-Projektion an PV-Knoten:             │
│         v_k = |V_spec| · v'_k / |v'_k|  für k ∈ Ω_PV       │
│                                                             │
│    3. Q-Rückrechnung (EXAKT aus Netzgleichung):             │
│         i_k = Σ Y_dd[k,m] · v_m + Y_ds[k] @ v_s            │
│         Q_k = Im(v_k · i_k*)      ← EXAKT, nicht Approx.   │
│                                                             │
│    4. Leistungsvektor und Matrizen aktualisieren:           │
│         s_k = P_spec + j·Q_k                                │
│         A, B, c, F, w neu berechnen                         │
│                                                             │
│    5. Konvergenz prüfen:                                    │
│         max(|v_new - v|) < tol UND                          │
│         max(|V_PV| - V_spec) < tol_pv                       │
└─────────────────────────────────────────────────────────────┘

Unterschied zu Methode C:
- Methode C: Q-Korrektur via Thevenin-Empfindlichkeit (ΔQ = (V_spec² - V_calc²)/(2·X_kk))
- Methode B: Q wird EXAKT aus der Netzgleichung zurückgerechnet
- Methode B: Alle Matrizen (A, B, c, F, w) werden each Iteration aktualisiert

Vorteil: Robustere Konvergenz, keine Thevenin-Approximation nötig.
Nachteil: Höherer Rechenaufwand (Matrix-Updates each Iteration).
"""

import numpy as np
from numpy.typing import NDArray
import time
from dataclasses import dataclass, field

from tpf.core.network import NetworkData
from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver


@dataclass
class PVConvergenceInfoB:
    """Detaillierte Konvergenz-Informationen für Methode B."""
    iterations: int
    pv_v_error_final: float
    pv_q_final: NDArray
    pv_v_final: NDArray
    converged: bool

    contraction_factor_history: list[float] = field(default_factory=list)
    spectral_radius_history: list[float] = field(default_factory=list)
    pv_v_error_history: list[float] = field(default_factory=list)
    voltage_change_history: list[float] = field(default_factory=list)
    v_history: list[NDArray] = field(default_factory=list)
    q_history: list[NDArray] = field(default_factory=list)


class TPFDensePVMethodB(BaseSolver):
    """
    Tensor Power Flow mit PV-Knoten.
    Methode B: Eingebettete Q-Korrektur (Single-Pass).

    Im Gegensatz zu Methode A (äußere Q-Schleife) und Methode C (vereinfachte
    Thevenin-Korrektur) wird hier die Blindleistung EXAKT aus der Netzgleichung
    zurückgerechnet, und alle Hilfsmatrizen werden each Iteration aktualisiert.

    Parameters
    ----------
    tol : float
        Konvergenztoleranz für Spannungsänderung.
    max_iter : int
        Maximale Iterationen.
    tol_pv : float
        Konvergenztoleranz für |V| an PV-Knoten.
    enforce_q_lims : bool
        Wenn True: Q-Grenzen aus NetworkData einhalten (Clamping).
    track_convergence : bool
        Wenn True: Speichert Kontractionsfaktor und Spektralradius each Iteration.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 100,
        tol_pv: float = 1e-5,
        enforce_q_lims: bool = True,
        track_convergence: bool = True,
        omega: float = 1.0,
        omega_q: float = 1.0,
    ):
        super().__init__(tol, max_iter)
        self.tol_pv = tol_pv
        self.enforce_q_lims = enforce_q_lims
        self.track_convergence = track_convergence
        self.omega = omega
        self.omega_q = omega_q
        self.pv_info: PVConvergenceInfoB | None = None

    def solve(self, network: NetworkData) -> PowerFlowResult:
        """Solve single power flow with PV buses."""
        return self.solve_batch(network, network.s_nom.reshape(-1, 1))

    def solve_batch(
        self, network: NetworkData, s_batch: NDArray
    ) -> PowerFlowResult:
        """Solve power flows with PV support using Method B."""
        t_start = time.perf_counter()

        if not network.has_pv:
            return self._solve_pq_only(network, s_batch, t_start)

        result = self._solve_with_pv(network, s_batch, t_start)
        return result

    def _solve_pq_only(
        self, network: NetworkData, s_batch: NDArray, t_start: float
    ) -> PowerFlowResult:
        """Standard TPF without PV buses."""
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        K, L = self._precompute_constant_power(network)
        V, n_iter, converged, tol_val, _ = self._inner_fpi(
            K, L, np.conj(s_batch), bphi, tau, collect_history=False
        )

        elapsed = time.perf_counter() - t_start
        self.pv_info = None

        return PowerFlowResult(
            voltages=V,
            iterations=n_iter,
            converged=converged,
            elapsed_time_s=elapsed,
            max_mismatch=tol_val,
        )

    def _solve_with_pv(
        self, network: NetworkData, s_batch: NDArray, t_start: float
    ) -> PowerFlowResult:
        """Core Algorithm: Method B - Embedded Q-Correction."""
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        pv_idx = network.pv_indices
        n_pv = len(pv_idx)
        v_spec = network.pv_v_setpoint

        q_min = None
        q_max = None
        if self.enforce_q_lims:
            if network.pv_q_min is not None:
                q_min = network.pv_q_min
            if network.pv_q_max is not None:
                q_max = network.pv_q_max

        s_work = s_batch.copy()
        p_pv_fixed = s_work[pv_idx, :].real.copy()
        q_pv = np.zeros((n_pv, tau))
        s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        V = np.ones((bphi, tau), dtype=np.complex128)

        converged = False
        n_iter = 0
        pv_v_error = np.inf

        contraction_history = []
        spectral_radius_history = []
        pv_v_error_history = []
        voltage_change_history = []
        v_history = []
        q_history = []

        prev_delta_v = None

        for n in range(self.max_iter):
            A, B, c, B_inv, F, w = self._compute_matrices(
                network, s_work, bphi, tau
            )

            V_recp = 1.0 / np.conj(V)
            V_raw = F @ V_recp + w.reshape(-1, 1)
            V_new = (1 - self.omega) * V + self.omega * V_raw

            if tau == 1 and V_new.ndim == 1:
                V_new = V_new.reshape(-1, 1)

            voltage_change = float(np.max(np.abs(np.abs(V_new) - np.abs(V))))
            voltage_change_history.append(voltage_change)

            contraction_factor = 0.0
            if prev_delta_v is not None and prev_delta_v > 1e-15:
                contraction_factor = voltage_change / prev_delta_v
            contraction_history.append(contraction_factor)
            prev_delta_v = voltage_change

            if self.track_convergence and n > 0:
                spec_rad = self._compute_spectral_radius_approx(
                    network, s_work, V, pv_idx
                )
                spectral_radius_history.append(spec_rad)

            v_mag_pv_before = np.abs(V_new[pv_idx, :])
            pv_v_error = np.max(np.abs(v_mag_pv_before - v_spec.reshape(-1, 1)))
            pv_v_error_history.append(float(pv_v_error))

            if voltage_change < self.tol and pv_v_error < self.tol_pv:
                for k in range(n_pv):
                    pv_k_idx = pv_idx[k]
                    V_new[pv_k_idx, :] = v_spec[k] * V_new[pv_k_idx, :] / np.abs(V_new[pv_k_idx, :])
                v_history.append(V_new.copy())

                q_pv_new_conv = np.zeros((n_pv, tau))
                for k in range(n_pv):
                    pv_k_idx = pv_idx[k]
                    i_k = (network.Y_dd[pv_k_idx, :] @ V_new[:, 0]) + (network.Y_ds[pv_k_idx, :] @ network.v_s)
                    s_calc = V_new[pv_k_idx, 0] * np.conj(i_k)
                    q_pv_new_conv[k, 0] = -s_calc.imag
                q_pv = q_pv_new_conv
                q_history.append(q_pv.copy())

                converged = True
                n_iter = n + 1
                V = V_new
                break

            for k in range(n_pv):
                pv_k_idx = pv_idx[k]
                V_new[pv_k_idx, :] = v_spec[k] * V_new[pv_k_idx, :] / np.abs(V_new[pv_k_idx, :])

            v_history.append(V_new.copy())

            q_pv_new = np.zeros((n_pv, tau))
            for k in range(n_pv):
                pv_k_idx = pv_idx[k]
                i_k = (network.Y_dd[pv_k_idx, :] @ V_new[:, 0]) + (network.Y_ds[pv_k_idx, :] @ network.v_s)
                s_calc = V_new[pv_k_idx, 0] * np.conj(i_k)
                q_pv_new[k, 0] = -s_calc.imag

            if q_min is not None:
                q_pv_new = np.maximum(q_pv_new, q_min.reshape(-1, 1))
            if q_max is not None:
                q_pv_new = np.minimum(q_pv_new, q_max.reshape(-1, 1))

            q_pv = (1 - self.omega_q) * q_pv + self.omega_q * q_pv_new
            q_history.append(q_pv.copy())

            s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

            n_iter = n + 1
            V = V_new

        if not converged:
            n_iter = self.max_iter

        elapsed = time.perf_counter() - t_start

        self.pv_info = PVConvergenceInfoB(
            iterations=n_iter,
            pv_v_error_final=pv_v_error,
            pv_q_final=q_pv[:, 0].copy() if tau == 1 else q_pv.copy(),
            pv_v_final=np.abs(V[pv_idx, :]).copy(),
            converged=converged,
            contraction_factor_history=contraction_history,
            spectral_radius_history=spectral_radius_history,
            pv_v_error_history=pv_v_error_history,
            voltage_change_history=voltage_change_history,
            v_history=v_history,
            q_history=q_history,
        )

        return PowerFlowResult(
            voltages=V,
            iterations=n_iter,
converged=converged,
            elapsed_time_s=elapsed,
            max_mismatch=pv_v_error,
            pv_indices=pv_idx,
            pv_q_pu=q_pv[:, 0] if tau == 1 else q_pv,
            pv_v_setpoint_pu=v_spec,
        )

    def _compute_decoupled_start(
        self, network: NetworkData, s_batch: NDArray
    ) -> NDArray:
        """
        Compute initial voltage using decoupled P-θ solution (DC load flow).
        This provides a better starting point than flat start (V=1.0).
        """
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        Y = network.Y_dd
        v_s = network.v_s

        B_matrix = np.imag(Y)
        np.fill_diagonal(B_matrix, 0)

        P_vec = s_batch[:, 0].real
        for idx in network.pv_indices:
            P_vec[idx] = 0

        slack_idx = 0
        P_vec[slack_idx] = 0

        try:
            B_reduced = np.delete(np.delete(B_matrix, slack_idx, axis=0), slack_idx, axis=1)
            P_reduced = np.delete(P_vec, slack_idx)

            theta_reduced = np.linalg.solve(B_reduced, P_reduced)

            theta = np.zeros(bphi)
            theta[slack_idx] = 0
            mask = np.ones(bphi, dtype=bool)
            mask[slack_idx] = False
            theta[mask] = theta_reduced

        except np.linalg.LinAlgError:
            theta = np.zeros(bphi)

        V = np.ones((bphi, tau), dtype=np.complex128)
        for i in range(bphi):
            V[i, 0] = 1.0 * np.exp(1j * theta[i])

        if tau > 1:
            for t in range(1, tau):
                V[:, t] = V[:, 0]

        return V

    def _compute_matrices(
        self, network: NetworkData, s_work: NDArray, bphi: int, tau: int
    ):
        """Compute helper matrices A, B, c, F, w."""
        alpha_p = network.alpha_p
        alpha_i = network.alpha_i
        alpha_z = network.alpha_z

        A = np.diag(alpha_p * np.conj(s_work[:, 0]))

        if np.any(alpha_z != 0):
            B = np.diag(alpha_z * np.conj(s_work[:, 0])) + network.Y_dd
        else:
            B = network.Y_dd.copy()

        B_inv = np.linalg.inv(B)

        c_base = network.Y_ds @ network.v_s
        if np.any(alpha_i != 0):
            c = c_base + alpha_i * np.conj(s_work[:, 0])
        else:
            c = c_base

        F = -B_inv @ A
        w = -B_inv @ c

        return A, B, c, B_inv, F, w

    def _precompute_constant_power(self, network: NetworkData):
        """Precompute K and L for constant power case (no PV)."""
        Z_B = np.linalg.inv(network.Y_dd)
        K = -Z_B
        L = K @ network.Y_ds @ network.v_s
        return K, L

    def _inner_fpi(
        self,
        K: NDArray,
        L: NDArray,
        S_conj: NDArray,
        bphi: int,
        tau: int,
        collect_history: bool = False,
    ) -> tuple[NDArray, int, bool, float, list[float]]:
        """Standard Fixed-Point Iteration."""
        V = np.ones((bphi, tau), dtype=np.complex128)
        L_col = L.reshape(-1, 1)

        converged = False
        n_iter = 0
        tol_val = np.inf
        history = []

        for n in range(self.max_iter):
            LAMBDA = S_conj * (1.0 / np.conj(V))
            V_new = K @ LAMBDA + L_col

            tol_val = float(np.max(np.abs(np.abs(V_new) - np.abs(V))))

            n_iter = n + 1
            V = V_new

            if collect_history:
                history.append(tol_val)

            if tol_val < self.tol:
                converged = True
                break

        return V, n_iter, converged, tol_val, history

    def _compute_spectral_radius_approx(
        self,
        network: NetworkData,
        s_work: NDArray,
        V: NDArray,
        pv_idx: NDArray,
    ) -> float:
        """Approximate spectral radius of the iteration matrix."""
        try:
            bphi = network.n_bus_phases
            n_pv = len(pv_idx)

            A, B, c, B_inv, F, w = self._compute_matrices(
                network, s_work, bphi, 1
            )

            v_squared = np.conj(V[:, 0]) * np.conj(V[:, 0])
            J_approx = F * (np.conj(s_work[:, 0]) / v_squared).T

            if n_pv > 0:
                J_pv = J_approx[np.ix_(pv_idx, pv_idx)]
                eigenvalues = np.linalg.eigvals(J_pv)
            else:
                eigenvalues = np.linalg.eigvals(J_approx)

            spec_rad = float(np.max(np.abs(eigenvalues)))
            return spec_rad
        except:
            return 0.0
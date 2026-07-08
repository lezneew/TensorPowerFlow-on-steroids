# tensor_power_flow/src/tpf/solvers/tpf_pv_method_a.py
"""
TPF mit PV-Knoten: Methode A (Äußere Q-Schleife)
==================================================
Erweitert um τ-parallele Zeitreihenberechnung mit Chunking.
"""

import numpy as np
from numpy.typing import NDArray
import time
from dataclasses import dataclass, field

from tpf.core.network import NetworkData
from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver


@dataclass
class PVConvergenceInfo:
    outer_iterations: int
    inner_iterations_total: int
    inner_iterations_per_outer: list[int]
    pv_v_error_final: float
    pv_q_final: NDArray
    pv_v_final: NDArray
    converged_inner: bool
    converged_outer: bool
    pv_v_error_history: list[float] = None
    v_change_history: list[float] = None
    inner_v_change_all: list[float] = field(default_factory=list)
    outer_start_indices: list[int] = field(default_factory=list)

    # NEU: Per-Szenario-Tracking (für τ > 1)
    outer_iterations_per_scenario: NDArray | None = None   # (τ,) int32
    inner_iterations_per_scenario: NDArray | None = None   # (τ,) int32
    pv_v_error_per_scenario:      NDArray | None = None    # (τ,) float64
    converged_per_scenario:       NDArray | None = None    # (τ,) bool
    n_scenarios: int = 1
    n_converged_scenarios: int = 1


class TPFDensePVMethodA(BaseSolver):
    def __init__(
        self,
        tol: float = 1e-6,
        max_iter_inner: int = 50,
        max_iter_outer: int = 30,
        tol_pv: float = 1e-5,
        omega: float = 1.0,
        enforce_q_lims: bool = False,
        cold_start: bool = False,
    ):
        super().__init__(tol, max_iter_inner)
        self.max_iter_outer = max_iter_outer
        self.tol_pv = tol_pv
        self.omega = omega
        self.enforce_q_lims = enforce_q_lims
        self.cold_start = cold_start
        self.pv_info: PVConvergenceInfo | None = None

    # ══════════════════════════════════════════════════════════════════
    #  Öffentliche API
    # ══════════════════════════════════════════════════════════════════

    def solve(self, network: NetworkData) -> PowerFlowResult:
        return self.solve_batch(network, network.s_nom.reshape(-1, 1))

    def solve_batch(self, network: NetworkData, s_batch: NDArray) -> PowerFlowResult:
        t_start = time.perf_counter()
        if not network.has_pv:
            return self._solve_pq_only(network, s_batch, t_start)
        return self._solve_with_pv(network, s_batch, t_start)

    def solve_timeseries(
        self,
        network: NetworkData,
        s_batch: NDArray,
        chunk_size: int | None = None,
        verbose: bool = False,
    ) -> PowerFlowResult:
        """
        Löst τ Lastflüsse mit automatischem Chunking (für τ > 50k).

        Parameters
        ----------
        network : NetworkData
        s_batch : NDArray, shape (bφ, τ)
            Zeitabhängige Leistungsinjektionen.
            - PQ-Knoten:  P(t) + j·Q(t)
            - PV-Knoten:  P(t) + j·0   (Q wird gelöst)
        chunk_size : int | None
            Anzahl Szenarien pro Chunk. None → auto-tune.
        verbose : bool

        Returns
        -------
        PowerFlowResult mit voltages=(bφ, τ) und per-Szenario-Konvergenz.
        """
        t_start = time.perf_counter()

        bphi = network.n_bus_phases
        if s_batch.ndim == 1:
            s_batch = s_batch.reshape(-1, 1)
        tau = s_batch.shape[1]

        if chunk_size is None:
            chunk_size = self._auto_chunk_size(bphi, tau)
        chunk_size = max(1, min(chunk_size, tau))

        # Vorberechnung EINMALIG für alle Chunks
        K, L = self._precompute(network)
        Z_B = -K

        # Output-Arrays
        V_all = np.zeros((bphi, tau), dtype=np.complex128)
        n_pv = network.n_pv
        q_pv_all = np.zeros((n_pv, tau), dtype=np.float64) if network.has_pv else None
        converged_ps = np.zeros(tau, dtype=bool)
        outer_iters_ps = np.zeros(tau, dtype=np.int32)
        inner_iters_ps = np.zeros(tau, dtype=np.int32)
        pv_v_err_ps = np.full(tau, np.inf, dtype=np.float64)

        n_chunks = (tau + chunk_size - 1) // chunk_size

        for c in range(n_chunks):
            s_start = c * chunk_size
            s_end = min(s_start + chunk_size, tau)

            s_chunk = s_batch[:, s_start:s_end]

            if network.has_pv:
                V_c, q_c, info_c = self._solve_chunk_with_pv(
                    network, s_chunk, K, L, Z_B
                )
                q_pv_all[:, s_start:s_end] = q_c
            else:
                V_c, info_c = self._solve_chunk_pq_only(
                    network, s_chunk, K, L
                )

            V_all[:, s_start:s_end] = V_c
            converged_ps[s_start:s_end] = info_c["converged"]
            outer_iters_ps[s_start:s_end] = info_c["outer_iters"]
            inner_iters_ps[s_start:s_end] = info_c["inner_iters"]
            pv_v_err_ps[s_start:s_end] = info_c["pv_v_error"]

            if verbose:
                n_conv = int(np.sum(converged_ps[:s_end]))
                print(f"  Chunk {c+1:>3}/{n_chunks} "
                      f"[{s_start:>6}:{s_end:>6}] "
                      f"conv {n_conv}/{s_end} "
                      f"({100*n_conv/s_end:5.1f}%)")

        elapsed = time.perf_counter() - t_start
        n_conv_total = int(np.sum(converged_ps))
        n_div = tau - n_conv_total
        # V_all has shape (bφ, τ)
        I_s = network.Y_ss @ network.v_s.reshape(-1, 1) + network.Y_sd @ V_all
        s_slack = network.v_s.reshape(-1, 1) * np.conj(I_s)  # (1, τ)

        if n_div > 0:
            div_idx = np.where(~converged_ps)[0]
            print(f"  [WARN] {n_div}/{tau} Szenarien nicht konvergiert. "
                  f"Erste 5 Indizes: {div_idx[:5].tolist()}")

        self.pv_info = PVConvergenceInfo(
            outer_iterations=int(np.max(outer_iters_ps)) if tau > 0 else 0,
            inner_iterations_total=int(np.sum(inner_iters_ps)),
            inner_iterations_per_outer=[],
            pv_v_error_final=float(np.max(pv_v_err_ps[np.isfinite(pv_v_err_ps)]))
                              if np.any(np.isfinite(pv_v_err_ps)) else np.inf,
            pv_q_final=q_pv_all if q_pv_all is not None else np.zeros((0, tau)),
            pv_v_final=(np.abs(V_all[network.pv_indices, :])
                        if network.has_pv else np.zeros((0, tau))),
            converged_inner=True,
            converged_outer=bool(n_conv_total == tau),
            outer_iterations_per_scenario=outer_iters_ps,
            inner_iterations_per_scenario=inner_iters_ps,
            pv_v_error_per_scenario=pv_v_err_ps,
            converged_per_scenario=converged_ps,
            n_scenarios=tau,
            n_converged_scenarios=n_conv_total,
        )

        return PowerFlowResult(
            voltages=V_all,
            iterations=int(np.max(inner_iters_ps)) if tau > 0 else 0,
            converged=bool(n_conv_total == tau),
            elapsed_time_s=elapsed,
            max_mismatch=self.pv_info.pv_v_error_final,
            pv_indices=network.pv_indices if network.has_pv else None,
            pv_q_pu=q_pv_all,
            pv_v_setpoint_pu=network.pv_v_setpoint,
            s_slack=s_slack,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Chunk-Solver: PQ-only
    # ══════════════════════════════════════════════════════════════════

    def _solve_chunk_pq_only(self, network, s_chunk, K, L):
        bphi = network.n_bus_phases
        tau = s_chunk.shape[1]
        V, n_iter, conv, tol_val, _ = self._inner_fpi(
            K, L, np.conj(s_chunk), bphi, tau, collect_history=False
        )
        return V, {
            "converged": np.full(tau, conv, dtype=bool),
            "outer_iters": np.zeros(tau, dtype=np.int32),
            "inner_iters": np.full(tau, n_iter, dtype=np.int32),
            "pv_v_error": np.full(tau, tol_val, dtype=np.float64),
        }

    # ══════════════════════════════════════════════════════════════════
    #  Chunk-Solver: mit PV (Kern-Algorithmus mit per-Szenario-Masken)
    # ══════════════════════════════════════════════════════════════════

    def _solve_chunk_with_pv(self, network, s_chunk, K, L, Z_B):
        bphi = network.n_bus_phases
        tau = s_chunk.shape[1]

        pv_idx = network.pv_indices
        n_pv = len(pv_idx)
        v_spec = network.pv_v_setpoint
        v_spec_2d = v_spec.reshape(-1, 1)         # (n_pv, 1)
        v_spec_sq_2d = (v_spec ** 2).reshape(-1, 1)

        X_th = np.imag(Z_B[pv_idx, pv_idx])
        X_th_safe = np.where(np.abs(X_th) > 1e-10, X_th, 1e-10)
        X_th_col = X_th_safe.reshape(-1, 1)

        s_work = s_chunk.copy()
        p_pv_fixed = s_work[pv_idx, :].real.copy()   # (n_pv, τ) time-varying P
        q_pv = np.zeros((n_pv, tau))
        s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        V = np.ones((bphi, tau), dtype=np.complex128)

        # Per-Szenario Tracking
        converged_mask = np.zeros(tau, dtype=bool)
        outer_iters = np.zeros(tau, dtype=np.int32)
        inner_iters = np.zeros(tau, dtype=np.int32)
        pv_v_error = np.full(tau, np.inf)

        # Q-Limits (STATISCH, broadcast auf (n_pv, 1))
        q_min_col = None
        q_max_col = None
        if self.enforce_q_lims:
            if network.pv_q_min is not None:
                q_min_col = network.pv_q_min.reshape(-1, 1)
            if network.pv_q_max is not None:
                q_max_col = network.pv_q_max.reshape(-1, 1)

        err_per_col = np.full(tau, np.inf)

        for ell in range(self.max_iter_outer):
            if self.cold_start:
                V = np.ones((bphi, tau), dtype=np.complex128)

            S_conj = np.conj(s_work)
            V, n_inner, _, _, _ = self._inner_fpi(
                K, L, S_conj, bphi, tau, V_init=V, collect_history=False
            )

            # Inner-Iterationen NUR für noch aktive Szenarien zählen
            active = ~converged_mask
            inner_iters[active] += n_inner

            # PV-Fehler pro Spalte
            v_mag_pv = np.abs(V[pv_idx, :])                       # (n_pv, τ)
            err_per_col = np.max(np.abs(v_mag_pv - v_spec_2d), axis=0)  # (τ,)

            # Neu konvergierte Szenarien
            newly_conv = (err_per_col < self.tol_pv) & (~converged_mask)
            outer_iters[newly_conv] = ell + 1
            pv_v_error[newly_conv] = err_per_col[newly_conv]
            converged_mask |= newly_conv

            if converged_mask.all():
                break

            # Q-Update (Thévenin)
            delta_q = (v_spec_sq_2d - v_mag_pv ** 2) / (2.0 * X_th_col)

            # Q einfrieren für bereits konvergierte Szenarien
            delta_q[:, converged_mask] = 0.0

            q_pv = q_pv - self.omega * delta_q

            # Q-Grenzen (v1: CLIP-ONLY)
            if q_min_col is not None:
                q_pv = np.maximum(q_pv, q_min_col)
            if q_max_col is not None:
                q_pv = np.minimum(q_pv, q_max_col)

            s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        # Log & continue: nicht-konvergierte Szenarien behalten letzten Zustand
        still_div = ~converged_mask
        outer_iters[still_div] = self.max_iter_outer
        pv_v_error[still_div] = err_per_col[still_div]

        return V, q_pv, {
            "converged": converged_mask,
            "outer_iters": outer_iters,
            "inner_iters": inner_iters,
            "pv_v_error": pv_v_error,
        }

    # ══════════════════════════════════════════════════════════════════
    #  Chunk-Auto-Tune
    # ══════════════════════════════════════════════════════════════════

    def _auto_chunk_size(self, bphi: int, tau: int) -> int:
        """
        Auto-Tune für Chunking. CPU-only, konservativ.
        Ziel: peak RAM < 40% des verfügbaren.
        """
        try:
            import psutil
            available = psutil.virtual_memory().available
        except Exception:
            available = 8 * 1024**3   # Fallback: 8 GB

        # Hot Tensors pro Chunk: V, V_new, LAMBDA, S_conj, s_work, temp ≈ 6
        # complex128 = 16 bytes
        n_hot = 6
        bytes_per_col = bphi * 16 * n_hot
        max_from_mem = int(0.4 * available / bytes_per_col)

        # Sinnvolle Ober-/Untergrenze
        return int(np.clip(max_from_mem, 1024, 100_000))

    # ══════════════════════════════════════════════════════════════════
    #  Fallbacks (unverändert)
    # ══════════════════════════════════════════════════════════════════

    def _solve_pq_only(self, network, s_batch, t_start):
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        K, L = self._precompute(network)
        V, n_iter, converged, tol_val, _ = self._inner_fpi(
            K, L, np.conj(s_batch), bphi, tau, collect_history=False
        )

        elapsed = time.perf_counter() - t_start
        self.pv_info = None

        s_slack = self._compute_slack_power(network, V)

        return PowerFlowResult(
            voltages=V,
            iterations=n_iter,
            converged=converged,
            elapsed_time_s=elapsed,
            max_mismatch=tol_val,
            s_slack=s_slack,  # NEU
        )

    def _solve_with_pv(self, network, s_batch, t_start):
        """Bestehende Einzel-Solve-Logik (unverändert für τ=1 Regression)."""
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        K, L = self._precompute(network)
        Z_B = -K

        pv_idx = network.pv_indices
        n_pv = len(pv_idx)
        v_spec = network.pv_v_setpoint
        X_th = np.imag(Z_B[pv_idx, pv_idx])
        X_th_safe = np.where(np.abs(X_th) > 1e-10, X_th, 1e-10)

        s_work = s_batch.copy()
        p_pv_fixed = s_work[pv_idx, :].real.copy()
        q_pv = np.zeros((n_pv, tau))
        s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        V = np.ones((bphi, tau), dtype=np.complex128)
        converged_outer = False
        converged_inner = False
        outer_iter = 0
        inner_iter_total = 0
        inner_iter_log = []
        pv_v_error = np.inf

        q_min = network.pv_q_min.reshape(-1, 1) * np.ones((1, tau)) \
            if (self.enforce_q_lims and network.pv_q_min is not None) else None
        q_max = network.pv_q_max.reshape(-1, 1) * np.ones((1, tau)) \
            if (self.enforce_q_lims and network.pv_q_max is not None) else None

        pv_v_error_history = []
        v_change_history = []
        inner_v_change_all = []
        outer_start_indices = []

        for ell in range(self.max_iter_outer):
            outer_iter = ell + 1
            if self.cold_start:
                V = np.ones((bphi, tau), dtype=np.complex128)
            outer_start_indices.append(len(inner_v_change_all))

            S_conj = np.conj(s_work)
            V, n_inner, converged_inner, tol_inner, inner_history = self._inner_fpi(
                K, L, S_conj, bphi, tau, V_init=V, collect_history=True
            )
            inner_iter_total += n_inner
            inner_iter_log.append(n_inner)
            inner_v_change_all.extend(inner_history)

            if not converged_inner:
                break

            v_mag_pv = np.abs(V[pv_idx, :])
            v_spec_2d = v_spec.reshape(-1, 1)
            pv_v_error = np.max(np.abs(v_mag_pv - v_spec_2d))
            pv_v_error_history.append(float(pv_v_error))
            v_change_history.append(float(tol_inner))

            if pv_v_error < self.tol_pv:
                converged_outer = True
                break

            delta_q = ((v_spec_2d ** 2 - v_mag_pv ** 2)
                       / (2.0 * X_th_safe.reshape(-1, 1)))
            q_pv = q_pv - self.omega * delta_q

            if self.enforce_q_lims:
                if q_min is not None: q_pv = np.maximum(q_pv, q_min)
                if q_max is not None: q_pv = np.minimum(q_pv, q_max)

            s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        elapsed = time.perf_counter() - t_start
        self.pv_info = PVConvergenceInfo(
            outer_iterations=outer_iter,
            inner_iterations_total=inner_iter_total,
            inner_iterations_per_outer=inner_iter_log,
            pv_v_error_final=pv_v_error,
            pv_q_final=q_pv.copy(),
            pv_v_final=np.abs(V[pv_idx, :]).copy(),
            converged_inner=converged_inner,
            converged_outer=converged_outer,
            pv_v_error_history=pv_v_error_history,
            v_change_history=v_change_history,
            inner_v_change_all=inner_v_change_all,
            outer_start_indices=outer_start_indices,
        )
        s_slack = self._compute_slack_power(network, V)
        return PowerFlowResult(
            voltages=V,
            iterations=inner_iter_total,
            converged=converged_inner and converged_outer,
            elapsed_time_s=elapsed,
            max_mismatch=pv_v_error,
            pv_indices=pv_idx,
            pv_q_pu=q_pv[:, 0] if tau == 1 else q_pv,
            pv_v_setpoint_pu=v_spec,
            s_slack=s_slack,
        )

    @staticmethod
    def _precompute(network: NetworkData):
        Z_B = np.linalg.inv(network.Y_dd)
        K = -Z_B
        L = K @ network.Y_ds @ network.v_s
        return K, L

    @staticmethod
    def _compute_slack_power(
            network: NetworkData, V: NDArray
    ) -> NDArray | None:
        """
        Berechnet die vom Slack eingespeiste Scheinleistung.

            I_s      = Y_ss · v_s + Y_sd · V         (φ × τ)
            S_slack  = v_s ⊙ conj(I_s)               (φ × τ)

        S_slack[k, i] > 0  →  Slack liefert Leistung (import)
        S_slack[k, i] < 0  →  Slack absorbiert (export, z.B. PV-Überschuss)

        Returns
        -------
        s_slack : (φ, τ) oder None wenn Y_ss/Y_sd fehlen.
        """
        if not network.has_slack_blocks:
            return None

        V_mat = V if V.ndim == 2 else V.reshape(-1, 1)
        v_s = network.v_s.reshape(-1, 1)  # (φ, 1)
        I_s = network.Y_ss @ v_s + network.Y_sd @ V_mat  # (φ, τ)
        return v_s * np.conj(I_s)  # (φ, τ)


    def _inner_fpi(self, K, L, S_conj, bphi, tau,
                   V_init=None, collect_history=False):
        if V_init is not None:
            V = V_init.copy()
        else:
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
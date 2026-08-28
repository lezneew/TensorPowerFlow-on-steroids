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

    inner_fpi_time_ms: list[float] = field(default_factory=list)
    total_inner_fpi_time_ms: float = 0.0

    # NEU: Per-Szenario-Tracking (für τ > 1)
    outer_iterations_per_scenario: NDArray | None = None   # (τ,) int32
    inner_iterations_per_scenario: NDArray | None = None   # (τ,) int32
    pv_v_error_per_scenario:      NDArray | None = None    # (τ,) float64
    converged_per_scenario:       NDArray | None = None    # (τ,) bool
    n_scenarios: int = 1
    n_converged_scenarios: int = 1

    # Per-Szenario-Tracking (τ > 1)
    v_min_per_scenario: NDArray | None = None      # (τ,)
    v_max_per_scenario: NDArray | None = None      # (τ,)

    # Timing (ms)
    t_precompute_ms: float = 0.0                   # Y_dd^-1, L, (2X_pp)^-1
    t_solve_ms: float = 0.0                        # alle Chunks
    t_chunks_ms: list[float] = field(default_factory=list)
    chunk_size: int = 0
    n_chunks: int = 0
    n_inner_calls: int = 0                         # Zahl der _inner_fpi-Aufrufe
    n_gemm: int = 0                                # Zahl ausgeführter GEMMs
    work_executed_cols: int = 0                    # Σ GEMM · Spaltenzahl
    work_useful_cols: int = 0                      # Σ_szen. eigene inner iters
    flops_gemm: float = 0.0                        # 8·bφ²·Spalten·GEMM
    active_per_outer: list[int] = field(default_factory=list)  # über Chunks summiert
    warm_mode: str = "flat"

    # X_pp-Diagnostik
    cond_Xpp: float = float("nan")
    rho_jacobi: float = float("nan")
    min_diag_off: float = float("nan")


class TPFDensePVMethodA(BaseSolver):
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
        use_decoupled: bool = False,
    ):
        super().__init__(tol, max_iter_inner)
        self.max_iter_outer = max_iter_outer
        self.tol_pv = tol_pv
        self.omega = omega
        self.enforce_q_lims = enforce_q_lims
        self.cold_start = cold_start
        self.max_iter_inner_per_outer = max_iter_inner_per_outer
        self.adaptive_inner = adaptive_inner
        self.use_decoupled = use_decoupled
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
        V_init: NDArray | None = None,      # (bφ,) oder (bφ,τ)
        q_init: NDArray | None = None,      # (n_pv,) oder (n_pv,τ)
        warm_mode: str = "flat",            # flat|provided|carry_last|carry_mean
        collect_scenario_stats: bool = True,
        diagnostics: bool = True,
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
        V_init : NDArray | None
            Initial guess for voltages. Shape (bφ,) or (bφ, τ).
        q_init : NDArray | None
            Initial guess for PV reactive power. Shape (n_pv,) or (n_pv, τ).
        warm_mode : str
            "flat": use V_init/q_init for all chunks (or ones if None)
            "provided": use V_init/q_init only for first chunk
            "carry_last": carry V/q from last column of previous chunk
            "carry_mean": carry mean V/q from previous chunk
        collect_scenario_stats : bool
            Collect per-scenario statistics (v_min, v_max).
        diagnostics : bool
            Compute X_pp diagnostics (cond, rho_jacobi, etc.).

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
        chunk_size = max(1, min(int(chunk_size), tau))

        # Vorberechnung EINMALIG für alle Chunks
        t_pre0 = time.perf_counter()
        K, L = self._precompute(network)
        Z_B = -K

        A_pv_inv = None
        cond_Xpp = rho_j = min_ratio = float("nan")
        if network.has_pv:
            pv_pre = network.pv_indices
            X_pp = np.imag(Z_B[np.ix_(pv_pre, pv_pre)])
            n_pv_pre = len(pv_pre)
            A_pv = 2.0 * X_pp + 1e-12 * np.eye(n_pv_pre)
            A_pv_inv = np.linalg.inv(A_pv)
        t_precompute_ms = (time.perf_counter() - t_pre0) * 1e3

        # Diagnostik NACH der Zeitmessung (verfälscht t_0 sonst)
        if diagnostics and network.has_pv and network.n_pv > 1:
            pv = network.pv_indices
            X_pp = np.imag(Z_B[np.ix_(pv, pv)])
            diag = np.abs(np.diag(X_pp))
            off = np.sum(np.abs(X_pp), axis=1) - diag
            min_ratio = float((diag / np.maximum(off, 1e-12)).min())
            cond_Xpp = float(np.linalg.cond(X_pp))
            rho_j = float(np.max(np.abs(np.linalg.eigvals(
                np.eye(len(pv)) - np.diag(1.0 / diag) @ X_pp))))
            if verbose:
                print(f"  [PV diag] min(diag/off)={min_ratio:.3f}  "
                      f"cond(X_pp)={cond_Xpp:.1f}  rho_J={rho_j:.3f}")

        # Output-Arrays
        V_all = np.zeros((bphi, tau), dtype=np.complex128)
        n_pv = network.n_pv
        q_pv_all = np.zeros((n_pv, tau), dtype=np.float64) if network.has_pv else None
        converged_ps = np.zeros(tau, dtype=bool)
        outer_iters_ps = np.zeros(tau, dtype=np.int32)
        inner_iters_ps = np.zeros(tau, dtype=np.int32)
        pv_v_err_ps = np.full(tau, np.inf, dtype=np.float64)
        vmin_ps = np.full(tau, np.nan)
        vmax_ps = np.full(tau, np.nan)

        # Warm-Start-Zustand
        V_carry = None
        q_carry = None
        if warm_mode == "provided" and V_init is not None:
            V_carry = np.asarray(V_init).reshape(bphi, -1)[:, 0].copy()
            if q_init is not None and network.has_pv:
                q_carry = np.asarray(q_init).reshape(n_pv, -1)[:, 0].copy()

        n_chunks = (tau + chunk_size - 1) // chunk_size
        t_chunks_ms: list[float] = []
        n_inner_calls = n_gemm = 0
        work_exec = 0
        flops = 0.0
        active_hist: dict[int, int] = {}

        t_solve0 = time.perf_counter()
        for c in range(n_chunks):
            s0, s1 = c * chunk_size, min((c + 1) * chunk_size, tau)
            cols = s1 - s0
            s_chunk = s_batch[:, s0:s1]

            V_init_chunk = None
            if V_carry is not None:
                V_init_chunk = np.repeat(V_carry.reshape(-1, 1), cols, axis=1)
            q_init_chunk = None
            if q_carry is not None:
                q_init_chunk = np.repeat(q_carry.reshape(-1, 1), cols, axis=1)

            t_c0 = time.perf_counter()
            if network.has_pv:
                V_c, q_c, info_c = self._solve_chunk_with_pv(
                    network, s_chunk, K, L, Z_B, A_pv_inv,
                    V_init=V_init_chunk, q_init=q_init_chunk,
                )
                q_pv_all[:, s0:s1] = q_c
            else:
                V_c, info_c = self._solve_chunk_pq_only(
                    network, s_chunk, K, L, V_init=V_init_chunk
                )
            t_c = (time.perf_counter() - t_c0) * 1e3
            t_chunks_ms.append(t_c)

            V_all[:, s0:s1] = V_c
            converged_ps[s0:s1] = info_c["converged"]
            outer_iters_ps[s0:s1] = info_c["outer_iters"]
            inner_iters_ps[s0:s1] = info_c["inner_iters"]
            pv_v_err_ps[s0:s1] = info_c["pv_v_error"]

            if collect_scenario_stats:
                absV = np.abs(V_c)
                vmin_ps[s0:s1] = absV.min(axis=0)
                vmax_ps[s0:s1] = absV.max(axis=0)

            n_inner_calls += info_c["n_inner_calls"]
            n_gemm += info_c["n_gemm"]
            work_exec += info_c["n_gemm"] * cols
            flops += 8.0 * bphi * bphi * cols * info_c["n_gemm"]
            for ell, n_act in enumerate(info_c["active_per_outer"]):
                active_hist[ell] = active_hist.get(ell, 0) + int(n_act)

            # Zustand für den nächsten Chunk
            if warm_mode in ("carry_last", "carry_mean"):
                V_carry = (V_c[:, -1].copy() if warm_mode == "carry_last"
                           else V_c.mean(axis=1))
                if network.has_pv:
                    q_carry = (q_c[:, -1].copy() if warm_mode == "carry_last"
                               else q_c.mean(axis=1))

            if verbose:
                n_conv = int(np.sum(converged_ps[:s1]))
                print(f"  Chunk {c+1:>3}/{n_chunks} [{s0:>7}:{s1:>7}] "
                      f"conv {n_conv}/{s1} ({100*n_conv/s1:5.1f}%)  {t_c:8.1f} ms")

        t_solve_ms = (time.perf_counter() - t_solve0) * 1e3
        elapsed = time.perf_counter() - t_start

        n_conv_total = int(np.sum(converged_ps))
        I_s = network.Y_ss @ network.v_s.reshape(-1, 1) + network.Y_sd @ V_all
        s_slack = network.v_s.reshape(-1, 1) * np.conj(I_s)

        if n_conv_total < tau:
            div_idx = np.where(~converged_ps)[0]
            print(f"  [WARN] {tau - n_conv_total}/{tau} Szenarien nicht "
                  f"konvergiert. Erste 5: {div_idx[:5].tolist()}")

        self.pv_info = PVConvergenceInfo(
            outer_iterations=int(np.max(outer_iters_ps)) if tau else 0,
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
            v_min_per_scenario=vmin_ps,
            v_max_per_scenario=vmax_ps,
            t_precompute_ms=t_precompute_ms,
            t_solve_ms=t_solve_ms,
            t_chunks_ms=t_chunks_ms,
            chunk_size=chunk_size,
            n_chunks=n_chunks,
            n_inner_calls=n_inner_calls,
            n_gemm=n_gemm,
            work_executed_cols=int(work_exec),
            work_useful_cols=int(np.sum(inner_iters_ps)),
            flops_gemm=flops,
            active_per_outer=[active_hist[k] for k in sorted(active_hist)],
            warm_mode=warm_mode,
            cond_Xpp=cond_Xpp,
            rho_jacobi=rho_j,
            min_diag_off=min_ratio,
        )

        return PowerFlowResult(
            voltages=V_all,
            iterations=int(np.max(inner_iters_ps)) if tau else 0,
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

    def _solve_chunk_pq_only(self, network, s_chunk, K, L, V_init=None):
        bphi = network.n_bus_phases
        tau = s_chunk.shape[1]
        V, n_iter, conv, tol_val, _ = self._inner_fpi(
            K, L, np.conj(s_chunk), bphi, tau,
            V_init=V_init, collect_history=False
        )
        return V, {
            "converged": np.full(tau, conv, dtype=bool),
            "outer_iters": np.zeros(tau, dtype=np.int32),
            "inner_iters": np.full(tau, n_iter, dtype=np.int32),
            "pv_v_error": np.full(tau, tol_val, dtype=np.float64),
            "n_inner_calls": 1,
            "n_gemm": n_iter,
            "active_per_outer": [tau],
        }

    # ══════════════════════════════════════════════════════════════════
    #  Chunk-Solver: mit PV (Kern-Algorithmus mit per-Szenario-Masken)
    # ══════════════════════════════════════════════════════════════════

    def _solve_chunk_with_pv(self, network, s_chunk, K, L, Z_B, A_pv_inv,
                             V_init=None, q_init=None):
        bphi = network.n_bus_phases
        tau = s_chunk.shape[1]

        pv_idx = network.pv_indices
        n_pv = len(pv_idx)
        v_spec = network.pv_v_setpoint
        v_spec_2d = v_spec.reshape(-1, 1)         # (n_pv, 1)
        v_spec_sq_2d = (v_spec ** 2).reshape(-1, 1)

        s_work = s_chunk.copy()
        p_pv_fixed = s_work[pv_idx, :].real.copy()   # (n_pv, τ) time-varying P
        q_pv = (np.zeros((n_pv, tau)) if q_init is None
                else np.asarray(q_init, dtype=float).reshape(n_pv, tau).copy())
        s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        V = (np.ones((bphi, tau), dtype=np.complex128) if V_init is None
             else np.asarray(V_init, dtype=np.complex128).copy())

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
        n_inner_calls = n_gemm = 0
        active_per_outer: list[int] = []

        inner_limit = (self.max_iter_inner_per_outer
                       if self.max_iter_inner_per_outer is not None else self.max_iter)
        use_two_phase = (self.max_iter_inner_per_outer is not None
                         and self.max_iter_inner_per_outer < self.max_iter)

        for ell in range(self.max_iter_outer):
            if self.cold_start:
                V = np.ones((bphi, tau), dtype=np.complex128)

            active_per_outer.append(int(np.sum(~converged_mask)))

            v_mag_pv = np.abs(V[pv_idx, :])
            err_per_col = np.max(np.abs(v_mag_pv - v_spec_2d), axis=0)

            S_conj = np.conj(s_work)
            V, n_inner, _, _, _ = self._inner_fpi(
                K, L, S_conj, bphi, tau, V_init=V, collect_history=False,
                max_iter_override=(inner_limit if not use_two_phase
                                   else self.max_iter_inner_per_outer),
                outer_error=err_per_col if self.adaptive_inner else None,
            )
            n_inner_calls += 1
            n_gemm += n_inner

            active = ~converged_mask
            inner_iters[active] += n_inner

            v_mag_pv = np.abs(V[pv_idx, :])
            err_per_col = np.max(np.abs(v_mag_pv - v_spec_2d), axis=0)

            newly = (err_per_col < self.tol_pv) & (~converged_mask)
            outer_iters[newly] = ell + 1
            pv_v_error[newly] = err_per_col[newly]
            converged_mask |= newly

            if converged_mask.all():
                break

            delta_v_sq = v_spec_sq_2d - v_mag_pv ** 2
            if self.use_decoupled:
                X_pp_diag = np.diag(np.imag(Z_B[np.ix_(pv_idx, pv_idx)]))
                delta_q = delta_v_sq / (2.0 * X_pp_diag.reshape(-1, 1))
            else:
                delta_q = A_pv_inv @ delta_v_sq
            delta_q[:, converged_mask] = 0.0
            q_pv = q_pv - self.omega * delta_q

            if q_min_col is not None:
                q_pv = np.maximum(q_pv, q_min_col)
            if q_max_col is not None:
                q_pv = np.minimum(q_pv, q_max_col)

            s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        if use_two_phase:
            if self.cold_start:
                V = np.ones((bphi, tau), dtype=np.complex128)
            v_mag_pv = np.abs(V[pv_idx, :])
            err_per_col = np.max(np.abs(v_mag_pv - v_spec_2d), axis=0)
            V, n_inner, _, _, _ = self._inner_fpi(
                K, L, np.conj(s_work), bphi, tau, V_init=V,
                collect_history=False, max_iter_override=self.max_iter,
                outer_error=err_per_col if self.adaptive_inner else None,
            )
            n_inner_calls += 1
            n_gemm += n_inner
            inner_iters += n_inner
            v_mag_pv = np.abs(V[pv_idx, :])
            err_per_col = np.max(np.abs(v_mag_pv - v_spec_2d), axis=0)
            newly = (err_per_col < self.tol_pv) & (~converged_mask)
            outer_iters[newly] += 1
            pv_v_error[newly] = err_per_col[newly]
            converged_mask |= newly

        still_div = ~converged_mask
        outer_iters[still_div] = self.max_iter_outer
        pv_v_error[still_div] = err_per_col[still_div]

        return V, q_pv, {
            "converged": converged_mask,
            "outer_iters": outer_iters,
            "inner_iters": inner_iters,
            "pv_v_error": pv_v_error,
            "n_inner_calls": n_inner_calls,
            "n_gemm": n_gemm,
            "active_per_outer": active_per_outer,
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
        V, n_iter, converged, tol_val, inner_history = self._inner_fpi(
            K, L, np.conj(s_batch), bphi, tau, collect_history=True
        )

        elapsed = time.perf_counter() - t_start

        self.pv_info = PVConvergenceInfo(
            outer_iterations=1,
            inner_iterations_total=n_iter,
            inner_iterations_per_outer=[n_iter],
            pv_v_error_final=0.0,
            pv_q_final=np.array([]),
            pv_v_final=np.array([]),
            converged_inner=converged,
            converged_outer=converged,
            inner_v_change_all=inner_history,
            outer_start_indices=[0],
        )

        s_slack = self._compute_slack_power(network, V)

        return PowerFlowResult(
            voltages=V,
            iterations=n_iter,
            converged=converged,
            elapsed_time_s=elapsed,
            max_mismatch=tol_val,
            s_slack=s_slack,
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

        # Coupled Thévenin: precompute (2*X_pp)^{-1} once
        X_pp = np.imag(Z_B[np.ix_(pv_idx, pv_idx)])
        A_pv = 2.0 * X_pp + 1e-12 * np.eye(n_pv)
        A_pv_inv = np.linalg.inv(A_pv)

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
        inner_fpi_time_ms = []

        inner_limit = self.max_iter_inner_per_outer if self.max_iter_inner_per_outer is not None else self.max_iter
        use_two_phase = self.max_iter_inner_per_outer is not None and self.max_iter_inner_per_outer < self.max_iter

        phase1_converged = False

        for ell in range(self.max_iter_outer):
            outer_iter = ell + 1
            if self.cold_start:
                V = np.ones((bphi, tau), dtype=np.complex128)
            outer_start_indices.append(len(inner_v_change_all))

            v_spec_2d = v_spec.reshape(-1, 1)
            v_mag_pv = np.abs(V[pv_idx, :])
            pv_v_error = np.max(np.abs(v_mag_pv - v_spec_2d))

            S_conj = np.conj(s_work)
            t_inner_start = time.perf_counter()
            V, n_inner, converged_inner, tol_inner, inner_history = self._inner_fpi(
                K, L, S_conj, bphi, tau, V_init=V, collect_history=True,
                max_iter_override=inner_limit if not use_two_phase else self.max_iter_inner_per_outer,
                outer_error=pv_v_error if self.adaptive_inner else None
            )
            t_inner_end = time.perf_counter()
            inner_fpi_time_ms.append((t_inner_end - t_inner_start) * 1000)
            inner_iter_total += n_inner
            inner_iter_log.append(n_inner)
            inner_v_change_all.extend(inner_history)

            v_mag_pv = np.abs(V[pv_idx, :])
            pv_v_error = np.max(np.abs(v_mag_pv - v_spec_2d))
            pv_v_error_history.append(float(pv_v_error))
            v_change_history.append(float(tol_inner))

            if pv_v_error < self.tol_pv:
                phase1_converged = True
                if not use_two_phase:
                    converged_outer = True
                    break
                else:
                    break

            # Q-update: decoupled (diagonal) or coupled
            delta_v_sq = v_spec_2d ** 2 - v_mag_pv ** 2
            if self.use_decoupled:
                X_pp_diag = np.diag(np.imag(Z_B[np.ix_(pv_idx, pv_idx)]))
                delta_q = delta_v_sq / (2.0 * X_pp_diag.reshape(-1, 1))
            else:
                delta_q = A_pv_inv @ delta_v_sq
            q_pv = q_pv - self.omega * delta_q

            if self.enforce_q_lims:
                if q_min is not None: q_pv = np.maximum(q_pv, q_min)
                if q_max is not None: q_pv = np.minimum(q_pv, q_max)

            s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        if use_two_phase:
            outer_start_indices.append(len(inner_v_change_all))
            outer_iter += 1
            v_mag_pv = np.abs(V[pv_idx, :])
            v_spec_2d = v_spec.reshape(-1, 1)
            pv_v_error = np.max(np.abs(v_mag_pv - v_spec_2d))
            S_conj = np.conj(s_work)
            t_inner_start = time.perf_counter()
            V, n_inner, converged_inner, tol_inner, inner_history = self._inner_fpi(
                K, L, S_conj, bphi, tau, V_init=V, collect_history=True,
                max_iter_override=self.max_iter,
                outer_error=pv_v_error if self.adaptive_inner else None
            )
            t_inner_end = time.perf_counter()
            inner_fpi_time_ms.append((t_inner_end - t_inner_start) * 1000)
            inner_iter_total += n_inner
            inner_iter_log.append(n_inner)
            inner_v_change_all.extend(inner_history)

            v_mag_pv = np.abs(V[pv_idx, :])
            pv_v_error = np.max(np.abs(v_mag_pv - v_spec_2d))
            pv_v_error_history.append(float(pv_v_error))
            v_change_history.append(float(tol_inner))
            converged_outer = pv_v_error < self.tol_pv

        elapsed = time.perf_counter() - t_start
        total_inner_time = sum(inner_fpi_time_ms)
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
            inner_fpi_time_ms=inner_fpi_time_ms,
            total_inner_fpi_time_ms=total_inner_time,
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
                   V_init=None, collect_history=False, max_iter_override=None,
                   outer_error=None):
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.ones((bphi, tau), dtype=np.complex128)
        L_col = L.reshape(-1, 1)

        converged = False
        n_iter = 0
        tol_val = np.inf
        history = []

        max_iter = max_iter_override if max_iter_override is not None else self.max_iter

        if self.adaptive_inner and outer_error is not None:
            oe = float(np.max(np.asarray(outer_error)))
            adaptive_tol = max(self.tol, oe)
        else:
            adaptive_tol = self.tol

        for n in range(max_iter):
            LAMBDA = S_conj * (1.0 / np.conj(V))
            V_new = K @ LAMBDA + L_col
            tol_val = float(np.max(np.abs(np.abs(V_new) - np.abs(V))))
            n_iter = n + 1
            V = V_new
            if collect_history:
                history.append(tol_val)
            if tol_val < adaptive_tol:
                converged = True
                break
        return V, n_iter, converged, tol_val, history
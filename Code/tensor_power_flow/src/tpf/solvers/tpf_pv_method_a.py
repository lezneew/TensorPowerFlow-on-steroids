# tensor_power_flow/src/tpf/solvers/tpf_pv_method_a.py
"""
TPF mit PV-Knoten: Methode A (Äußere Q-Schleife)
==================================================

Algorithmus:
┌─────────────────────────────────────────────────────────────┐
│  VORBERECHNUNG (einmalig):                                  │
│    K = -Y_dd⁻¹                    (konstant!)               │
│    L = K @ Y_ds @ v_s             (konstant!)               │
│    X_kk = Im(Z_B[k,k])           (Thévenin-Reaktanz)       │
│                                                             │
│  ÄUSSERE SCHLEIFE (ℓ = 0, 1, ...):                         │
│    1. s_work[pv] = p_pv + j·q_pv^(ℓ)                       │
│    2. INNERE SCHLEIFE (n = 0, 1, ...):                      │
│         V_{n+1} = K @ (s_work* ⊙ V*⁻¹) + L                │
│       → bis Konvergenz oder max_iter_inner                  │
│    3. Prüfe: |V_pv| ≈ V_spec ?                             │
│    4. Q-Korrektur:                                          │
│         ΔQ_k = (|V_spec_k|² - |V_calc_k|²) / (2·X_kk)     │
│         q_pv_k^(ℓ+1) = q_pv_k^(ℓ) - ω·ΔQ_k               │
│    5. Optional: Q-Grenzen einhalten (Clamping)              │
│                                                             │
│  Schlüsselvorteil: K und L bleiben über ALLE Iterationen    │
│  konstant → Parallelisierbarkeit bleibt erhalten!           │
└─────────────────────────────────────────────────────────────┘

Referenzen:
- Exposé Methode A (Thévenin-basierte Q-Korrektur)
- Giraldo et al. (2022): FPI/SAM Konvergenzbeweis
- Costa et al. (1999): PV-Behandlung über Current Injection
"""

import numpy as np
from numpy.typing import NDArray
import time
from dataclasses import dataclass

from tpf.core.network import NetworkData
from tpf.core.results import PowerFlowResult
from tpf.solvers.base_solver import BaseSolver


# ══════════════════════════════════════════════════════════════════════
#  Ergebnisklasse für PV-Details
# ══════════════════════════════════════════════════════════════════════

@dataclass
@dataclass
class PVConvergenceInfo:
    """Detaillierte Konvergenz-Informationen für die PV-Schleife."""
    outer_iterations: int
    inner_iterations_total: int
    inner_iterations_per_outer: list[int]
    pv_v_error_final: float
    pv_q_final: NDArray
    pv_v_final: NDArray
    converged_inner: bool
    converged_outer: bool
    pv_v_error_history: list[float] = None      # |V|-Fehler pro Outer-Iteration
    v_change_history: list[float] = None        # Inner-FPI tol pro Outer-Iteration


# ══════════════════════════════════════════════════════════════════════
#  Solver: Methode A
# ══════════════════════════════════════════════════════════════════════

class TPFDensePVMethodA(BaseSolver):
    """
    Tensor Power Flow mit PV-Knoten-Unterstützung.
    Methode A: Äußere Q-Schleife mit Thévenin-Korrektur.

    Parameters
    ----------
    tol : float
        Konvergenztoleranz für die innere FPI-Schleife (Spannungsänderung).
    max_iter_inner : int
        Maximale Iterationen der inneren FPI-Schleife.
    max_iter_outer : int
        Maximale Iterationen der äußeren Q-Schleife.
    tol_pv : float
        Konvergenztoleranz für |V| an PV-Knoten.
    omega : float
        Relaxationsfaktor für Q-Update (0 < ω ≤ 1).
        ω = 1.0: Volle Korrektur (Standard, gut für Verteilnetze)
        ω < 1.0: Unterdämpft (nötig bei η nahe 1 oder vermaschten Netzen)
    enforce_q_lims : bool
        Wenn True: Q-Grenzen aus NetworkData einhalten (Clamping).
        PV-Knoten, die an Q-Grenzen laufen, werden zu PQ-Knoten.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter_inner: int = 50,
        max_iter_outer: int = 30,
        tol_pv: float = 1e-5,
        omega: float = 1.0,
        enforce_q_lims: bool = False,
    ):
        super().__init__(tol, max_iter_inner)
        self.max_iter_outer = max_iter_outer
        self.tol_pv = tol_pv
        self.omega = omega
        self.enforce_q_lims = enforce_q_lims

        # Diagnostik (nach solve() verfügbar)
        self.pv_info: PVConvergenceInfo | None = None

    # ══════════════════════════════════════════════════════════════════
    #  Öffentliche API
    # ══════════════════════════════════════════════════════════════════

    def solve(self, network: NetworkData) -> PowerFlowResult:
        """Löst einen einzelnen Lastfluss mit PV-Knoten (τ=1)."""
        return self.solve_batch(network, network.s_nom.reshape(-1, 1))

    def solve_batch(
        self, network: NetworkData, s_batch: NDArray
    ) -> PowerFlowResult:
        """
        Löst τ Lastflüsse mit PV-Knoten-Unterstützung.

        Parameters
        ----------
        network : NetworkData
            Muss mit include_pv=True gebaut worden sein.
        s_batch : NDArray, shape (bφ, τ)
            Leistungsmatrix. Die Q-Werte an PV-Positionen werden
            vom Solver überschrieben und iterativ bestimmt.

        Returns
        -------
        PowerFlowResult
        """
        t_start = time.perf_counter()

        # ── Validierung ──
        if not network.has_pv:
            # Kein PV-Knoten → Fallback auf Standard-TPF
            return self._solve_pq_only(network, s_batch, t_start)

        # ── Solve mit PV-Behandlung ──
        result = self._solve_with_pv(network, s_batch, t_start)
        return result

    # ══════════════════════════════════════════════════════════════════
    #  Fallback: Reiner PQ-Fall (kein PV)
    # ══════════════════════════════════════════════════════════════════

    def _solve_pq_only(
        self, network: NetworkData, s_batch: NDArray, t_start: float
    ) -> PowerFlowResult:
        """Standard-TPF ohne PV-Knoten."""
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        K, L = self._precompute(network)
        V, n_iter, converged, tol_val = self._inner_fpi(
            K, L, np.conj(s_batch), bphi, tau
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

    # ══════════════════════════════════════════════════════════════════
    #  Hauptalgorithmus: Äußere Q-Schleife
    # ══════════════════════════════════════════════════════════════════

    def _solve_with_pv(
        self, network: NetworkData, s_batch: NDArray, t_start: float
    ) -> PowerFlowResult:
        """
        Kern-Algorithmus: Äußere Q-Schleife + innere FPI.
        """
        bphi = network.n_bus_phases
        tau = s_batch.shape[1]

        # ── Vorberechnung (EINMALIG) ──
        K, L = self._precompute(network)
        Z_B = -K  # Z_B = Y_dd^{-1}

        # PV-Knoten-Informationen
        pv_idx = network.pv_indices           # Lokale Indizes im d-Block
        n_pv = len(pv_idx)
        v_spec = network.pv_v_setpoint        # (n_pv,) Soll-|V|

        # Thévenin-Reaktanz an PV-Knoten: X_kk = Im(Z_B[k,k])
        X_th = np.imag(Z_B[pv_idx, pv_idx])  # (n_pv,)

        # Schutz vor X_th ≈ 0 (rein resistive Verbindung)
        X_th_safe = np.where(np.abs(X_th) > 1e-10, X_th, 1e-10)

        # ── Arbeitskopie von s_batch ──
        s_work = s_batch.copy()  # (bφ, τ)

        # Fixe P-Werte an PV-Knoten merken
        p_pv_fixed = s_work[pv_idx, :].real.copy()  # (n_pv, τ)

        # Q an PV-Knoten initialisieren: Start mit Q = 0
        # (Bedeutung: Netto-Blindleistung am Bus = 0)
        q_pv = np.zeros((n_pv, tau))  # (n_pv, τ)

        # s_work an PV-Knoten setzen
        s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        # ── Initialisierung ──
        V = np.ones((bphi, tau), dtype=np.complex128)  # Flat start
        converged_outer = False
        converged_inner = False
        outer_iter = 0
        inner_iter_total = 0
        inner_iter_log = []
        pv_v_error = np.inf

        # Q-Grenzen (falls vorhanden)
        q_min = None
        q_max = None
        if self.enforce_q_lims and network.pv_q_min is not None:
            q_min = network.pv_q_min.reshape(-1, 1) * np.ones((1, tau))
        if self.enforce_q_lims and network.pv_q_max is not None:
            q_max = network.pv_q_max.reshape(-1, 1) * np.ones((1, tau))

        # ══════════════════════════════════════════════════════════════
        #  ÄUSSERE SCHLEIFE
        # ══════════════════════════════════════════════════════════════
        pv_v_error_history = []
        v_change_history = []

        for ell in range(self.max_iter_outer):
            outer_iter = ell + 1

            # ── Schritt 1: Innere FPI mit aktuellem s_work ──
            S_conj = np.conj(s_work)  # (bφ, τ)

            V, n_inner, converged_inner, tol_inner = self._inner_fpi(
                K, L, S_conj, bphi, tau, V_init=V
            )

            inner_iter_total += n_inner
            inner_iter_log.append(n_inner)

            if not converged_inner:
                # Innere Schleife divergiert → Abbruch
                break

            # ── Schritt 2: PV-Spannungsfehler prüfen ──
            v_mag_pv = np.abs(V[pv_idx, :])  # (n_pv, τ)
            v_spec_2d = v_spec.reshape(-1, 1)  # (n_pv, 1) → broadcast

            # Fehler: max über alle PV-Knoten und Szenarien
            pv_v_error = np.max(np.abs(v_mag_pv - v_spec_2d))

            if pv_v_error < self.tol_pv:
                converged_outer = True
                break

            # ── Schritt 3: Q-Korrektur (Thévenin) ──
            # ΔQ = (|V_spec|² - |V_calc|²) / (2·X_kk)
            delta_q = (
                (v_spec_2d ** 2 - v_mag_pv ** 2)
                / (2.0 * X_th_safe.reshape(-1, 1))
            )  # (n_pv, τ)

            # Q-Update mit Relaxation
            # Convention: s_nom ist Verbraucherkonvention (positiv = Verbrauch)
            # ΔQ > 0 → brauchen mehr Einspeisung → weniger Verbrauch → q sinkt
            q_pv = q_pv - self.omega * delta_q

            # ── Schritt 4: Q-Grenzen (optional) ──
            if self.enforce_q_lims:
                if q_min is not None:
                    q_pv = np.maximum(q_pv, q_min)
                if q_max is not None:
                    q_pv = np.minimum(q_pv, q_max)

            pv_v_error_history.append(float(pv_v_error))
            v_change_history.append(float(tol_inner))

            # ── Schritt 5: s_work aktualisieren ──
            s_work[pv_idx, :] = p_pv_fixed + 1j * q_pv

        # ══════════════════════════════════════════════════════════════
        #  Ergebnis zusammenbauen
        # ══════════════════════════════════════════════════════════════

        elapsed = time.perf_counter() - t_start

        # Diagnostik speichern
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
        )

        return PowerFlowResult(
            voltages=V,
            iterations=inner_iter_total,
            converged=converged_inner and converged_outer,
            elapsed_time_s=elapsed,
            max_mismatch=pv_v_error,
            pv_indices=pv_idx,
            pv_q_pu=q_pv[:, 0] if tau == 1 else q_pv,
            pv_v_setpoint_pu=v_spec,
        )

    # ══════════════════════════════════════════════════════════════════
    #  Vorberechnung
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _precompute(network: NetworkData):
        """
        Berechnet K und L (EINMALIG, konstant über alle Iterationen).

            K = -Y_dd^{-1}
            L = K @ Y_ds @ v_s
        """
        Z_B = np.linalg.inv(network.Y_dd)
        K = -Z_B
        L = K @ network.Y_ds @ network.v_s  # (bφ,)
        return K, L

    # ══════════════════════════════════════════════════════════════════
    #  Innere FPI-Schleife
    # ══════════════════════════════════════════════════════════════════

    def _inner_fpi(
        self,
        K: NDArray,
        L: NDArray,
        S_conj: NDArray,
        bphi: int,
        tau: int,
        V_init: NDArray | None = None,
    ) -> tuple[NDArray, int, bool, float]:
        """
        Standard Fixed-Point Iteration (innere Schleife).

        V_{n+1} = K @ (S* ⊙ V*^{-1}) + L

        Parameters
        ----------
        K : (bφ, bφ) Impedanzmatrix (-Z_B)
        L : (bφ,) Offset-Vektor
        S_conj : (bφ, τ) konjugierte Leistungen
        bphi : Dimension
        tau : Anzahl Szenarien
        V_init : (bφ, τ) Startwert (Warm-Start aus vorheriger Outer-Iteration)

        Returns
        -------
        V : (bφ, τ) Spannungen
        n_iter : Anzahl Iterationen
        converged : bool
        tol_val : finaler Fehler
        """
        # Startwert
        if V_init is not None:
            V = V_init.copy()
        else:
            V = np.ones((bphi, tau), dtype=np.complex128)

        L_col = L.reshape(-1, 1)  # (bφ, 1) für Broadcasting

        converged = False
        n_iter = 0
        tol_val = np.inf

        for n in range(self.max_iter):
            # Λ = S* / V* = S* · (1/V*)
            LAMBDA = S_conj * (1.0 / np.conj(V))  # (bφ, τ)

            # V_{n+1} = K @ Λ + L
            V_new = K @ LAMBDA + L_col  # (bφ, τ)

            # Konvergenz: max Spannungsänderung
            tol_val = np.max(np.abs(np.abs(V_new) - np.abs(V)))

            n_iter = n + 1
            V = V_new

            if tol_val < self.tol:
                converged = True
                break

        return V, n_iter, converged, tol_val
# tensor_power_flow/src/tpf/analysis/rx_diagnostics.py
"""
Kontraktions- und Sensitivitaetsdiagnostik fuer den R/X-Sweep.
Alle Groessen werden in der Vorzeichenkonvention des TPF-Solvers
berechnet (K = -Y_dd^{-1}, V = K (S*/V*) + L).
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


# ══════════════════════════════════════════════════════════════════
#  Innere Schleife
# ══════════════════════════════════════════════════════════════════

def fpi_reference(
    Y_dd: NDArray, Y_ds: NDArray, v_s: NDArray, s_d: NDArray,
    tol: float = 1e-12, max_iter: int = 500,
) -> dict:
    """
    Instrumentierte FPI (PQ-only, τ=1). Liefert die Fehlerfolge
    e_n = ||v^n - v*||_inf gegenueber der eigenen Endloesung.
    """
    Z_B = np.linalg.inv(Y_dd)
    K = -Z_B
    L = (K @ Y_ds @ v_s.reshape(-1, 1)).ravel()

    S_conj = np.conj(s_d).ravel()
    V = np.ones(Y_dd.shape[0], dtype=np.complex128)

    traj, steps = [V.copy()], []
    for _ in range(max_iter):
        V_new = K @ (S_conj / np.conj(V)) + L
        steps.append(float(np.max(np.abs(np.abs(V_new) - np.abs(V)))))
        V = V_new
        traj.append(V.copy())
        if steps[-1] < tol:
            break

    V_star = traj[-1]
    err = np.array([np.max(np.abs(v - V_star)) for v in traj[:-1]])
    return {"V": V_star, "err": err, "steps": np.array(steps),
            "n_iter": len(steps), "converged": steps[-1] < tol,
            "v_min": float(np.min(np.abs(V_star)))}


def eta_empirical(err, floor_factor: float = 1e3,
                  top_factor: float = 0.1, min_pts: int = 3) -> float:
    """
    Geometrische Rate (e_b/e_a)^{1/(b-a)} auf dem Fenster
        1e3 * min(e) < e_n < 0.1 * e_0
    Liefert NaN, wenn zu wenige Punkte -> erkennbar statt irrefuehrend.
    """
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if e.size < min_pts + 1:
        return np.nan
    lo, hi = floor_factor * e.min(), top_factor * e[0]
    idx = np.where((e > lo) & (e < hi))[0]
    if idx.size < min_pts:
        return np.nan
    a, b = int(idx[0]), int(idx[-1])
    return float((e[b] / e[a]) ** (1.0 / (b - a)))


def eta_theoretical(Y_dd: NDArray, s_d: NDArray, v_min: float) -> dict:
    """
    eta = || Z_B diag(s*) || / v_min^2 in verschiedenen Normen
    plus knotenweise Thevenin-Naeherung max |Z_kk| |s_k| / v_min^2.
    """
    Z_B = np.linalg.inv(Y_dd)
    M = (Z_B * np.conj(s_d).reshape(1, -1)) / v_min**2
    z_th = np.abs(np.diag(Z_B))
    z_load = v_min**2 / np.maximum(np.abs(s_d).ravel(), 1e-30)
    return {
        "eta_inf": float(np.linalg.norm(M, np.inf)),
        "eta_1": float(np.linalg.norm(M, 1)),
        "eta_2": float(np.linalg.norm(M, 2)),
        "eta_thev": float(np.max(z_th / z_load)),
    }


# ══════════════════════════════════════════════════════════════════
#  PV-PV-Block
# ══════════════════════════════════════════════════════════════════

def xpp_metrics(Y_dd: NDArray, pv_idx: NDArray) -> dict:
    """Struktur- und Konditionsmasse des PV-PV-Reaktanzblocks."""
    if pv_idx is None or len(pv_idx) == 0:
        return {}
    Z_B = np.linalg.inv(Y_dd)
    Z_pp = Z_B[np.ix_(pv_idx, pv_idx)]
    X_pp = np.imag(Z_pp)
    R_pp = np.real(Z_pp)
    n = len(pv_idx)

    d = np.abs(np.diag(X_pp))
    off = np.sum(np.abs(X_pp), axis=1) - d
    out = {
        "n_pv": n,
        "x_kk_mean": float(np.mean(np.diag(X_pp))),
        "x_kk_min": float(np.min(np.abs(np.diag(X_pp)))),
        "rx_diag_mean": float(np.mean(np.abs(np.diag(R_pp) / np.diag(X_pp)))),
        "diag_dom_min": float(np.min(d / np.maximum(off, 1e-30))),
        "offdiag_ratio": float(np.sum(off) / max(np.sum(d), 1e-30)),
        "cond_xpp": float(np.linalg.cond(X_pp)) if n > 0 else np.nan,
    }
    if n > 1:
        J = np.eye(n) - np.diag(1.0 / np.diag(X_pp)) @ X_pp
        out["rho_jacobi"] = float(np.max(np.abs(np.linalg.eigvals(J))))
    else:
        out["rho_jacobi"] = 0.0
    return out


def exact_q_sensitivity(Z_B: NDArray, v: NDArray, pv_idx: NDArray) -> NDArray:
    """
    Exakte linearisierte Sensitivitaet d|V_pv|^2 / dQ ohne die Naeherungen
    gleicher Betraege und kleiner Winkeldifferenzen:

        J = 2 Re( diag(v_pv*) Z_pp j diag(1/v_pv*) )

    Fuer |v_k|=|v_j| und theta_k=theta_j reduziert sich J auf 2 X_pp.
    """
    v_pv = v[pv_idx]
    Z_pp = Z_B[np.ix_(pv_idx, pv_idx)]
    D1 = np.diag(np.conj(v_pv))
    D2 = np.diag(1.0 / np.conj(v_pv))
    return 2.0 * np.real(D1 @ (1j * Z_pp) @ D2)


# ══════════════════════════════════════════════════════════════════
#  Instrumentierte Methode A (aeussere Schleife)
# ══════════════════════════════════════════════════════════════════

def method_a_instrumented(
    Y_dd: NDArray, Y_ds: NDArray, v_s: NDArray, s_nom: NDArray,
    pv_idx: NDArray, v_spec: NDArray,
    tol_inner: float = 1e-8, tol_pv: float = 1e-6,
    max_outer: int = 60, max_inner: int = 200,
    omega: float = 1.0, decoupled: bool = False,
    exact_sensitivity: bool = False,
) -> dict:
    """
    Aeussere Q-Schleife mit Warm Start, identische Update-Regel wie
    TPFDensePVMethodA, zusaetzlich pro aeusserem Schritt:

        residual   = |V_spec|^2 - |v_pv|^2                     (vorhergesagt/omega)
        measured   = |v_pv_neu|^2 - |v_pv_alt|^2
        sens_error = ||measured - omega*residual|| / ||omega*residual||

    Das ist der relative Modellfehler der linearisierten Sensitivitaet.
    """
    Z_B = np.linalg.inv(Y_dd)
    K = -Z_B
    L = (K @ Y_ds @ v_s.reshape(-1, 1)).ravel()

    X_pp = np.imag(Z_B[np.ix_(pv_idx, pv_idx)])
    n_pv = len(pv_idx)
    A_static = 2.0 * X_pp + 1e-12 * np.eye(n_pv)

    s = s_nom.astype(np.complex128).ravel().copy()
    p_pv = s[pv_idx].real.copy()
    q_pv = np.zeros(n_pv)
    s[pv_idx] = p_pv + 1j * q_pv

    V = np.ones(Y_dd.shape[0], dtype=np.complex128)
    hist = {"v_err": [], "sens_error": [], "inner": [], "dq_norm": [],
            "q_max": []}
    inner_total, converged = 0, False

    for _ in range(max_outer):
        v_pv_old = V[pv_idx].copy()

        # --- innere FPI (Warm Start) ---
        S_conj = np.conj(s)
        n_in = 0
        for _ in range(max_inner):
            V_new = K @ (S_conj / np.conj(V)) + L
            d = np.max(np.abs(np.abs(V_new) - np.abs(V)))
            V = V_new
            n_in += 1
            if d < tol_inner:
                break
        inner_total += n_in
        hist["inner"].append(n_in)

        v_pv = V[pv_idx]
        residual = v_spec**2 - np.abs(v_pv) ** 2
        v_err = float(np.max(np.abs(np.abs(v_pv) - v_spec)))
        hist["v_err"].append(v_err)

        # Modellfehler der Sensitivitaet des VORHERGEHENDEN Schrittes
        if len(hist["dq_norm"]) > 0:
            measured = np.abs(v_pv) ** 2 - np.abs(v_pv_old) ** 2
            pred = omega * hist["_res_prev"]
            denom = max(np.linalg.norm(pred, np.inf), 1e-30)
            hist["sens_error"].append(
                float(np.linalg.norm(measured - pred, np.inf) / denom))
        hist["_res_prev"] = residual.copy()

        if v_err < tol_pv:
            converged = True
            break

        # --- Q-Korrektur ---
        if decoupled:
            dq = residual / (2.0 * np.diag(X_pp))
        elif exact_sensitivity:
            J = exact_q_sensitivity(Z_B, V, pv_idx)
            dq = np.linalg.solve(J + 1e-12 * np.eye(n_pv), residual)
        else:
            dq = np.linalg.solve(A_static, residual)

        q_pv = q_pv - omega * dq
        s[pv_idx] = p_pv + 1j * q_pv
        hist["dq_norm"].append(float(np.linalg.norm(dq, np.inf)))
        hist["q_max"].append(float(np.max(np.abs(q_pv))))

    hist.pop("_res_prev", None)
    return {"V": V, "q_pv": q_pv, "converged": converged,
            "outer": len(hist["v_err"]), "inner_total": inner_total,
            "v_err_final": hist["v_err"][-1] if hist["v_err"] else np.inf,
            "hist": hist}
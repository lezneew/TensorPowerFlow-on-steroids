import numpy as np
from dataclasses import dataclass
from tpf.core.results import PowerFlowResult


@dataclass
class ValidationResult:
    """Ergebnis eines Vergleichs TPF vs. Referenz."""
    max_voltage_error_pu: float
    mean_voltage_error_pu: float
    max_angle_error_deg: float
    iteration_ratio: float  # iter_tpf / iter_nr
    speedup: float  # t_nr / t_tpf
    both_converged: bool


def compare_results(
        tpf_result: PowerFlowResult,
        ref_result: PowerFlowResult,
        pq_indices: np.ndarray | None = None,
) -> ValidationResult:
    """Vergleicht TPF-Ergebnis mit Referenz (pandapower NR)."""

    # Nur PQ-Knoten vergleichen (TPF liefert keine Slack-Spannung)
    v_tpf = tpf_result.voltages.flatten()
    v_ref = ref_result.voltages[pq_indices] if pq_indices is not None \
        else ref_result.voltages

    # Spannungsbetragsfehler
    mag_error = np.abs(np.abs(v_tpf) - np.abs(v_ref))

    # Winkelfehler
    angle_error = np.abs(
        np.angle(v_tpf, deg=True) - np.angle(v_ref, deg=True)
    )

    return ValidationResult(
        max_voltage_error_pu=float(np.max(mag_error)),
        mean_voltage_error_pu=float(np.mean(mag_error)),
        max_angle_error_deg=float(np.max(angle_error)),
        iteration_ratio=tpf_result.iterations / max(ref_result.iterations, 1),
        speedup=ref_result.elapsed_time_s / max(tpf_result.elapsed_time_s, 1e-12),
        both_converged=tpf_result.converged and ref_result.converged,
    )
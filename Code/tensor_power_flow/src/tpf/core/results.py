from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class PowerFlowResult:
    """Ergebnisse einer Lastflussberechnung."""

    voltages: NDArray[np.complex128]  # (n_total_buses,) oder (bφ, τ)
    iterations: int
    converged: bool
    elapsed_time_s: float
    max_mismatch: float

    s_slack: NDArray[np.complex128] | None = None

    pv_indices: NDArray[np.int64] | None = None       # Indizes der PV-Knoten im PPC
    pv_q_pu: NDArray[np.float64] | None = None        # Blindleistung an PV-Knoten (p.u.)
    pv_v_setpoint_pu: NDArray[np.float64] | None = None  # Sollspannung der PV-Knoten

    @property
    def v_mag(self) -> NDArray[np.float64]:
        return np.abs(self.voltages)

    @property
    def v_angle_deg(self) -> NDArray[np.float64]:
        return np.angle(self.voltages, deg=True)
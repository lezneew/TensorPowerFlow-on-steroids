from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class PowerFlowResult:
    """Ergebnisse einer Lastflussberechnung."""

    voltages: NDArray[np.complex128]  # (b·φ,) oder (b·φ × τ)
    iterations: int
    converged: bool
    elapsed_time_s: float
    max_mismatch: float

    # Optional (nach Berechnung)
    s_slack: NDArray[np.complex128] | None = None

    @property
    def v_mag(self) -> NDArray[np.float64]:
        return np.abs(self.voltages)

    @property
    def v_angle_deg(self) -> NDArray[np.float64]:
        return np.angle(self.voltages, deg=True)
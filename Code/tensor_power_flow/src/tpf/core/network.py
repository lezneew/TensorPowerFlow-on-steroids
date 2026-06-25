from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NetworkData:
    """Internes Netzwerkmodell für den TPF (rein PQ + Slack)."""

    # Admittanzmatrix-Blöcke
    Y_dd: NDArray[np.complex128]  # (b·φ × b·φ) - Lastknoten untereinander
    Y_ds: NDArray[np.complex128]  # (b·φ × φ)   - Kopplung Last↔Slack

    # Slack-Spannung
    v_s: NDArray[np.complex128]  # (φ,) - z.B. [1.0+0j] für einphasig

    # Nominale Knotenleistungen (Verbraucherkonvention: positiv = Last)
    s_nom: NDArray[np.complex128]  # (b·φ,)

    # ZIP-Koeffizienten
    alpha_p: NDArray[np.float64]  # (b·φ,)
    alpha_i: NDArray[np.float64]  # (b·φ,)
    alpha_z: NDArray[np.float64]  # (b·φ,)

    # Metadaten
    n_buses: int  # Anzahl Lastknoten (b)
    n_phases: int  # Anzahl Phasen (φ)
    bus_names: list[str] | None = None

    @property
    def n_bus_phases(self) -> int:
        return self.n_buses * self.n_phases
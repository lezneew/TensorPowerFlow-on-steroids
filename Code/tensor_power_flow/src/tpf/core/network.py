from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NetworkData:
    """Internes Netzwerkmodell für den TPF (PQ + Slack + PV)."""

    # Admittanzmatrix-Blöcke (alle Nicht-Slack-Knoten: PQ ∪ PV)
    Y_dd: NDArray[np.complex128]  # (b·φ × b·φ)
    Y_ds: NDArray[np.complex128]  # (b·φ × φ)

    # Slack-Spannung
    v_s: NDArray[np.complex128]  # (φ,)

    # Nominale Knotenleistungen (Verbraucherkonvention: positiv = Last)
    s_nom: NDArray[np.complex128]  # (b·φ,)

    # ZIP-Koeffizienten
    alpha_p: NDArray[np.float64]  # (b·φ,)
    alpha_i: NDArray[np.float64]  # (b·φ,)
    alpha_z: NDArray[np.float64]  # (b·φ,)

    # Metadaten
    n_buses: int
    n_phases: int
    bus_names: list[str] | None = None

    # === NEU: PV-Knoten ===
    pv_mask: NDArray[np.bool_] | None = None        # (b·φ,) True an PV-Positionen
    pv_v_setpoint: NDArray[np.float64] | None = None  # (n_pv,) Soll-|V|
    pv_p_setpoint: NDArray[np.float64] | None = None  # (n_pv,) Soll-P in p.u.

    @property
    def n_bus_phases(self) -> int:
        return self.n_buses * self.n_phases

    @property
    def n_pv(self) -> int:
        if self.pv_mask is None:
            return 0
        return int(np.sum(self.pv_mask))

    @property
    def pv_indices(self) -> NDArray[np.int64] | None:
        """Lokale Indizes der PV-Knoten innerhalb des d-Blocks."""
        if self.pv_mask is None:
            return None
        return np.where(self.pv_mask)[0]

    @property
    def pq_indices(self) -> NDArray[np.int64]:
        """Lokale Indizes der reinen PQ-Knoten."""
        if self.pv_mask is None:
            return np.arange(self.n_bus_phases)
        return np.where(~self.pv_mask)[0]
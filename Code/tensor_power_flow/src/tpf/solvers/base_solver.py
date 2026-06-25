from abc import ABC, abstractmethod
from tpf.core.network import NetworkData
from tpf.core.results import PowerFlowResult
from numpy.typing import NDArray

class BaseSolver(ABC):
    """Abstrakte Basisklasse für alle Power-Flow-Solver."""

    def __init__(self, tol: float = 1e-6, max_iter: int = 100):
        self.tol = tol
        self.max_iter = max_iter

    @abstractmethod
    def solve(self, network: NetworkData) -> PowerFlowResult:
        """Löst den Lastfluss für ein gegebenes Netzwerk."""
        ...

    @abstractmethod
    def solve_batch(
            self, network: NetworkData, s_batch: NDArray
    ) -> PowerFlowResult:
        """Löst τ Lastflüsse gleichzeitig (Tensor-Version)."""
        ...
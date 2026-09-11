import hybrid
import dimod
from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePolicy(ABC):
    """Classe base astratta per tutte le policy di risoluzione."""
    
    def __init__(self, name: str, subproblem_size: int = 40, max_iter: int = 3, convergence: int = 3):
        self.name = name
        self.subproblem_size = subproblem_size
        self.max_iter = max_iter
        self.convergence = convergence

    @abstractmethod
    def solve(
        self, 
        bqm: dimod.BinaryQuadraticModel, 
        global_embedding: Dict, 
        is_all_embeddable: bool, 
        target_graph: Any = None
    ) -> Dict[str, Any]:
        """Metodo astratto da implementare in ogni policy specifica."""
        pass

    @staticmethod
    def _merge_substates(_, substates, **kwargs):
        a, b = substates
        return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))
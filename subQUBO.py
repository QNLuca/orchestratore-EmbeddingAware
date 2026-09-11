from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional
import dimod
import hybrid

from dwave.embedding.zephyr import find_clique_embedding
from minorminer import find_embedding

@dataclass
class SubQUBO:
    """
    Rappresentazione di un sotto-problema quadratico per estrazione rapida features.
    """
    bqm: dimod.BinaryQuadraticModel
    embedding: Optional[Dict] = None

    @property
    def variables(self) -> List[Union[int, str]]:
        return list(self.bqm.variables)

    @property
    def num_variables(self) -> int:
        return self.bqm.num_variables

    @property
    def density(self) -> float:
        num_vars = len(self.bqm.variables)
        if num_vars <= 1:
            return 0.0
        possible_edges = num_vars * (num_vars - 1) / 2
        return len(self.bqm.quadratic) / possible_edges

    @property
    def max_degree(self) -> int:
        if not self.bqm.variables:
            return 0
        degrees = {v: 0 for v in self.bqm.variables}
        for u, v in self.bqm.quadratic.keys():
            degrees[u] += 1
            degrees[v] += 1
        return max(degrees.values()) if degrees else 0

    @property
    def average_degree(self) -> int:
        num_vars = len(self.bqm.variables)
        if num_vars == 0:
            return 0
        #in un grafo la somma dei gradi è pari a 2 * numero_di_archi (interazioni quadratiche)
        total_degree = 2 * len(self.bqm.quadratic)
        
        #arrotondo al valore intero più vicino
        return round(total_degree / num_vars)

    @classmethod
    def compute_embedding(self, bqm: dimod.BinaryQuadraticModel, target_graph) -> Tuple[Dict, bool]:
            num_vars = self.num_variables
            if num_vars == 0:
                return {}, False
    
            try:
                embedding = find_clique_embedding(
                    k=num_vars,
                    target_graph=target_graph
                )
            except Exception:
                try:
                    edges = list(bqm.quadratic.keys()) or [(v, v) for v in bqm.variables]
                    embedding = find_embedding(
                        S=edges,
                        T=target_graph.edges,
                        timeout=100,
                        threads=4,
                        verbose=0
                    )
                except Exception:
                    embedding = {}
    
            is_all_embeddable = (len(embedding) == num_vars) and (num_vars > 0)
            self.embedding = embedding

            return embedding, is_all_embeddable
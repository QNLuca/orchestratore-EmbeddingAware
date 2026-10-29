from dataclasses import dataclass
import random
import time
from typing import Dict, List, Union, Optional
import dimod

#from dwave.embedding import is_valid_embedding
from dwave.embedding.zephyr import find_clique_embedding
from minorminer import find_embedding
import numpy as np

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
    def min_degree(self) -> int:
        if not self.bqm.variables:
            return 0
        degrees = {v: 0 for v in self.bqm.variables}
        for u, v in self.bqm.quadratic.keys():
            degrees[u] += 1
            degrees[v] += 1
        return min(degrees.values()) if degrees else 0

    @property
    def average_degree(self) -> float:
        num_vars = len(self.bqm.variables)
        if num_vars == 0:
            return 0
        #in un grafo la somma dei gradi è pari a 2 * numero_di_archi (interazioni quadratiche)
        total_degree = 2 * len(self.bqm.quadratic)
        
        return total_degree / num_vars

    @staticmethod
    def compute_embedding(bqm, target_graph) -> tuple[dict, float]:
        """
        Calcola l'embedding per un sub-BQM e ne misura il tempo esatto di calcolo.
        Ritorna una tupla: (embedding,is_all_embeddable, sub_embedding_time)
        """
        num_vars = bqm.num_variables
        embedding = {}

        edges = list(bqm.quadratic.keys()) + [(v, v) for v in bqm.variables]

        t_embed_0 = time.perf_counter()

        if num_vars < 184:  #clique completa maggiore su grafo Zephyr(12)
            try:
                raw_embedding = find_clique_embedding(
                    k=num_vars,
                    target_graph=target_graph,
                    seed=random.randint(1, 10)
                )

                var_list = list(bqm.variables)
    
                #remap necessario per embed_bqm:
                #prende la catena associata all'indice numerico i dal clique embedding e la riassocia alla i-esima variabile effettiva del BQM (var_list[i])
                embedding = {
                    var_list[i]: list(raw_embedding[i])
                    for i in range(num_vars)
                    if i in raw_embedding
                }

            except Exception:
                try:
                    embedding = find_embedding(
                        S=edges,
                        T=target_graph,
                        timeout=60,
                        random_seed=random.randint(1, 10),
                        threads=8,
                        verbose=0
                    )
                except Exception:
                    embedding = {}
        else:
            try:
                embedding = find_embedding(
                    S=edges,
                    T=target_graph,
                    timeout=60,
                    random_seed=random.randint(1, 10),
                    threads=8,
                    verbose=0
                )
            except Exception:
                embedding = {}

        is_all_embeddable = (len(embedding) == num_vars) and (num_vars > 0) 

        #più rigoroso ma più lento
        #is_all_embeddable = is_valid_embedding(
        #        emb=embedding,
        #        source=edges, 
        #        target=target_graph
        #)

        t_sub_embed = time.perf_counter() - t_embed_0

        return embedding, is_all_embeddable, t_sub_embed

    @staticmethod
    def compute_advanced_embedding_metrics(embedding, sub_bqm, chain_break_fraction=None):
        """
        Calcola metriche avanzate e l'Embedding Quality Index (EQI).
        Funziona sia per QPU (con chain_break_fraction reale) che per CPU (senza).
        """
        chain_lengths = [len(c) for c in embedding.values()]

        if not embedding or len(embedding) == 0 or len(chain_lengths) == 0:
            return {
                "has_embedding": False,
                "eqi": 0.0,
                "max_chain": 0,
                "mean_chain": 0.0,
                "median_chain": 0.0,
                "physical_nodes": 0,
                "expansion": 0.0,
                "long_chain_fraction": 0.0,
                "good_embedding": False
            }
        
        physical_nodes = sum(chain_lengths)
        num_vars = sub_bqm.num_variables

        #Expansion Factor: rapporto medio tra qubit fisici allocati e variabili logiche
        expansion = physical_nodes / num_vars if num_vars > 0 else 0.0

        lThr = 7.0

        #Frazione di catene critiche (> 7)
        long_chains_count = sum(1 for c in chain_lengths if c > lThr)
        long_chain_fraction = long_chains_count / len(chain_lengths) if len(chain_lengths) > 0 else 0.0

        ##Calcolo dell'Embedding Quality Index (EQI)##

        #termine di espansione: decadimento esponenziale che penalizza un'espansione fisica eccessiva
        term_expansion = np.exp(-(expansion-1)/4)

        #termine di penalizzazione per catene lunghe (1 complementare della frazione critica)
        term_long = 1.0 - long_chain_fraction

        mean_chain = np.mean(chain_lengths) if chain_lengths else 0
        median_chain = np.median(chain_lengths)

        max_chain = np.max(chain_lengths)

        #termine ponderato di lunghezza delle catene,
        #limitato rigorosamente nell'intervallo [0.0, 1.0] tramite clip.
        term_chain = np.clip(
            1.0 - max_chain / lThr,
            0.0,
            1.0
        )

        if chain_break_fraction is None:
        #ramo CPU: nessun dato empirico di chain break disponibile;
        #i pesi vengono rinormalizzati sulle metriche geometriche dell'embedding.
            eqi = (
                0.45*term_chain +
                0.35*term_expansion +
                0.20*term_long
            )
        else:
            #ramo QPU: integra esplicitamente la stabilità empirica misurata dai chain break calcolati.
            term_chain_break = 1.0 - float(chain_break_fraction)
            
            eqi = (
                0.35*term_chain +
                0.30*term_expansion +
                0.20*term_chain_break +
                0.15*term_long
            )   
        #clamp matematico finale per garantire che l'EQI sia un indice normalizzato in [0, 1]     
        eqi = max(0.0, min(1.0, eqi))

        good_embedding = eqi > 0.65

        return {
            "has_embedding": True,
            "eqi": eqi,
            "max_chain": max_chain,
            "mean_chain": mean_chain,
            "median_chain": median_chain,
            "physical_nodes": physical_nodes,
            "expansion": expansion,
            "long_chain_fraction": long_chain_fraction,
            "good_embedding": good_embedding
        }

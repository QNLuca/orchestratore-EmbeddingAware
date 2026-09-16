import time
import hybrid

from dwave.embedding.zephyr import find_clique_embedding
from minorminer import find_embedding
from subQUBO import SubQUBO

class BaseRouter(hybrid.Runnable):
    """Router base che gestisce le metriche e l'esecuzione su QPU/CPU."""
    def __init__(self, qpu_sampler, cpu_sampler, **runopts):
        super().__init__(**runopts)
        self.qpu_sampler = qpu_sampler
        self.cpu_sampler = cpu_sampler
        self.qpu_calls = 0
        self.cpu_calls = 0
        self.qpu_time = 0.0
        self.cpu_time = 0.0

    def _execute_route(self, state, route_to_qpu: bool, embedding=None):
        t0 = time.perf_counter()
        if route_to_qpu:
            self.qpu_calls += 1
            clean_state = state.updated(subsamples=None) #resetta subsamples per evitare errori di mismatch: https://support.dwavesys.com/hc/en-us/community/posts/360037074933-Error-message-when-trying-to-solve-sub-problems
            qpu_state = clean_state.updated(embedding=embedding) if embedding else state
            res_state = self.qpu_sampler.run(qpu_state).result()
            self.qpu_time += (time.perf_counter() - t0)
        else:
            self.cpu_calls += 1
            clean_state = state.updated(subsamples=None)
            res_state = self.cpu_sampler.run(clean_state).result()
            self.cpu_time += (time.perf_counter() - t0)

        return state.updated(subsamples=res_state.subsamples)

    @staticmethod
    def find_embedding(bqm_subproblem, target_graph=None):

        sub_bqm = bqm_subproblem
        num_vars = len(sub_bqm.variables)
        
        try:
            embedding = find_clique_embedding(
                            k=num_vars,
                            target_graph=target_graph
                        )
        except Exception:
                try:
                    global_edges = list(sub_bqm.quadratic.keys()) or [(v, v) for v in sub_bqm.variables]
                    embedding = find_embedding(
                        S=global_edges,
                        T=target_graph.edges,
                        timeout=100,
                        threads=4,
                        verbose=0
                    )
                except Exception:
                    embedding = {}
        
        is_embeddable = (len(embedding) == num_vars) and (num_vars > 0)

        return embedding, is_embeddable
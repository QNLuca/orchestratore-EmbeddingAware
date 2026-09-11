import time
from typing import Dict
import hybrid

from dwave.embedding.zephyr import find_clique_embedding
from minorminer import find_embedding

from subQUBO import SubQUBO

class ConfigurableRouterSampler(hybrid.Runnable):
    """Router che invia i sotto-problemi a QPU o CPU con decisione policy-dependent."""
    def __init__(
        self, 
        mode: str,
        global_embedding: Dict,
        is_all_embeddable: bool,
        qpu_sampler, 
        cpu_sampler, 
        predict_fn = None,
        predicted_solver_mqt = "SA",
        target_graph=None, 
        max_chain_length: int = 8, 
        max_size: int = 40, 
        max_density: float = 0.5, 
        verbose: bool = False,
        **runopts
    ):
        super().__init__(**runopts)
        self.mode = mode
        self.qpu_sampler = qpu_sampler
        self.cpu_sampler = cpu_sampler
        self.target_graph = target_graph
        self.max_chain_length = max_chain_length
        self.max_size = max_size
        self.max_density = max_density
        self.verbose = verbose
        self.predicted_solver_mqt = predicted_solver_mqt
        self.predict_fn = predict_fn
        
        self.global_embedding = global_embedding
        self.is_all_embeddable = is_all_embeddable
        
        self.qpu_calls = 0
        self.cpu_calls = 0
        self.qpu_time = 0.0
        self.cpu_time = 0.0

    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        sub_qubo = SubQUBO(bqm=sub_bqm)
        num_vars = len(sub_bqm.variables)
        avg_degree = sub_qubo.average_degree

        embedding = None
        is_embeddable = False

        if self.mode not in ["always_cpu"] and num_vars < 231: #numero massimo per rappresentazione grafo zephyr:
            try:
                embedding = find_clique_embedding(
                            k=num_vars,
                            target_graph=self.target_graph
                )
            except Exception:
                try:
                    global_edges = list(sub_bqm.quadratic.keys()) or [(v, v) for v in sub_bqm.variables]
                    embedding = find_embedding(
                        S=global_edges,
                        T=self.target_graph.edges,
                        timeout=100,
                        threads=4,
                        verbose=0
                    )
                except Exception:
                    embedding = {}
        else:
            embedding = {}

        is_embeddable = (len(embedding) == num_vars) and (num_vars > 0)

        # Decisione di Routing sui sotto-problemi
        route_to_qpu = False

        if self.mode == 'always_cpu':
            route_to_qpu = False #mai verso QPU
        elif self.mode == 'size_based':
            route_to_qpu = is_embeddable and (num_vars <= self.max_size)
        elif self.mode == 'only_my_features_based':
            route_to_qpu = is_embeddable and (num_vars <= self.max_size) and (sub_qubo.density <= self.max_density) and avg_degree < 15
        elif self.mode == 'mqt_qao':
            if self.predict_fn is not None:
                solver_scelto = self.predict_fn(sub_bqm)
            else:
                solver_scelto = self.predicted_solver_mqt

            if self.verbose:
                print(f"\n[MQT-QAO] Best solver scelto: {solver_scelto}")

            if (solver_scelto == 'QA') and is_embeddable:
                route_to_qpu = True
        elif self.mode=="always_qpu_embeddable":
            route_to_qpu = is_embeddable #se il sottoproblema è embeddable lo devio alla QPU a prescindere dai parametri di chain_length etc...

        target_device = "QPU" if route_to_qpu else "CPU"

        if self.verbose:
            print(
                f"DEBUG SUBPROBLEM, N={num_vars} | "
                f"Density={sub_qubo.density:.3f} | "
                f"AvgDeg={avg_degree:.1f} | "
                f"Embeddable={str(is_embeddable)} | "
                f"--> Route: {target_device}"
            )

        t0 = time.perf_counter()
        if route_to_qpu:
            self.qpu_calls += 1
            qpu_state = state.updated(embedding=embedding) if embedding else state
            res_state = self.qpu_sampler.run(qpu_state).result()
            self.qpu_time += (time.perf_counter() - t0)
        else:
            self.cpu_calls += 1
            res_state = self.cpu_sampler.run(state).result()
            self.cpu_time += (time.perf_counter() - t0)

        return state.updated(subsamples=res_state.subsamples)
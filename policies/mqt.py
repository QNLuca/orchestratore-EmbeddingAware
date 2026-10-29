import os
import hybrid
import dimod
import time
from typing import Dict, Any
import pickle
import pandas as pd
import numpy as np

from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation

from cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy
from policies.baseRouter import BaseRouter

#per sopprimere warning sklearn, tornare indietro di versione romperebbe l'attuale versione di Ocean SDK D-Wave 
import warnings
from sklearn.exceptions import InconsistentVersionWarning

from subQUBO import SubQUBO

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class MQTAORouter(BaseRouter):
    """Router basato sul classificatore ML di MQT-QAO."""
    def __init__(self, qpu_sampler, cpu_sampler, target_graph=None, chain_strength_prefactor=None, **kwargs):
        super().__init__(qpu_sampler=qpu_sampler, cpu_sampler=cpu_sampler,chain_strength_prefactor=chain_strength_prefactor, **kwargs)
        self.target_graph = target_graph

        self.chain_break_fractions = []
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.tn = 0  # True Negative
        self.fn = 0  # False Negative

        self.all_eqi = []

    def extract_qubo_features(self,bqm: dimod.BinaryQuadraticModel) -> pd.DataFrame:
        num_vars = len(bqm.variables)
        linear_coeffs = list(bqm.linear.values())
        num_a = len(linear_coeffs)
        mean_a = float(np.mean(linear_coeffs)) if num_a > 0 else 0.0
        var_a = float(np.var(linear_coeffs)) if num_a > 0 else 0.0

        quad_coeffs = list(bqm.quadratic.values())
        num_b = len(quad_coeffs)
        mean_b = float(np.mean(quad_coeffs)) if num_b > 0 else 0.0
        var_b = float(np.var(quad_coeffs)) if num_b > 0 else 0.0

        all_coeffs = linear_coeffs + quad_coeffs
        mean_all = float(np.mean(all_coeffs)) if len(all_coeffs) > 0 else 0.0
        var_all = float(np.var(all_coeffs)) if len(all_coeffs) > 0 else 0.0

        feature_names = [
            "var qubo", "n 1", "n 2", "avg 1", "avg 2", 
            "var 1", "var 2", "total average", "total variance"
        ]
        features = [[num_vars, num_a, num_b, mean_a, mean_b, var_a, var_b, mean_all, var_all]]
        return pd.DataFrame(features, columns=feature_names)

    def predict_mqt_best_solver(self,bqm: dimod.BinaryQuadraticModel) -> str:
            try:
                #importato modelli dal lavoro di Volpe D. et al, 2024: https://github.com/cda-tum/mqt-qao/tree/main/src/mqt/qao/model/RandomForest
                with open(os.path.join("modelsRF", "model.pkl"), "rb") as f:
                    model = pickle.load(f)
                with open(os.path.join("modelsRF", "Scaler.pkl"), "rb") as f:
                    scaler = pickle.load(f)
                with open(os.path.join("modelsRF", "ScalerKCross.pkl"), "rb") as f:
                    scalerk = pickle.load(f)
    
                features = self.extract_qubo_features(bqm)
                features_scaled = scalerk.transform(features)
                features_scaled = scaler.transform(features_scaled)
    
                prediction = int(model.predict(features_scaled)[0])
                label_map = {0: 'QA', 1: 'QAOA', 2: 'VQE', 3: 'GAS', 4: 'SA'}
                return label_map.get(prediction, "SA")
            except Exception as e:
                print(e)
                return "SA"

    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        sub_qubo = SubQUBO(sub_bqm)
        solver_scelto = self.predict_mqt_best_solver(sub_bqm)
        
        is_embeddable = False
        embedding = None

        if solver_scelto == 'QA':
            embedding, is_embeddable, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph)

        res_state = self._execute_route(state, route_to_qpu=is_embeddable, embedding=embedding)
        
        proxy_metrics = res_state.info.get("proxy_metrics", {})

        if proxy_metrics:
            from_qpu = proxy_metrics.get("from_qpu", False)
            good_embedding = proxy_metrics.get("good_embedding", False)
            if "chain_break_fraction" in proxy_metrics:
                self.chain_break_fractions.append(proxy_metrics["chain_break_fraction"])
            if "eqi" in proxy_metrics:
                self.all_eqi.append(proxy_metrics["eqi"])

            if good_embedding and from_qpu:
                self.tp += 1
            if not good_embedding and from_qpu:
                self.fp +=1
            if good_embedding and not from_qpu:
                self.fn +=1
            if not good_embedding and not from_qpu:
                self.tn +=1

        return state.updated(subsamples=res_state)

class MQTQAOPolicy(BasePolicy):
    """
    Policy basata sulla selezione predittiva del solutore tramite Machine Learning (modello Random Forest).

    Lavori a riguardo:
        - Volpe, D., et al. (2024). "A Predictive Approach for Selecting the Best Quantum Solver 
          for an Optimization Problem". IEEE QCE. arXiv:2408.03613.
        - Volpe, D., et al. (2024). "Towards an Automatic Framework for Solving Optimization 
          Problems with Quantum Computers". IEEE QSW. arXiv:2406.12840.
    """
    def __init__(self, **kwargs):
        super().__init__(name="mqt_qao", **kwargs)

    def solve(self, meta, bqm, global_embedding, is_all_embeddable, target_graph=None) -> Dict[str, Any]:
        lp_filename = meta.get("lp_filename", None)
        bestknown_solution = float(meta.get("bestknown_solution",None))

        invert_func = meta.get("invert_func", None)
                
        cqm = meta.get("cqm",None)

        is_maximize = meta.get("is_maximize", False)

        chain_strength_prefactor = meta.get('chain_strength_prefactor', None)
        
        qpu_target = hybrid.SimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)

        router = MQTAORouter(
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=target_graph,
            chain_strength_prefactor = chain_strength_prefactor
        )

        start_time = time.perf_counter()
                
        predicted_solver = router.predict_mqt_best_solver(bqm)

        qpu_target_global = hybrid.SimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)

        if predicted_solver == "QA" and is_all_embeddable:
            #start_time = time.perf_counter()

            if meta is not None and chain_strength_prefactor is not None:

                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=chain_strength_prefactor #aggiusta stima prefactor
                )
            else:
                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=1.414
                )
                        
            target_bqm = embed_bqm(
                bqm, 
                embedding=global_embedding, 
                target_adjacency=target_graph,
                chain_strength=chain_strength
            )

            init_state = hybrid.State.from_problem(target_bqm)

            
            end_state = qpu_target_global.run(init_state).result()

            physical_sampleset = end_state.samples

            break_freq_dict = chain_break_frequency(physical_sampleset, global_embedding)
            
            sampleset = unembed_sampleset(physical_sampleset, global_embedding, bqm, chain_break_method=majority_vote)

            final_time = time.perf_counter() - start_time

            total_wall_clock = final_time + self.embedding_time

            best_energy = float(sampleset.first.energy)

            if is_maximize:
                best_energy = -best_energy

            rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
            abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

            best_sample = dict(sampleset.first.sample)

            chain_break_fraction = 0.0         
            if break_freq_dict:
                #calcola la media dei valori nel dizionario
                chain_break_fraction = sum(break_freq_dict.values()) / len(break_freq_dict)

            good_embedding = SubQUBO.compute_advanced_embedding_metrics(global_embedding,bqm,chain_break_fraction)

            cqmResults = {}

            if len(cqm.constraints)>0:
                cqmResults = CqmUtilities.evaluate_cqm_feasibility(best_sample,invert_func, cqm)

            return {
                "instance_id": lp_filename,
                "is_maximization_objective": is_maximize,
                "policy": self.name,
                "best_energy": best_energy,
                "qplib_reference": bestknown_solution,
                "abs_gap": abs_gap,
                "rel_gap": rel_gap,
                "wall_clock_time": total_wall_clock,
                "time_global_embedding": self.embedding_time,
                "qpu_calls": 1,
                "cpu_calls": 0,
                
                "qpu_utilization_ratio": 1.0,
                "tp": 1 if good_embedding else 0, 
                "fp": 1 if not good_embedding else 0, "tn": 0, "fn": 0,

                "chain_break_fraction": chain_break_fraction,

                "decomposed": False,

                **cqmResults
            }

        decomposer = hybrid.Unwind(
            hybrid.EnergyImpactDecomposer(
                size=min(self.subproblem_size, len(bqm.variables)),
                rolling=True,
                rolling_history=0.3,
                traversal='bfs'
            )
        )

        subproblem_pipeline = (
            decomposer
            | hybrid.Map(router)
            | hybrid.Reduce(hybrid.Lambda(self._merge_substates))
            | hybrid.SplatComposer()
        )

        iteration = hybrid.ParallelBranches(
            hybrid.SimulatedAnnealingProblemSampler(num_reads=1, num_sweeps=10000),
            hybrid.TabuProblemSampler(),
            subproblem_pipeline
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True)

        main = hybrid.Loop(iteration, max_iter=self.max_iter, convergence=self.convergence)
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)

        final_state = main.run(init_state).result()

        final_time = time.perf_counter() - start_time

        total_wall_clock = final_time + self.embedding_time

        total_calls = router.qpu_calls + router.cpu_calls

        best_energy = float(final_state.samples.first.energy)

        if is_maximize:
            best_energy = -best_energy

        rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
        abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

        best_sample = dict(final_state.samples.first.sample)

        cqmResults = {}
        
        if len(cqm.constraints)>0:
            cqmResults = CqmUtilities.evaluate_cqm_feasibility(best_sample,invert_func, cqm)

        all_eqi = router.all_eqi
        
        if all_eqi:
            eqi_stats = {
                "eqi_mean": float(np.mean(all_eqi)),
                "eqi_std": float(np.std(all_eqi)),
                "eqi_min": float(np.min(all_eqi)),
                "eqi_max": float(np.max(all_eqi)),
                "eqi_median": float(np.median(all_eqi)),
            }
        else:
            eqi_stats = {
                "eqi_mean": 0.0,
                "eqi_std": 0.0,
                "eqi_min": 0.0,
                "eqi_max": 0.0,
                "eqi_median": 0.0,
            }
        
        avg_chain_break = float(np.mean(router.chain_break_fractions)) if router.chain_break_fractions else 0.0

        return {
            "instance_id": lp_filename,
            "is_maximization_objective": is_maximize,
            "policy": self.name,
            "best_energy": best_energy,
            "qplib_reference": bestknown_solution,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "wall_clock_time": total_wall_clock,
            "time_global_embedding": self.embedding_time,
            "qpu_calls": router.qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": router.qpu_calls/total_calls if total_calls > 0 else 0.0,
            "tp": router.tp, "fp": router.fp, "tn": router.tn, "fn": router.fn,

            "chain_break_fraction": avg_chain_break,
            
            **cqmResults,
            **eqi_stats
        }
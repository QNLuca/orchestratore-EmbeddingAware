import time
from typing import Any, Dict
import dimod
import hybrid
import numpy as np

from cqmUtilities import CqmUtilities
from subQUBO import SubQUBO
from policies.basePolicy import BasePolicy

from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation

from policies.baseRouter import BaseRouter

class OnlyEARouter(BaseRouter):
    """Invia a QPU ogni sub-problema per cui si riesce a calcolare un embedding valido."""
    def __init__(self, qpu_sampler, cpu_sampler, target_graph=None, chain_strength_prefactor=None, **kwargs):
        super().__init__(qpu_sampler=qpu_sampler, cpu_sampler=cpu_sampler, chain_strength_prefactor=chain_strength_prefactor, **kwargs)
        self.target_graph = target_graph

        self.chain_break_fractions = []
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.tn = 0  # True Negative
        self.fn = 0  # False Negative

        self.all_eqi = []


    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        sub_qubo_num_vars = len(sub_bqm.variables)
        sub_qubo = SubQUBO(bqm=sub_bqm)
        max_chain_length = 7


        embedding, is_embeddable, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph)

        chain_lengths = [len(chain) for chain in embedding.values()] if embedding else []
        max_chain = max(chain_lengths) if chain_lengths else float('inf')
        
        physical_nodes = sum(chain_lengths)
        
        expansion = physical_nodes / sub_qubo_num_vars if sub_qubo_num_vars > 0 else 0.0

        is_qpu_ready = max_chain <= max_chain_length and expansion <= 5.0
        
        route_to_qpu = is_embeddable and is_qpu_ready

        res_state = self._execute_route(state, route_to_qpu=route_to_qpu, embedding=embedding)

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

class OnlyEABasedPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="only_EA_based", **kwargs)

    def solve(self, meta, bqm, global_embedding, is_all_embeddable, target_graph=None) -> Dict[str, Any]:
        lp_filename = meta.get("lp_filename", None)

        bestknown_solution = float(meta.get("bestknown_solution",None))

        invert_func = meta.get("invert_func", None)
                
        cqm = meta.get("cqm",None)

        is_maximize = meta.get("is_maximize", False)

        chain_strength_prefactor = meta.get('chain_strength_prefactor')

        mybqm = SubQUBO(bqm)

        bqm_num_vars = mybqm.num_variables
        max_chain_length = 7

        chain_lengths_global = [len(chain) for chain in global_embedding.values()] if global_embedding else []
        max_chain_global = max(chain_lengths_global) if chain_lengths_global else float('inf')

        physical_nodes = sum(chain_lengths_global)

        expansion_global = physical_nodes / bqm_num_vars if bqm_num_vars > 0 else 0.0

        is_qpu_ready = max_chain_global <= max_chain_length and expansion_global <= 5.0

        qpu_target_global = hybrid.SimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)

        start_time = time.perf_counter()
        
        if is_all_embeddable and is_qpu_ready:
            
            if meta is not None and chain_strength_prefactor is not None:

                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=chain_strength_prefactor #aggiusta stima prefactor
                )
            else:
                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=1.414 #aggiusta stima prefactor
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

            break_freq_dict = chain_break_frequency(physical_sampleset, global_embedding) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.

            sampleset = unembed_sampleset(physical_sampleset, global_embedding, bqm, chain_break_method=majority_vote)
            
            final_time = (time.perf_counter() - start_time)

            total_wall_clock = final_time + self.embedding_time

            best_energy = float(sampleset.first.energy)

            best_sample = dict(sampleset.first.sample)

            if is_maximize:
                best_energy = -best_energy

            rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
            abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

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
                "qpu_calls": 1,
                "cpu_calls": 0,
                "qpu_utilization_ratio": 1.0,
                "tp": 1 if good_embedding else 0, 
                "fp": 1 if not good_embedding else 0, "tn": 0, "fn": 0,

                "chain_break_fraction": chain_break_fraction,

                "decomposed": False,
                
                **cqmResults
            }

        #start_time = time.perf_counter()
        qpu_target = hybrid.SimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)

        router = OnlyEARouter(
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=target_graph,
            chain_strength_prefactor = chain_strength_prefactor
        )

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

        best_energy = float(final_state.samples.first.energy)

        if is_maximize:
            best_energy = -best_energy

        rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
        abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

        best_sample = dict(final_state.samples.first.sample)

        cqmResults = {}
        
        if len(cqm.constraints)>0:
            cqmResults = CqmUtilities.evaluate_cqm_feasibility(best_sample,invert_func, cqm)

        total_calls = router.qpu_calls + router.cpu_calls

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
            "qpu_calls": router.qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": router.qpu_calls/total_calls if total_calls > 0 else 0.0,
            "tp": router.tp, "fp": router.fp, "tn": router.tn, "fn": router.fn,

            "chain_break_fraction": avg_chain_break,

            **cqmResults,
            **eqi_stats
            }
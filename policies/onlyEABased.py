import os
import time
from typing import Any, Dict
import dimod
import hybrid
import numpy as np

from utils.cqmUtilities import CqmUtilities
from subQUBO import SubQUBO
from policies.basePolicy import BasePolicy
from policies.baseRouter import BaseRouter

from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_KEY = os.getenv("DWAVE_API_TOKEN")

class OnlyEARouter(BaseRouter):
    """Invia a QPU ogni sub-problema per cui si riesce a calcolare un embedding valido."""
    def __init__(self, qpu_sampler, cpu_sampler, qpu_identity=None, target_graph=None, chain_strength_prefactor=None, seed = None,**kwargs):
        super().__init__(qpu_sampler=qpu_sampler, cpu_sampler=cpu_sampler, chain_strength_prefactor=chain_strength_prefactor, **kwargs)
        self.target_graph = target_graph
        self.seed = seed
        self.qpu_identity = qpu_identity

        self.chain_break_fractions = []
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.tn = 0  # True Negative
        self.fn = 0  # False Negative

        self.all_eqi = []
        self.all_max_chains = []
        self.all_mean_chains = []
        self.all_median_chains = []
        self.all_long_chain_fractions = []
        self.all_expansions = []
        self.all_physical_nodes = []


    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        sub_qubo_num_vars = len(sub_bqm.variables)
        sub_qubo = SubQUBO(bqm=sub_bqm)
        max_chain_length = 7


        embedding, is_embeddable, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph, self.seed)

        chain_lengths = [len(chain) for chain in embedding.values()] if embedding else []
        max_chain = max(chain_lengths) if chain_lengths else float('inf')
        
        physical_nodes = sum(chain_lengths)
        
        expansion = physical_nodes / sub_qubo_num_vars if sub_qubo_num_vars > 0 else 0.0

        is_qpu_ready = max_chain <= max_chain_length and expansion <= 5.0
        
        route_to_qpu = is_embeddable and is_qpu_ready

        res_state = self._execute_route(state, route_to_qpu=route_to_qpu, embedding=embedding, qpu_identity=self.qpu_identity)

        proxy_metrics = res_state.info.get("proxy_metrics", {})

        if proxy_metrics:
            from_qpu = proxy_metrics.get("from_qpu", False)
            good_embedding = proxy_metrics.get("good_embedding", False)
            if "chain_break_fraction" in proxy_metrics:
                self.chain_break_fractions.append(proxy_metrics["chain_break_fraction"])
            if "eqi" in proxy_metrics:
                self.all_eqi.append(proxy_metrics["eqi"])
                self.all_max_chains.append(proxy_metrics["max_chain"])
                self.all_mean_chains.append(proxy_metrics["mean_chain"])
                self.all_median_chains.append(proxy_metrics["median_chain"])
                self.all_long_chain_fractions.append(proxy_metrics["long_chain_fraction"])
                self.all_expansions.append(proxy_metrics["expansion"]) # == mean_chain
                self.all_physical_nodes.append(proxy_metrics["physical_nodes"])

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

    def solve(self, meta, bqm, global_embedding, is_all_embeddable,seed, qpu_identity, target_graph=None) -> Dict[str, Any]:
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
                                    
            if DWAVE_API_KEY:
                            
                qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity))
                
                fixed_composite = FixedEmbeddingComposite(qpu_sampler, embedding=global_embedding)

                t_start = time.perf_counter()

                res_sampleset = fixed_composite.sample(
                    bqm,
                    chain_strength=chain_strength,
                    num_reads=40,
                    annealing_time =20
                )

                t_end = time.perf_counter()

                wall_service_time = t_end - t_start

                final_time = t_end - start_time

                timing_info = res_sampleset.info.get('timing', {})

                chain_break_fraction = res_sampleset.record.chain_break_fraction #media fra tutte le letture
                
                #Raccolti tutti metadati temporali
                qpu_full_timing = {
                    "qpu_sampling_time": timing_info.get('qpu_sampling_time'),
                    "qpu_anneal_time_per_sample": timing_info.get('qpu_anneal_time_per_sample'),
                    "qpu_readout_time_per_sample": timing_info.get('qpu_readout_time_per_sample'),
                    "qpu_access_time": timing_info.get('qpu_access_time'),
                    "qpu_access_overhead_time": timing_info.get('qpu_access_overhead_time'),
                    "qpu_programming_time": timing_info.get('qpu_programming_time'),
                    "qpu_delay_time_per_sample": timing_info.get('qpu_delay_time_per_sample'),
                    "total_post_processing_time": timing_info.get('total_post_processing_time'),
                    "post_processing_overhead_time": timing_info.get('post_processing_overhead_time'),
                    "service_time": wall_service_time
                }                

                extra_metrics = {
                     **qpu_full_timing,
                }        
                        
            else:               
                t_build_0 = time.perf_counter()         
                target_bqm = embed_bqm(
                    bqm, 
                    embedding=global_embedding, 
                    target_adjacency=target_graph,
                    chain_strength=chain_strength
                )
                t_build = time.perf_counter() - t_build_0

                t_start_solve = time.perf_counter()
                init_state = hybrid.State.from_problem(target_bqm)

                end_state = qpu_target_global.run(init_state).result()

                physical_sampleset = end_state.samples
                t_solve_total = time.perf_counter() - t_start_solve

                t_unembed_0 = time.perf_counter()
                res_sampleset = unembed_sampleset(physical_sampleset, global_embedding, bqm, chain_break_method=majority_vote)
                t_unembed = time.perf_counter() - t_unembed_0

                final_time = time.perf_counter() - start_time

                chain_break_fraction = list(chain_break_frequency(physical_sampleset, global_embedding).values()) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.

                extra_metrics = {
                    "physical_build_time": t_build,
                    "unembed_time": t_unembed,
                    "t_solve": t_solve_total
                }

            total_wall_clock = final_time + self.embedding_time

            avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

            best_energy = float(res_sampleset.first.energy)

            best_sample = dict(res_sampleset.first.sample)

            if is_maximize:
                best_energy = -best_energy

            rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
            abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

            emb_metrics = SubQUBO.compute_advanced_embedding_metrics(global_embedding,bqm,avg_chain_break)

            good_embedding = bool(emb_metrics["good_embedding"])

            eqi = emb_metrics["eqi"]
            max_chain = float(emb_metrics["max_chain"])
            median_chain = float(emb_metrics["median_chain"])
            mean_chain = float(emb_metrics["mean_chain"])
            long_chain_fraction = float(emb_metrics["long_chain_fraction"])
            physical_nodes = float(emb_metrics["physical_nodes"])
            expansion = float(emb_metrics["expansion"])

            cqmResults = {}
                    
            if len(cqm.constraints)>0:
                cqmResults = CqmUtilities.evaluate_cqm_feasibility(best_sample,invert_func, cqm)

            return {
                "instance_id": lp_filename,
                "seed": seed,
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

                "chain_break_fraction": avg_chain_break,

                "decomposed": False,

                "chain_strength_prefactor": chain_strength_prefactor if chain_strength_prefactor is not None else 1.414,
                
                "eqi_mean": eqi,
                "max_chain": max_chain,
                "mean_chain": mean_chain,
                "median_chain": median_chain,
                "physical_nodes": physical_nodes,
                "expansion": expansion,
                "long_chain_fraction": long_chain_fraction,
                
                **extra_metrics,
                **cqmResults
            }

        #start_time = time.perf_counter()
        qpu_target = hybrid.SimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)

        router = OnlyEARouter(
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=target_graph,
            chain_strength_prefactor = chain_strength_prefactor,
            seed=seed,
            qpu_identity = qpu_identity
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
        all_max_chains = router.all_max_chains
        all_mean_chains = router.all_mean_chains
        all_median_chains = router.all_median_chains
        all_physical_nodes = router.all_physical_nodes
        all_expansions = router.all_expansions
        all_long_chain_fractions = router.all_long_chain_fractions

        if all_eqi:
            eqi_stats = {
                "eqi_mean": float(np.mean(all_eqi)),
                "eqi_std": float(np.std(all_eqi)),
                "eqi_min": float(np.min(all_eqi)),
                "eqi_max": float(np.max(all_eqi)),
                "eqi_median": float(np.median(all_eqi)),
                "max_chain": float(np.mean(all_max_chains)) if all_max_chains else 0.0,
                "mean_chain": float(np.mean(all_mean_chains)) if all_mean_chains else 0.0,
                "median_chain": float(np.mean(all_median_chains)) if all_median_chains else 0.0,
                "physical_nodes": float(np.mean(all_physical_nodes)) if all_physical_nodes else 0.0,
                "expansion": float(np.mean(all_expansions)) if all_expansions else 0.0,
                "long_chain_fraction": float(np.mean(all_long_chain_fractions)) if all_long_chain_fractions else 0.0,
            }
        else:
            eqi_stats = {
                "eqi_mean": 0.0,
                "eqi_std": 0.0,
                "eqi_min": 0.0,
                "eqi_max": 0.0,
                "eqi_median": 0.0,
                "max_chain": 0.0,
                "mean_chain": 0.0,
                "median_chain": 0.0,
                "physical_nodes": 0.0,
                "expansion": 0.0,
                "long_chain_fraction": 0.0,
            }

        avg_chain_break = float(np.mean(router.chain_break_fractions)) if router.chain_break_fractions else 0.0

        return {
            "instance_id": lp_filename,
            "seed": seed,
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
            "decomposed": True,

            "chain_strength_prefactor": chain_strength_prefactor if chain_strength_prefactor is not None else 1.414,

            "chain_break_fraction": avg_chain_break,

           **eqi_stats,
           **cqmResults            
            }
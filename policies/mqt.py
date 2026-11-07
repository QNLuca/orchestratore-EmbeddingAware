import os
import random
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

from utils.cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy
from policies.baseRouter import BaseRouter
from dwave.cloud.exceptions import SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout

#per sopprimere warning sklearn, tornare indietro di versione romperebbe l'attuale versione di Ocean SDK D-Wave 
import warnings
from sklearn.exceptions import InconsistentVersionWarning

from subQUBO import SubQUBO

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dotenv import load_dotenv

from utils.mockSolverOffline import OfflineQPUSolver

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_KEY = os.getenv("DWAVE_API_TOKEN")

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class MQTAORouter(BaseRouter):
    """Router basato sul classificatore ML di MQT-QAO."""
    def __init__(self, qpu_sampler, cpu_sampler, qpu_identity=None, target_graph=None, chain_strength_prefactor=None, seed=None, **kwargs):
        super().__init__(qpu_sampler=qpu_sampler, cpu_sampler=cpu_sampler,chain_strength_prefactor=chain_strength_prefactor, 
                         target_graph=target_graph, seed=seed, qpu_identity=qpu_identity, **kwargs)


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

        
        #is_embeddable = False
        #embedding = None
        #if solver_scelto == 'QA':
        #logicamente corretto ma così non ho embedding per valutare se la scelta di inoltro su CPU è buona o meno
        embedding, is_embeddable, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph, self.seed)

        route_to_qpu = solver_scelto == 'QA' and is_embeddable

        res_state = self._execute_route(state, route_to_qpu=route_to_qpu, embedding=embedding)
        
        proxy_metrics = res_state.info.get("proxy_metrics", {})

        if proxy_metrics:
            self._compute_proxy_metrics(proxy_metrics)


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

    def solve(self, meta, bqm, global_embedding, is_all_embeddable,seed,qpu_identity, target_graph=None) -> Dict[str, Any]:
        lp_filename = meta.get("lp_filename", None)
        bestknown_solution = float(meta.get("bestknown_solution",None))

        invert_func = meta.get("invert_func", None)
                
        cqm = meta.get("cqm",None)

        is_maximize = meta.get("is_maximize", False)

        chain_strength_prefactor = meta.get('chain_strength_prefactor', None)

        mybqm = SubQUBO(bqm)

        bqm_num_vars = bqm.num_variables
        bqm_avg_degree = mybqm.average_degree
        bqm_density = mybqm.density
        
        qpu_target = hybrid.SimulatedAnnealingProblemSampler(num_reads=40, num_sweeps=1000)
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)

        router = MQTAORouter(
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=target_graph,
            chain_strength_prefactor = chain_strength_prefactor,
            seed=seed,
            qpu_identity=qpu_identity
        )

        start_time = time.perf_counter()
                
        predicted_solver = router.predict_mqt_best_solver(bqm)

        qpu_target_global = hybrid.SimulatedAnnealingProblemSampler(num_reads=40, num_sweeps=1000)

        if predicted_solver == "QA" and is_all_embeddable:

            if meta is not None and chain_strength_prefactor is not None:

                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=chain_strength_prefactor #aggiusta stima prefactor
                )
            else:
                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=1.414
                )
            successLeap = False

            if DWAVE_API_KEY:
                                        
                qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity))
                
                fixed_composite = FixedEmbeddingComposite(qpu_sampler, embedding=global_embedding)

                t_start = time.perf_counter()

                
                max_retries = 3
                retry_delay = 1.0
                res_sampleset = None

                for attempt in range(1, max_retries + 1):
                    try:
                        t_start = time.perf_counter()   
                        res_sampleset = fixed_composite.sample(
                            bqm,
                            chain_strength=chain_strength,
                            num_reads=40,
                            annealing_time=20
                        )   
                        t_end = time.perf_counter()
                        wall_service_time = t_end - t_start 
                        final_time = t_end - start_time
                        successLeap = True
                        break  #successo, esco dal ciclo
                    
                    except (SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout) as e:
                        print(f"Tentativo {attempt}/{max_retries} fallito sulla QPU: {type(e).__name__}: {e}")
                        if attempt == max_retries:
                            print(f"QPU non disponibile dopo {max_retries} tentativi. "
                                            f"Procedo con il fallback locale.")
                            #esco dal ciclo
                            break
                        #attesa variabile prima del prossimo tentativo
                        sleep_time = retry_delay + random.uniform(0, 0.5)
                        time.sleep(sleep_time)
                        retry_delay *= 2
                    except Exception as e:
                        #errori non temporanei nessun altro retry
                        print(f"Errore non retryable su QPU: {type(e).__name__}: {e}")
                        break

                if successLeap and res_sampleset is not None:

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
                    }                
    
                    extra_metrics = {
                        "solve_time": wall_service_time,
                         **qpu_full_timing,
                    }        
                        
            if not DWAVE_API_KEY or not successLeap:               
                t_build_0 = time.perf_counter()         
                target_bqm = embed_bqm(
                    bqm, 
                    embedding=global_embedding, 
                    target_adjacency=target_graph,
                    chain_strength=chain_strength
                )
                t_build = time.perf_counter() - t_build_0

                ################# PER TEST LOCALE QPU_ACCESS_TIME #####################

                estimated_runtime_us = OfflineQPUSolver.estimate_qpu_access_time(
                    num_qubits=len(target_bqm.variables), 
                    num_reads=40, 
                    annealing_time=20
                )
                
                ################################################################

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
                    "solve_time": t_solve_total,
                    "estimated_leap_qpu_time_us": estimated_runtime_us,
                }

            total_wall_clock = final_time + self.embedding_time

            best_energy = float(res_sampleset.first.energy)

            if is_maximize:
                best_energy = -best_energy

            rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
            abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

            best_sample = dict(res_sampleset.first.sample)

            avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

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
                "num_vars": bqm_num_vars,
                "avg_degree": bqm_avg_degree,
                "density": bqm_density,
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
        all_max_chains = router.all_max_chains
        all_mean_chains = router.all_mean_chains
        all_median_chains = router.all_median_chains
        all_physical_nodes = router.all_physical_nodes
        all_expansions = router.all_expansions
        all_long_chain_fractions = router.all_long_chain_fractions

        all_estimated_qpu = router.estimated_runtime_total

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
            "num_vars": bqm_num_vars,
            "avg_degree": bqm_avg_degree,
            "density": bqm_density,
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
            "decomposed": True,

            "chain_strength_prefactor": chain_strength_prefactor if chain_strength_prefactor is not None else 1.414,

            "chain_break_fraction": avg_chain_break,
            "physical_build_time": router.total_build_time,
            "solve_time": router.total_solve_time,
            "unembed_time": router.total_unembed_time,
            
           **eqi_stats,
           **cqmResults,
           "estimated_leap_qpu_time_us": all_estimated_qpu,        
        }
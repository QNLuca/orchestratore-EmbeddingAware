import random
import threading
import time
import hybrid
from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation
import dwave.graphs as dnx

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.cloud.exceptions import SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout
import numpy as np

from subQUBO import SubQUBO
from utils.mockSolverOffline import OfflineQPUSolver

class BaseRouter(hybrid.Runnable):
    """Router base che gestisce le metriche e l'esecuzione su QPU/CPU."""
    def __init__(self, qpu_sampler, cpu_sampler, seed=None,qpu_identity=None, target_graph=None, chain_strength_prefactor = None, **runopts):
        super().__init__(**runopts)
        self.qpu_sampler = qpu_sampler
        self.cpu_sampler = cpu_sampler
        self.qpu_calls = 0
        self.cpu_calls = 0
        self.chain_strength_prefactor = chain_strength_prefactor

        self.target_graph = target_graph
        self.seed = seed
        self.qpu_identity = qpu_identity

        self.total_build_time = 0.0
        self.total_solve_time = 0.0
        self.total_unembed_time = 0.0
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
        self.estimated_runtime_total = 0

        self.lock = threading.Lock()

    def _compute_proxy_metrics(self, proxy_metrics: dict):
                
        with self.lock:
            self.total_solve_time += float(proxy_metrics["solve_time"]) #sia qpu che cpu
            from_qpu = proxy_metrics.get("from_qpu", False)
            real_qpu = proxy_metrics.get("real_qpu", False)
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
            if from_qpu and not real_qpu:
                self.total_build_time += float(proxy_metrics["physical_build_time"])
                self.total_unembed_time += float(proxy_metrics["unembed_time"])
                self.estimated_runtime_total += float(proxy_metrics["estimated_runtime_us"])

            if good_embedding and from_qpu:
                self.tp += 1
            if not good_embedding and from_qpu:
                self.fp +=1
            if good_embedding and not from_qpu:
                self.fn +=1
            if not good_embedding and not from_qpu:
                self.tn +=1

    def _execute_route(self, state, route_to_qpu: bool, embedding=None):

        if route_to_qpu:
            self.qpu_calls += 1
            qpu_state_clean = state.updated(subsamples=None) #resetta subsamples per evitare errori di mismatch: https://support.dwavesys.com/hc/en-us/community/posts/360037074933-Error-message-when-trying-to-solve-sub-problems
            
            qpu_state = qpu_state_clean.subproblem
            
            if self.chain_strength_prefactor is not None:
                chain_strength = uniform_torque_compensation(
                    qpu_state, embedding=embedding, prefactor=self.chain_strength_prefactor
                )
            else:
                chain_strength = uniform_torque_compensation(
                    qpu_state, embedding=embedding, prefactor=1.414
                )

            successLeap = False

            if self.qpu_identity:
                #sono per forza in un benchamrk e dunque se ho API ho anche qpu_identity
                qpu_sampler = DWaveSampler(solver=dict(identity=self.qpu_identity))
                
                fixed_composite = FixedEmbeddingComposite(qpu_sampler, embedding=embedding)

                max_retries = 3
                retry_delay = 1.0
                res_sampleset = None

                for attempt in range(1, max_retries + 1):
                    try:
                        t_start = time.perf_counter()

                        res_sampleset = fixed_composite.sample(
                            qpu_state,
                            chain_strength=chain_strength,
                            num_reads=40,
                            annealing_time = 20
                        )
                        t_end = time.perf_counter()
                        wall_service_time = t_end - t_start
                        successLeap = True
                        break

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

                #estrazione di timing
                if successLeap and res_sampleset is not None:
                    timing_info = res_sampleset.info.get('timing', {})

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

                    chain_break_fraction = res_sampleset.record.chain_break_fraction #media fra tutte le letture

                    avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

                    emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, qpu_state, avg_chain_break)

                    extra_metrics = {
                        "from_qpu": True,
                        "real_qpu": True,
                        "chain_break_fraction": avg_chain_break,
                        "solve_time": wall_service_time,
                         **qpu_full_timing,
                         **emb_metrics
                    }
                
            if not self.qpu_identity or not successLeap:              
                t_build_0 = time.perf_counter()  
                target_bqm = embed_bqm(
                    qpu_state, 
                    embedding=embedding, 
                    target_adjacency=dnx.zephyr_graph(12),
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

                end_state = self.qpu_sampler.run(init_state).result()

                physical_sampleset = end_state.samples
                t_solve_total = time.perf_counter() - t_start_solve

                t_unembed_0 = time.perf_counter()
                res_sampleset = unembed_sampleset(physical_sampleset, embedding, qpu_state, chain_break_method=majority_vote)
                t_unembed = time.perf_counter() - t_unembed_0

                chain_break_fraction = list(chain_break_frequency(physical_sampleset, embedding).values()) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.
                
                avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

                emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, qpu_state, avg_chain_break)

                extra_metrics = {
                                "from_qpu": True,
                                "real_qpu": False,
                                "physical_build_time": t_build,
                                "unembed_time": t_unembed,
                                "solve_time": t_solve_total,
                                "estimated_runtime_us": estimated_runtime_us,
                                "chain_break_fraction": avg_chain_break,
                                **emb_metrics
                            }

        else:
            self.cpu_calls += 1
            clean_state = state.updated(subsamples=None)
            final_state = self.cpu_sampler.run(clean_state).result()

            res_sampleset = final_state.subsamples

            if embedding:
                 emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, state.subproblem, None)
            else:
                emb_metrics = {
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
            
            extra_metrics = {
                "from_qpu": False,
                **emb_metrics
            }


        if hasattr(res_sampleset, 'info') and isinstance(res_sampleset.info, dict):
            res_sampleset.info["proxy_metrics"] = extra_metrics

        return res_sampleset
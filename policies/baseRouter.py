import time
import hybrid
from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation
import dwave.graphs as dnx

from dwave.system import DWaveSampler, FixedEmbeddingComposite
import numpy as np

from subQUBO import SubQUBO

class BaseRouter(hybrid.Runnable):
    """Router base che gestisce le metriche e l'esecuzione su QPU/CPU."""
    def __init__(self, qpu_sampler, cpu_sampler, chain_strength_prefactor = None, **runopts):
        super().__init__(**runopts)
        self.qpu_sampler = qpu_sampler
        self.cpu_sampler = cpu_sampler
        self.qpu_calls = 0
        self.cpu_calls = 0
        self.chain_strength_prefactor = chain_strength_prefactor

    def _execute_route(self, state, route_to_qpu: bool, qpu_identity, embedding=None):

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

            if qpu_identity:
                qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity))
                
                fixed_composite = FixedEmbeddingComposite(qpu_sampler, embedding=embedding)

                t_start = time.perf_counter()

                res_state = fixed_composite.sample(
                    qpu_state,
                    chain_strength=chain_strength,
                    num_reads=40,
                    annealing_time = 20
                )
                t_end = time.perf_counter()
                
                wall_service_time = t_end - t_start

                timing_info = res_state.info.get('timing', {})
                
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

                chain_break_fraction = res_state.record.chain_break_fraction #media fra tutte le letture

                avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0
                
                emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, qpu_state, avg_chain_break)

                extra_metrics = {
                    "from_qpu": True,
                    "chain_break_fraction": avg_chain_break,
                     **qpu_full_timing,
                     **emb_metrics
                }
                
            else:              
                target_bqm = embed_bqm(
                    qpu_state, 
                    embedding=embedding, 
                    target_adjacency=dnx.zephyr_graph(12),
                    chain_strength=chain_strength
                )

                init_state = hybrid.State.from_problem(target_bqm)

                end_state = self.qpu_sampler.run(init_state).result()

                physical_sampleset = end_state.samples

                chain_break_fraction = list(chain_break_frequency(physical_sampleset, embedding).values()) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.

                avg_chain_break = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

                res_state = unembed_sampleset(physical_sampleset, embedding, qpu_state, chain_break_method=majority_vote)

                emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, qpu_state, avg_chain_break)

                extra_metrics = {
                                "from_qpu": True,
                                "chain_break_fraction": avg_chain_break,
                                **emb_metrics
                            }

        else:
            self.cpu_calls += 1
            clean_state = state.updated(subsamples=None)
            final_state = self.cpu_sampler.run(clean_state).result()

            res_state = final_state.subsamples

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


        if hasattr(res_state, 'info') and isinstance(res_state.info, dict):
            res_state.info["proxy_metrics"] = extra_metrics

        return res_state
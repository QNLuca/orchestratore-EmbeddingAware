import os
import time
import numpy as np
import redis
from rq import Worker, Queue, Connection, get_current_job
import dimod
import hybrid
import dwave.graphs as dnx
import pickle
from utils.mockSolverOffline import OfflineQPUSolver 

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.cloud.exceptions import SolverError, InvalidAPIResponseError, PollingTimeout, ProblemUploadError, RequestTimeout
import random

from subQUBO import SubQUBO
from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)


def solve_subqubo_rq(sub_bqm: dimod.BinaryQuadraticModel,
                  sampler_type: str, embedding=None, chain_strength_prefactor: float = None, 
                  realQPU: bool = False, qpu_identity = None) -> dimod.SampleSet:
        """
        Risolve direttamente il sub-QUBO con il sampler indicato.
        """

        if sampler_type == "qpu_queue":
            #ho un embedding

            if chain_strength_prefactor is not None:
                            
                chain_strength = uniform_torque_compensation(
                        sub_bqm, embedding=embedding, prefactor=chain_strength_prefactor #aggiusta stima prefactor
                )
            
            else:
                chain_strength = uniform_torque_compensation(
                    sub_bqm, embedding=embedding, prefactor=1.414
                )

            successLeap = False

            if realQPU: 
                if qpu_identity:
                    qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity))
                else: 
                    DWaveSampler(solver={'topology__type': 'zephyr'})

                fixed_composite = FixedEmbeddingComposite(qpu_sampler, embedding=embedding)

                max_retries = 3
                retry_delay = 1.0
                res_sampleset = None
                wall_service_time = 0.0

                for attempt in range(1, max_retries + 1):
                    try:
                        t_start = time.perf_counter()

                        res_sampleset = fixed_composite.sample(
                            sub_bqm,
                            chain_strength=chain_strength,
                            num_reads=40,
                            annealing_time=20
                        )

                        t_end = time.perf_counter()
                        wall_service_time = t_end - t_start
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
                        print(f"[QPU] errore non retryable: {type(e).__name__}: {e}")
                        break

                #estrazione di timing
                if successLeap and res_sampleset is not None:
                    timing_info = res_sampleset.info.get('timing', {})

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

                    chain_break_fraction = res_sampleset.record.chain_break_fraction
                    avg_chain_break_fraction = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0
                    emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, sub_bqm, avg_chain_break_fraction)

                    extra_metrics = {
                        "from_qpu": True,
                        "real_qpu": True,
                        "qpu_leap_failure": False,
                        "solve_time": wall_service_time,
                        "chain_break_fraction": avg_chain_break_fraction,
                        **qpu_full_timing,
                        **emb_metrics
                    }

            if not realQPU or not successLeap:

                target_graph = dnx.zephyr_graph(12)
                if realQPU:
                    #sono qui dopo retry e fallimento su QPU reale
                    if qpu_identity:
                        qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity))
                    else: 
                        DWaveSampler(solver={'topology__type': 'zephyr'})
                    target_graph = qpu_sampler.to_networkx_graph() #uso almeno topologia reale seppur su SA

                t_build_0 = time.perf_counter()
                target_bqm = embed_bqm(
                    sub_bqm, 
                    embedding=embedding, 
                    target_adjacency= target_graph,
                    chain_strength=chain_strength
                )
                t_build = time.perf_counter() - t_build_0

                t_solve_0 = time.perf_counter()

                #inizializzo lo State di dwave-hybrid usando il BQM fisico (target_bqm)
                init_state = hybrid.State.from_problem(target_bqm)

                #istanzio ed eseguo il sampler da dwave-hybrid
                sa_workflow = hybrid.SimulatedAnnealingProblemSampler(num_reads=40, num_sweeps=1000)
                end_state = sa_workflow.run(init_state).result()

                #estraggo il physical_sampleset risultante
                physical_sampleset = end_state.samples

                t_solve = time.perf_counter() - t_solve_0

                chain_break_fraction = list(chain_break_frequency(physical_sampleset, embedding).values()) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.

                avg_chain_break_fraction = float(np.mean(chain_break_fraction)) if len(chain_break_fraction) > 0 else 0.0

                t_unembed_0 = time.perf_counter()
                logical_sampleset = unembed_sampleset(physical_sampleset, embedding, sub_bqm, chain_break_method=majority_vote)
                t_unembed = time.perf_counter() - t_unembed_0

                ################# PER TEST LOCALE QPU_ACCESS_TIME #####################
                offline_solver = OfflineQPUSolver()

                estimated_runtime_us = offline_solver.estimate_qpu_access_time(
                    num_qubits=len(target_bqm.variables), 
                    num_reads=40, 
                    annealing_time=20
                )
                ################################################################

                res_sampleset = logical_sampleset

                emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, sub_bqm, avg_chain_break_fraction)

                extra_metrics = {
                    "from_qpu": True,
                    "real_qpu": False,
                    "qpu_leap_failure": not successLeap if realQPU else False,
                    "physical_build_time": t_build,
                    "solve_time": t_solve,
                    "unembed_time": t_unembed,
                    "estimated_runtime_us": estimated_runtime_us,
                    "chain_break_fraction": avg_chain_break_fraction,
                    "chain_strength": chain_strength,
                    **emb_metrics
                }
            
        else: #cpu_queue
            init_state = hybrid.State(subproblem=sub_bqm)
            t_solve_0 = time.perf_counter()
            final_state = hybrid.TabuSubproblemSampler(num_reads=20).run(init_state).result()
            t_solve = time.perf_counter() - t_solve_0

            res_sampleset = final_state.subsamples

            if embedding:
                emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, sub_bqm, None)
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
                "solve_time": t_solve,
                **emb_metrics
            }

        if hasattr(res_sampleset, 'info') and isinstance(res_sampleset.info, dict):
            res_sampleset.info["proxy_metrics"] = extra_metrics

        job = get_current_job()
        if job and job.connection:
            payload = pickle.dumps(res_sampleset)
            job.connection.publish(f"job_done:{job.id}", payload)

        return res_sampleset


if __name__=="__main__":
    #avvio del worker RQ
    target_queue = os.getenv("TARGET_QUEUE", "cpu_queue")
    
    redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)
    
    print(f"[RQ WORKER] Avvio worker in ascolto sulla coda: {target_queue}")
    
    with Connection(redis_conn):
        worker = Worker([Queue(target_queue)])
        worker.work()
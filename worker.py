import os
import time
import redis
from rq import Worker, Queue, Connection, get_current_job
import dimod
import hybrid
import dwave.graphs as dnx
import pickle

from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation

from dotenv import load_dotenv

from subQUBO import SubQUBO

#carica le variabili contenute nel file .env
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)


def solve_subqubo_rq(sub_bqm: dimod.BinaryQuadraticModel,
                  sampler_type: str, embedding=None, chain_strength_prefactor: float = None) -> dimod.SampleSet:
        """
        Risolve direttamente il sub-QUBO con il sampler indicato.
        """

        if sampler_type == "QPU":

            if chain_strength_prefactor is not None:

                chain_strength = uniform_torque_compensation(
                    sub_bqm, embedding=embedding, prefactor=chain_strength_prefactor #aggiusta stima prefactor
                )

            else:
                chain_strength = uniform_torque_compensation(
                    sub_bqm, embedding=embedding, prefactor=1.414
                )


            t_build_0 = time.perf_counter()
            target_bqm = embed_bqm(
                sub_bqm, 
                embedding=embedding, 
                target_adjacency=dnx.zephyr_graph(12),
                chain_strength=chain_strength
            )
            t_build = time.perf_counter() - t_build_0

            t_solve_0 = time.perf_counter()
            
            #inizializzo lo State di dwave-hybrid usando il BQM fisico (target_bqm)
            init_state = hybrid.State.from_problem(target_bqm)

            #istanzio ed eseguo il sampler da dwave-hybrid
            sa_workflow = hybrid.SimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)
            end_state = sa_workflow.run(init_state).result()

            #estraggo il physical_sampleset risultante
            physical_sampleset = end_state.samples

            t_solve = time.perf_counter() - t_solve_0

            break_freq_dict = chain_break_frequency(physical_sampleset, embedding) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.

            chain_break_fraction = 0.0  
            if break_freq_dict:
                #calcolo la media dei valori nel dizionario
                chain_break_fraction = sum(break_freq_dict.values()) / len(break_freq_dict)

            t_unembed_0 = time.perf_counter()
            logical_sampleset = unembed_sampleset(physical_sampleset, embedding, sub_bqm, chain_break_method=majority_vote)
            t_unembed = time.perf_counter() - t_unembed_0

            res_sampleset = logical_sampleset

            emb_metrics = SubQUBO.compute_advanced_embedding_metrics(embedding, sub_bqm, chain_break_fraction)

            extra_metrics = {
                "from_qpu": True,
                "physical_build_time": t_build,
                "solve_time": t_solve,
                "unembed_time": t_unembed,
                "chain_break_fraction": chain_break_fraction,
                "chain_strength": chain_strength,
                **emb_metrics
            }
            
        else: #CPU
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
                "physical_build_time": 0.0,
                "solve_time": t_solve,
                "unembed_time": 0.0,
                "chain_break_fraction": 0.0,
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
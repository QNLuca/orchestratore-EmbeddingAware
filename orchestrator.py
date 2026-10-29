import hashlib
import math
import os
import threading
import time
from typing import Any, Dict, Optional
import uuid

import dimod
import dwave.graphs as dnx
import hybrid
import numpy as np
import pickle

from dwave.embedding import embed_bqm, unembed_sampleset, chain_break_frequency
from dwave.embedding.chain_breaks import majority_vote
from dwave.embedding.chain_strength import uniform_torque_compensation

import redis
from rq import Queue

from subQUBO import SubQUBO
from cqmUtilities import CqmUtilities

from dotenv import load_dotenv
from cachetools import LRUCache

#carica le variabili contenute nel file .env
load_dotenv()

#config Redis
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

REDIS_POOL = redis.ConnectionPool(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    db=0, 
    decode_responses=False,
    max_connections=50  #per assicurare connessioni sufficienti al ThreadPool
)


class AsyncRedisDispatcherSampler(hybrid.Runnable):
    """
    Dispatcher asiincrono, invia il task a Redis e restituisce un Future.
    Consente a hybrid.Map di inviare N sub-QUBO in parallelo ai worker.
    """
    def __init__(self, target_queue: str, connection_pool: redis.ConnectionPool, timeout: int = 500, max_workers: int = 8, meta: Dict = None, **runopts):
        super().__init__(**runopts)
        self.target_queue = target_queue
        self.timeout = timeout
        self.pool_ref = connection_pool
        self.meta = meta

        #Istanza Redis e Coda PERSISTENTI create una sola volta
        self.r_client = redis.Redis(connection_pool=self.pool_ref, decode_responses=False)
        self.q = Queue(self.target_queue, connection=self.r_client)

        #tracciamento metriche interne
        self.total_build_time = 0.0
        self.total_solve_time = 0.0
        self.total_unembed_time = 0.0
        self.chain_break_fractions = []

        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.tn = 0  # True Negative
        self.fn = 0  # False Negative

        self.all_eqi = []

    def next(self, state, **kwargs):

        sub_bqm = state.subproblem
        if sub_bqm is None or len(sub_bqm.variables) == 0:
            return state

        embedding = getattr(state, 'embedding', None)

        job_id = str(uuid.uuid4())

        chain_strength_prefactor = None
        if self.meta is not None:
            chain_strength_prefactor = self.meta.get("chain_strength_prefactor")

        #sottomissione del task alla funzione eseguita dal worker RQ
        job = self.q.enqueue(
            'worker.solve_subqubo_rq', 
            args=(sub_bqm, self.target_queue, embedding, chain_strength_prefactor),
            timeout=self.timeout,
            result_ttl=300,  #mantengo il risultato per 500 secondi prima del cleanup
            job_id=job_id
        )

        channel = f"job_done:{job.id}"
        pubsub = self.r_client.pubsub()
        try:
            pubsub.subscribe(channel)
            start_time = time.time()

            start_sub = time.time()
            while time.time() - start_sub < 3.0:
                msg = pubsub.get_message(ignore_subscribe_messages=False, timeout=0.2)
                if msg and msg['type'] == 'subscribe':
                    break
            else:
                raise TimeoutError(f"Impossibile sottoscriversi al canale {channel}")

            #rimango in attesa del messaggio di completamento
            start_time = time.time()
            while (time.time() - start_time) < self.timeout:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message['type'] == 'message':
                    
                    sampleset = pickle.loads(message['data'])

                    #estrazione metriche del proxy inviate dal worker
                    proxy_metrics = sampleset.info.get("proxy_metrics", {})
                    if proxy_metrics:
                        self.total_build_time += proxy_metrics.get("physical_build_time", 0.0)
                        self.total_solve_time += proxy_metrics.get("solve_time", 0.0)
                        self.total_unembed_time += proxy_metrics.get("unembed_time", 0.0)
                        good_embedding = proxy_metrics.get("good_embedding", False)
                        from_qpu = proxy_metrics.get("from_qpu", False)
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

                    return state.updated(subsamples=sampleset)
        finally:
            pubsub.close()

        raise TimeoutError(f"Timeout RQ ({self.timeout}s) sulla coda '{self.target_queue}' per job {job.id}")



class RouterSampler(hybrid.Runnable):
    """Router che invia i sotto-problemi a QPU o CPU in base all'embedding."""
    def __init__(
        self, 
        mode: str,
        global_embedding: Dict,
        is_all_embeddable: bool,
        qpu_dispatcher: hybrid.Runnable, 
        cpu_dispatcher: hybrid.Runnable,
        target_graph=None, 
        max_chain_length: int = 7,
        verbose: bool = 0,
        **runopts
    ):
        super().__init__(**runopts)
        self.mode = mode
        self.target_graph = target_graph
        self.max_chain_length = max_chain_length

        self.qpu_dispatcher = qpu_dispatcher
        self.cpu_dispatcher = cpu_dispatcher
        
        self.global_embedding = global_embedding
        self.is_all_embeddable = is_all_embeddable
        
        self.qpu_calls = 0
        self.cpu_calls = 0

        self.verbose = verbose

        #lista per tracciare gli intervalli temporali degli embedding concorrenti
        self.embedding_intervals = []

        #raccolta metriche di routing
        self.total_routing_time = 0.0

        #cache strutturale: frozenset(variabili_subproblema) a embedding
        self.embedding_cache = LRUCache(maxsize=100)

        self.lock = threading.Lock()

    def next(self, state, **kwargs):

        sub_bqm = state.subproblem
        
        sub_qubo = SubQUBO(bqm=sub_bqm)

        sub_qubo_num_vars = sub_bqm.num_variables
        sub_qubo_avg_degree = sub_qubo.average_degree
        sub_qubo_density = sub_qubo.density

        edges_hash = hashlib.md5(str(sorted(sub_bqm.quadratic.keys())).encode()).hexdigest()

        #chiave di cache compatta basata su interi e hash degli archi
        cache_key = (frozenset(sub_bqm.variables), edges_hash)

        max_chain = float('inf')

        t_embed_start = time.perf_counter()
        embedding = None
        computed_emb_is_all = False

        ###STRATEGIA 1: Controllo nella Cache Strutturale delle iterazioni precedenti
        if embedding is None:
            with self.lock:
                embedding = self.embedding_cache.get(cache_key)

        #STRATEGIA 2: Slicing dall'Embedding Globale (se interamente embeddabile)
        if self.global_embedding and self.is_all_embeddable:
            sub_emb = {v: self.global_embedding[v] for v in sub_bqm.variables if v in self.global_embedding}
            if len(sub_emb) == sub_qubo_num_vars:
                embedding = sub_emb

        #STRATEGIA 3: Fallback su minorminer se non trovato già altrove
        if embedding is None:
            computed_emb, computed_emb_is_all, _ = sub_qubo.compute_embedding(sub_bqm, self.target_graph)
            embedding = computed_emb

            if computed_emb and computed_emb_is_all:
                with self.lock:
                    self.embedding_cache[cache_key] = embedding

        t_embed_end = time.perf_counter()


        with self.lock:
            self.embedding_intervals.append((t_embed_start, t_embed_end))

    
        t_route_0 = time.perf_counter()

        chain_lengths = [len(chain) for chain in embedding.values()] if embedding else []
        
        max_chain = max(chain_lengths) if chain_lengths else float('inf')

        physical_nodes = sum(chain_lengths)
        expansion = physical_nodes / sub_qubo_num_vars if sub_qubo_num_vars > 0 else 0.0

        #Problematici:
        # - grado > 80 and num_vars > 192 and density > 0.3
        # - num_vars > 250 e density >= 0.1

        is_qpu_ready = not ((sub_qubo_avg_degree > 52 and sub_qubo_num_vars > 192 and sub_qubo_density > 0.052) 
                                          or (sub_qubo_num_vars > 250 and sub_qubo_density >= 0.052))

        route_to_qpu = (
            computed_emb_is_all and 
            is_qpu_ready and
            max_chain <= self.max_chain_length and
            expansion <= 5.0
            
            )

        target_device = "QPU" if route_to_qpu else "CPU"

        with self.lock:
            self.total_routing_time += (time.perf_counter() - t_route_0)

        if self.verbose:
            print(
                f"DEBUG SUBPROBLEM, N={sub_qubo_num_vars} | "
                f"Density={sub_qubo_density:.3f} | "
                f"AvgDeg={sub_qubo_avg_degree} | "
                f"Embeddable={str(computed_emb_is_all)} | "
                f"MaxChain={max_chain if max_chain != float('inf') else 'N/A'} | "
                f"--> Route: {target_device}"
            )

        t0 = time.perf_counter()
        if route_to_qpu:
            qpu_state = state.updated(embedding=embedding) if embedding else state
            
            res_state = self.qpu_dispatcher.run(qpu_state).result()

            with self.lock:
                self.qpu_calls += 1

        else:            
            cpu_state = state.updated(embedding=embedding) if embedding else state

            res_state = self.cpu_dispatcher.run(cpu_state).result()
            
            with self.lock:
                self.cpu_calls += 1

        return state.updated(subsamples=res_state.subsamples)

    @property
    def total_sub_embedding_time(self) -> float:
        """Calcola il vero wall-clock time fondendo gli intervalli concorrenti."""
        with self.lock:
            if not self.embedding_intervals:
                return 0.0

            #ordina gli intervalli in base al tempo di inizio
            sorted_intervals = sorted(self.embedding_intervals, key=lambda x: x[0])
            
            merged = [sorted_intervals[0]]
            for current_start, current_end in sorted_intervals[1:]:
                prev_start, prev_end = merged[-1]
                #se c'è sovrapposizione, estendo l'intervallo precedente
                if current_start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, current_end))
                else:
                    merged.append((current_start, current_end))

            #sommo la durata degli intervalli fusi
            total_duration = sum(end - start for start, end in merged)
            return total_duration


class EmbeddingAwareOrchestrator:
    """Orchestratore per la risoluzione ottimizzata di problemi QUBO/BQM."""
    def __init__(
        self,
        policy_mode: str = 'embedding_aware', 
        target_graph=None,
        subproblem_size: Optional[int] = None,
        max_iter: int = 10,
        convergence: int = 3,
    ):
        self.policy_mode = policy_mode
        self.target_graph = target_graph or dnx.zephyr_graph(12)
        self.subproblem_size = subproblem_size
        self.max_iter = max_iter
        self.convergence = convergence

        #connessione a Redis
        self.redis_pool = REDIS_POOL

    def _calculate_subproblem_size(self, bqm_len: int) -> int:
        if self.subproblem_size is not None:
            return min(self.subproblem_size, bqm_len)
        raw_size = int(math.ceil(2.0 * math.sqrt(bqm_len)))
        return min(max(10, min(raw_size, 400)), bqm_len)

    def solve(self, bqm: dimod.BinaryQuadraticModel, 
              meta: Dict,
              global_embedding_bench: Optional[Dict] = None, 
              is_all_embeddable_bench: Optional[bool] = None, 
              embedding_time: Optional[float] = None, 
              benchmark: bool = False) -> Dict[str, Any]:
        """
        Risolve un BQM applicando il routing dinamico basato su embedding e topologia.
        """
        invert_func = meta.get("invert_func", None)
                
        cqm = meta.get("cqm",None)

        lp_filename = meta.get("lp_filename", None)

        bestknown_solution = float(meta.get("bestknown_solution",None))

        is_maximize = meta.get("is_maximize", False)

        chain_strength_prefactor = meta.get('chain_strength_prefactor',None)

        verbose = meta.get('verbose', False)

        mybqm = SubQUBO(bqm)

        max_chain_length = 7
        max_global_size = 1337 #embedding più grande portato a termine con successo con 300s di timeout nei test
        
        #features del bqm
        bqm_num_vars = bqm.num_variables
        bqm_avg_degree = mybqm.average_degree
        bqm_density = mybqm.density


        is_qpu_ready = not ((bqm_avg_degree > 52 and bqm_num_vars > 192 and bqm_density > 0.052) 
                                  or (bqm_num_vars > 250 and bqm_density >= 0.052))

        #Problematici:
        # - grado > 80 and num_vars > 192 and density > 0.3
        # - num_vars > 250 and density > 0.1

        effective_sub_size = self._calculate_subproblem_size(bqm_num_vars)

        if not benchmark:
            print("Calcolo embedding globale per il BQM in corso...")
            t_global_embed_0  = time.perf_counter() #inizio a contare tempo da calcolo dell'embedding globale
            if verbose:
                print(f"num_var: {bqm_num_vars}, density: {bqm_density}, avg_degree: {bqm_avg_degree}")
            if bqm_num_vars<max_global_size and is_qpu_ready:
                global_embedding, is_all_embeddable, _ = mybqm.compute_embedding(bqm, self.target_graph)
            else:
                global_embedding = {}
                is_all_embeddable = False

            t_embedding = time.perf_counter() - t_global_embed_0
            print(f"Embedding globale completato in {t_embedding}s. Embeddabile interamente: {is_all_embeddable}\n")
        else:
            global_embedding = global_embedding_bench
            is_all_embeddable = is_all_embeddable_bench
            t_embedding = embedding_time

        #inizializzazione dei Dispatcher Redis per le rispettive code
        qpu_dispatcher = AsyncRedisDispatcherSampler(target_queue="qpu_queue", connection_pool=self.redis_pool, meta= meta)
        cpu_dispatcher = AsyncRedisDispatcherSampler(target_queue="cpu_queue", connection_pool=self.redis_pool)

        chain_lengths_global = [len(chain) for chain in global_embedding.values()] if global_embedding else []
        max_chain_global = max(chain_lengths_global) if chain_lengths_global else float('inf')

        physical_nodes = sum(chain_lengths_global)

        expansion_global = physical_nodes / bqm_num_vars if bqm_num_vars > 0 else 0.0

        router = RouterSampler(
            mode=self.policy_mode,
            global_embedding=global_embedding,
            is_all_embeddable=is_all_embeddable,
            qpu_dispatcher=qpu_dispatcher,
            cpu_dispatcher=cpu_dispatcher,
            target_graph=self.target_graph,
            max_chain_length=max_chain_length,
            verbose=verbose
        )

        #esecuzione diretta senza decomposizione se il problema globale è interamente embeddabile e rispetta condizioni della policy
        should_run_direct = (
            global_embedding and
            is_all_embeddable and
            is_qpu_ready and
            max_chain_global <= max_chain_length and
            expansion_global <= 5.0
            )

        
        if should_run_direct:
            print("Risoluzione diretta")

            if meta is not None and chain_strength_prefactor is not None:
                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=chain_strength_prefactor
                )

            else:
                chain_strength = uniform_torque_compensation(
                    bqm, embedding=global_embedding, prefactor=1.414
                )


            t_build_0 = time.perf_counter()
            target_bqm = embed_bqm(
                bqm, 
                embedding=global_embedding, 
                target_adjacency=self.target_graph,
                chain_strength=chain_strength
            )
            t_build = time.perf_counter() - t_build_0
            
            t_start_solve = time.perf_counter()
            init_state = hybrid.State.from_problem(target_bqm)

            #istanzia ed esegui il sampler
            sa_workflow = hybrid.SimulatedAnnealingProblemSampler(num_reads=50, num_sweeps=1000)
            end_state = sa_workflow.run(init_state).result()

            physical_sampleset = end_state.samples

            t_solve_total = time.perf_counter() - t_start_solve

            break_freq_dict = chain_break_frequency(physical_sampleset, global_embedding) # Returns: dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s is a variable in the source graph, and frequency, a float, is the fraction of broken chains.

            chain_break_fraction = 0.0  
            if break_freq_dict:
                #calcola la media dei valori nel dizionario
                chain_break_fraction = sum(break_freq_dict.values()) / len(break_freq_dict)


            t_unembed_0 = time.perf_counter()
            logical_sampleset = unembed_sampleset(physical_sampleset, global_embedding, bqm, chain_break_method=majority_vote)
            t_unembed = time.perf_counter() - t_unembed_0
            
            total_wall_clock = t_solve_total + t_embedding + t_build + t_unembed

            extra_metrics = {
                "physical_build_time": t_build,
                "unembed_time": t_unembed,
                "chain_break_fraction": chain_break_fraction,
            }

            best_energy = float(logical_sampleset.first.energy)

            if is_maximize:
                best_energy = -best_energy


            #gap rispetto a QPLIB se fornito
            rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
            abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None


            best_sample = dict(logical_sampleset.first.sample)

            good_embedding = SubQUBO.compute_advanced_embedding_metrics(global_embedding,bqm,chain_break_fraction)

            cqmResults = {}

            if len(cqm.constraints)>0:
                cqmResults = CqmUtilities.evaluate_cqm_feasibility(best_sample,invert_func, cqm)

            print("Risoluzione diretta terminata.")

            results = {
                #Routing e Policy
                "instance_id": lp_filename,
                "is_maximization_objective": is_maximize,
                "policy": self.policy_mode,
                "qpu_calls": 1,
                "cpu_calls": 0,
                "qpu_utilization_ratio": 1.0,
                "time_routing_total": 0.0,
                "tp": 1 if good_embedding else 0, 
                "fp": 1 if not good_embedding else 0, "tn": 0, "fn": 0,
                #Soluzione e Gap
                "best_energy": best_energy,
                "qplib_reference": bestknown_solution,
                "abs_gap": abs_gap,
                "rel_gap": rel_gap,
                #Proxy Physical Metrics
                "solve_time": t_solve_total,
                #End-to-End Breakdown
                "wall_clock_time": total_wall_clock,
                "time_global_embedding": t_embedding,
                "time_total_embedding": t_embedding,
                "decomposed": False,

                **extra_metrics,
                **cqmResults,
            }

            if verbose:
                print(results)

            return results

        #pipeline per decomposizione
        def merge_substates(_, substates, **kwargs):
            a, b = substates
            return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))


        decomposer = hybrid.Unwind(
            hybrid.EnergyImpactDecomposer(
                size=min(effective_sub_size, len(bqm.variables)), 
                rolling=True, 
                rolling_history=0.3, 
                traversal="bfs"
            )
        )

        subproblem_pipeline = (
            decomposer
            # hybrid.Const(subsamples=None) #in questo caso non serve poiché ricostruisco State con i subsamples nel worker dopo aver serializzato su coda Redis
            | hybrid.Map(router)
            | hybrid.Reduce(hybrid.Lambda(merge_substates))
            | hybrid.SplatComposer()
        )

        #ParallelBranches runs multiple workflows of type Runnable in parallel, blocking until all finish.
        #consente di evitare errori di cancellazione anticipata sollevati da hybrid.Race senza compromettere velocità
        iteration = hybrid.ParallelBranches(
            hybrid.SimulatedAnnealingProblemSampler(num_reads=1, num_sweeps=10000),
            hybrid.TabuProblemSampler(),
            subproblem_pipeline
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True)


        main = hybrid.Loop(iteration, max_iter=self.max_iter, convergence=self.convergence)

        t_start_solve = time.perf_counter()
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)

        final_state = main.run(init_state).result()

        t_solve_total = time.perf_counter() - t_start_solve
        
        total_wall_clock = t_solve_total + t_embedding

        total_sub_embedding_time = router.total_sub_embedding_time

        time_total_embedding = t_embedding + total_sub_embedding_time

        best_energy = float(final_state.samples.first.energy)

        if is_maximize:
            best_energy = -best_energy
        
        rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
        abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

        avg_chain_break = float(np.mean(qpu_dispatcher.chain_break_fractions)) if qpu_dispatcher.chain_break_fractions else 0.0
        total_calls = router.qpu_calls + router.cpu_calls

        
        best_sample = dict(final_state.samples.first.sample)

        cqmResults = {}

        if len(cqm.constraints)>0:
            cqmResults = CqmUtilities.evaluate_cqm_feasibility(best_sample,invert_func, cqm)

        total_tp = cpu_dispatcher.tp + qpu_dispatcher.tp
        total_fp = cpu_dispatcher.fp + qpu_dispatcher.fp
        total_tn = cpu_dispatcher.tn + qpu_dispatcher.tn
        total_fn = cpu_dispatcher.fn + qpu_dispatcher.fn

        all_eqi = qpu_dispatcher.all_eqi + cpu_dispatcher.all_eqi

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

        results = {
            # 1. Area Routing & Policy
            "instance_id": lp_filename,
            "is_maximization_objective": is_maximize,
            "policy": self.policy_mode,
            "qpu_calls": router.qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": router.qpu_calls / total_calls if total_calls > 0 else 0.0,
            "tp": total_tp, "fp": total_fp, "tn": total_tn, "fn": total_fn,
            
            # 2. Area Soluzione
            "best_energy": best_energy,
            "qplib_reference": bestknown_solution,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            
            # 3. Area Proxy Metrics
            "chain_break_fraction": avg_chain_break,
            "physical_build_time": qpu_dispatcher.total_build_time,
            "solve_time": qpu_dispatcher.total_solve_time,
            "unembed_time": qpu_dispatcher.total_unembed_time,
            
            # 4. Area End-to-End Breakdown
            "wall_clock_time": total_wall_clock,
            "time_global_embedding": t_embedding,
            "time_subqubo_embedding": total_sub_embedding_time,
            "time_total_embedding": time_total_embedding,
            "time_routing_total": router.total_routing_time,
            "decomposed": True,

            #"prefactor": meta.get("chain_strength_prefactor", None),  ##############################

            **cqmResults,
            **eqi_stats
        }
        
        if verbose:
            print("\n" + "="*110)
            print(f"{'POLICY':<22} | {'ENERGY':<12} | {'QPLIB REF':<12} | {'WALL TIME (s)':<13} | {'QPU CALLS':<10} | {'CPU CALLS':<10} | {'QPU %':<6}")
            print("="*101)

            energy_str = f"{results['best_energy']:<12.4f}"
            qplib_ref = f"{bestknown_solution:<12.4f}"
            wall_str = f"{results['wall_clock_time']:<13.4f}"

            if math.isnan(results["qpu_calls"]):
                qpu_calls_str = "-"
                cpu_calls_str = "-"
                qpu_ratio_str = "-"
            else:
                qpu_calls_str = f"{int(results['qpu_calls']):<10}"
                cpu_calls_str = f"{int(results['cpu_calls']):<10}"
                qpu_ratio_str = f"{results['qpu_utilization_ratio']*100:<5.1f}%"

            print(f"{results['policy']:<22} | {energy_str:<12} | {qplib_ref:<12} | {wall_str:<13} | {qpu_calls_str:<10} | {cpu_calls_str:<10} | {qpu_ratio_str:<6}")

            print("="*110)
        
        print(f"Risoluzione terminata per {lp_filename}")

        return results

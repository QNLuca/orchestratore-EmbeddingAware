import io
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

import urllib

import dimod
import dwave.graphs as dnx
import hybrid
from dwave.embedding.zephyr import find_clique_embedding
from minorminer import find_embedding

import redis

from subQUBO import SubQUBO

from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

#config Redis
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

from concurrent.futures import ThreadPoolExecutor
import uuid
import json
import time
import redis
import dimod
import hybrid

class AsyncRedisDispatcherSampler(hybrid.Runnable):
    """
    Dispatcher asiincrono, invia il task a Redis e restituisce un Future.
    Consente a hybrid.Map di inviare N sub-QUBO in parallelo ai worker.
    """
    def __init__(self, target_queue: str, redis_client: redis.Redis, timeout: int = 30, max_workers: int = 8, **runopts):
        super().__init__(**runopts)
        self.target_queue = target_queue
        self.r = redis_client
        self.timeout = timeout
        #thread pool per gestire l'attesa io pub/sub in parallelo
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _submit_and_wait(self, sub_bqm, job_id, task_id):

        payload = {
            "job_id": job_id,
            "task_id": task_id,
            "bqm": sub_bqm.to_serializable()
        }

        #BQM non è serializzabile con metodo consigliato dimod...
        #payload_json = json.dumps(payload, cls=DimodEncoder)

        payload_json = json.dumps(payload)

        response_channel = f"results:{job_id}"
        pubsub = self.r.pubsub()
        pubsub.subscribe(response_channel)

        #spingo il task su Redis
        self.r.rpush(self.target_queue, payload_json)

        #in attesa del risultato da pub/sub
        start_time = time.time()
        subsample = None

        while (time.time() - start_time) < self.timeout:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                data = json.loads(message['data'])
                if data.get("task_id") == task_id:
                    subsample = dimod.SampleSet.from_serializable(data["sampleset"])
                    break

        pubsub.unsubscribe(response_channel)

        if subsample is None:
            raise TimeoutError(f"Timeout Redis su '{self.target_queue}' per task {task_id}")

        return subsample

    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        #if sub_bqm is None or len(sub_bqm.variables) == 0:
        #    return state

        job_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        #delega l'attesa pub/sub al ThreadPool senza bloccare il flusso principale
        future = self.pool.submit(self._submit_and_wait, sub_bqm, job_id, task_id)
        
        #risolve il future quando lo stato viene richiesto
        subsample = future.result()
        return state.updated(subsamples=subsample)


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
        max_chain_length: int = 8, 
        max_size: int = 40, 
        max_density: float = 0.5, 
        max_average_degree: int = 15,
        verbose: bool = False,
        **runopts
    ):
        super().__init__(**runopts)
        self.mode = mode
        self.target_graph = target_graph
        self.max_chain_length = max_chain_length
        self.max_size = max_size
        self.max_density = max_density
        self.max_average_degree = max_average_degree
        self.verbose = verbose

        self.qpu_dispatcher = qpu_dispatcher
        self.cpu_dispatcher = cpu_dispatcher
        
        self.global_embedding = global_embedding
        self.is_all_embeddable = is_all_embeddable

        self.meta = {}
        
        self.qpu_calls = 0
        self.cpu_calls = 0
        self.qpu_time = 0.0
        self.cpu_time = 0.0

    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        sub_qubo = SubQUBO(bqm=sub_bqm)
        num_vars = len(sub_bqm.variables)
        avg_degree = sub_qubo.average_degree
        density = sub_qubo.density

        embedding = None
        max_chain = float('inf')
        is_embeddable = False

        try:
            embedding = find_clique_embedding(
                        k=num_vars,
                        target_graph=self.target_graph
            )
        except Exception:
            try:
                global_edges = list(sub_bqm.quadratic.keys()) or [(v, v) for v in sub_bqm.variables]
                if num_vars < 231: #numero massimo per rappresentazione grafo zephyr
                    embedding = find_embedding(
                        S=global_edges,
                        T=self.target_graph.edges,
                        timeout=100,
                        threads=4,
                        verbose=0
                    )
                else:
                    embedding = {}
            except Exception:
                    embedding = {}

        chain_lengths = [len(chain) for chain in embedding.values()] if embedding else []
        is_embeddable = (len(embedding) == num_vars) and (num_vars > 0)
        max_chain = max(chain_lengths) if chain_lengths else float('inf')

        route_to_qpu = is_embeddable and (max_chain <= self.max_chain_length) and num_vars<self.max_size and density<self.max_density and avg_degree<self.max_average_degree

        target_device = "QPU" if route_to_qpu else "CPU"

        if self.verbose:
            print(
                f"DEBUG SUBPROBLEM, N={num_vars} | "
                f"Density={density:.3f} | "
                f"AvgDeg={avg_degree} | "
                f"Embeddable={str(is_embeddable)} | "
                f"MaxChain={max_chain if max_chain != float('inf') else 'N/A'} | "
                f"--> Route: {target_device}"
            )

        t0 = time.perf_counter()
        if route_to_qpu:
            self.qpu_calls += 1
            qpu_state = state.updated(embedding=embedding) if embedding else state
            res_state = self.qpu_dispatcher.run(qpu_state).result()
            self.qpu_time += (time.perf_counter() - t0)
        else:
            self.cpu_calls += 1
            res_state = self.cpu_dispatcher.run(state).result()
            self.cpu_time += (time.perf_counter() - t0)

        return state.updated(subsamples=res_state.subsamples)


class EmbeddingAwareOrchestrator:
    """Orchestratore per la risoluzione ottimizzata di problemi QUBO/BQM."""
    def __init__(
        self, 
        policy_mode: str = 'embedding_aware', 
        target_graph=None,
        subproblem_size: Optional[int] = None,
        max_iter: int = 10,
        convergence: int = 3,
        verbose: bool = False
    ):
        self.policy_mode = policy_mode
        self.target_graph = target_graph or dnx.zephyr_graph(15)
        self.subproblem_size = subproblem_size
        self.max_iter = max_iter
        self.convergence = convergence
        self.verbose = verbose

        self.meta = {}

        #connessione a Redis
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

    def _calculate_subproblem_size(self, bqm_len: int) -> int:
        if self.subproblem_size is not None:
            return min(self.subproblem_size, bqm_len)
        raw_size = int(math.ceil(2.0 * math.sqrt(bqm_len)))
        return min(max(10, min(raw_size, 400)), bqm_len)

    def load_qubo_lp(self,
    url: str,
    lagrange_multiplier: float = 10.0, #per convertire i vincoli di un problema matematico in penalità energetiche
    auto_scale_penalty: bool = True,
) -> Tuple[dimod.BinaryQuadraticModel, List[Tuple[Dict[str, float], float, str]], Dict]:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            content = response.read()

        #rileva se la funzione obiettivo richiede di massimizzare o minimizzare
        text = content.decode("utf-8", errors="ignore").lower()
        is_maximize = False
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("\\") or line.startswith("/"):
                continue
            if line.startswith("maximize") or line.startswith("max"):
                is_maximize = True
                break
            if line.startswith("minimize") or line.startswith("min"):
                is_maximize = False
                break

        cqm = dimod.lp.load(io.BytesIO(content))

        meta = {
            "url": url,
            "n_variables_original": len(cqm.variables),
            "n_constraints": len(cqm.constraints),
            "equalities": 0,
            "inequalities": 0,
            "is_maximize": is_maximize,
            "warnings": [],
        }

        self.meta = meta

        obj = cqm.objective
        objective_bqm = dimod.BinaryQuadraticModel(
            {v: float(bias) for v, bias in obj.linear.items()},
            {(u, v): float(bias) for (u, v), bias in obj.quadratic.items()},
            float(obj.offset),
            vartype=dimod.BINARY,
        )

        if auto_scale_penalty and (objective_bqm.linear or objective_bqm.quadratic):
            max_coeff = max(
                [abs(b) for b in objective_bqm.linear.values()] +
                [abs(b) for b in objective_bqm.quadratic.values()] +
                [1.0]
            )
            base_P = max(lagrange_multiplier, 5.0 * max_coeff)
        else:
            base_P = float(lagrange_multiplier)

        meta["lagrange_multiplier"] = base_P

        def add_equality_penalty(bqm, linear_coeffs, target, P):
            bqm.offset += P * (target ** 2)
            vars_list = list(linear_coeffs.keys())
            for i, u in enumerate(vars_list):
                a_u = linear_coeffs[u]
                bqm.add_linear(u, P * (a_u**2 - 2.0 * target * a_u))
                for v in vars_list[i+1:]:
                    a_v = linear_coeffs[v]
                    bqm.add_quadratic(u, v, P * 2.0 * a_u * a_v)

        inequality_constraints = []

        for cname, constraint in cqm.constraints.items():
            lhs = constraint.lhs
            rhs = float(constraint.rhs)
            sense = constraint.sense

            if lhs.quadratic:
                meta["warnings"].append(
                    f"Vincolo '{cname}' ha termini quadratici: considerata solo la parte lineare."
                )

            linear_coeffs = {v: float(b) for v, b in lhs.linear.items()}
            if not linear_coeffs:
                continue

            if sense == "==":
                add_equality_penalty(objective_bqm, linear_coeffs, rhs, base_P)
                meta["equalities"] += 1
            else:
                inequality_constraints.append((linear_coeffs, rhs, sense))
                meta["inequalities"] += 1

        meta["n_variables_objective"] = len(objective_bqm.variables)

        return objective_bqm, inequality_constraints, meta

    def solve(self, bqm: Union[dimod.BinaryQuadraticModel]) -> Dict[str, Any]:
        """
        Risolve un BQM applicando il routing dinamico basato su embedding e topologia.
        """
        mybqm = SubQUBO(bqm)

        max_chain_length=8
        max_size=231 #massima clique completa rappresentabile sulla QPU Zephyr modellata per il test
        max_density=0.5
        max_avg_degree=15
        num_var = mybqm.num_variables

        effective_sub_size = self._calculate_subproblem_size(num_var)

        if mybqm.num_variables<231 and mybqm.density<=max_density and mybqm.average_degree<=max_avg_degree: #limiti strutturali del solver in esame per test + features selezionate
            global_embedding, is_all_embeddable = mybqm.compute_embedding(bqm, self.target_graph)
        else:
            global_embedding = {}
            is_all_embeddable = False

        #inizializzazione dei Dispatcher Redis per le rispettive code
        qpu_dispatcher = AsyncRedisDispatcherSampler(target_queue="queue:QPU", redis_client=self.redis_client)
        cpu_dispatcher = AsyncRedisDispatcherSampler(target_queue="queue:CPU", redis_client=self.redis_client)

        qpu_target_global = hybrid.InterruptableSimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)


        chain_lengths_global = [len(chain) for chain in global_embedding.values()] if global_embedding else []
        max_chain_global = max(chain_lengths_global) if chain_lengths_global else float('inf')

        router = RouterSampler(
            mode=self.policy_mode,
            global_bqm=bqm,
            global_embedding=global_embedding,
            is_all_embeddable=is_all_embeddable,
            qpu_dispatcher=qpu_dispatcher,
            cpu_dispatcher=cpu_dispatcher,
            target_graph=self.target_graph,
            max_chain_length=max_chain_length,
            max_size=max_size, #massima clique completa rappresentabile sulla QPU Zephyr modellata per il test
            max_density=max_density,
            max_average_degree=max_avg_degree,
            verbose=self.verbose
        )

        #esecuzione diretta senza decomposizione se il problema globale è interamente embeddabile e rispetta condizioni della policy
        should_run_direct = router.is_all_embeddable and (max_chain_global <= max_chain_length) and (mybqm.num_variables <= max_size) and (mybqm.density <= max_density) and mybqm.average_degree <= max_avg_degree 

        if should_run_direct:
            start_time = time.perf_counter()
            init_state = hybrid.State.from_sample(
                dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample,
                #oppure anche una soluzione random se si vuole velocizzare pre-esecuzione
                bqm
            ).updated(embedding=router.global_embedding)

            res_state = qpu_target_global.run(init_state).result()
            total_wall_clock = time.perf_counter() - start_time
            sampleset = res_state.subsamples if res_state.subsamples is not None else res_state.samples

            return {
                "policy": self.policy_mode,
                #"best_sample": dict(sampleset.first.sample),
                "best_energy": float(sampleset.first.energy),
                "wall_clock_time": total_wall_clock,
                "qpu_calls": 1,
                "cpu_calls": 0,
                "qpu_time": total_wall_clock,
                "cpu_time": 0.0,
                "decomposed": False,
                "qpu_utilization_ratio": 1.0,
                "iterations": 1
            }

        #pipeline per decomposizione
        def merge_substates(_, substates, **kwargs):
            a, b = substates
            return a.updated(subsamples=hybrid.hstack_samplesets(a.subsamples, b.subsamples))

        subproblems = hybrid.Unwind(
            hybrid.EnergyImpactDecomposer(
                size=min(effective_sub_size, len(bqm.variables)), 
                rolling=True, 
                rolling_history=1, 
                traversal="bfs"
            )
        )

        subproblem_pipeline = (
            subproblems
            #| hybrid.Const(subsamples=None) #aggiunta per risolvere mismatch bqm e initial_state
            | hybrid.Map(router)
            | hybrid.Reduce(hybrid.Lambda(merge_substates))
            | hybrid.SplatComposer()
        )

        iteration = hybrid.Race(
            hybrid.InterruptableTabuSampler(timeout=200),
            subproblem_pipeline
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True)

        main = hybrid.Loop(iteration, max_iter=self.max_iter, convergence=self.convergence)

        start_time = time.perf_counter()
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)

        final_state = main.run(init_state).result()
        total_wall_clock = time.perf_counter() - start_time

        
        best_sample = dict(final_state.samples.first.sample)
        if self.meta.get("is_maximize", False):
            best_energy = -float(final_state.samples.first.energy)
        else:
            best_energy = float(final_state.samples.first.energy)
        total_iterations = final_state.get('loop_iter', self.max_iter)

        return {
            "policy": self.policy_mode,
            #"best_sample": best_sample, #rimuovo per leggibilità soluzione
            "best_energy": best_energy,
            "wall_clock_time": total_wall_clock,
            "qpu_calls": router.qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_time": router.qpu_time,
            "cpu_time": router.cpu_time,
            "qpu_utilization_ratio": router.qpu_calls / (router.qpu_calls + router.cpu_calls) if (router.qpu_calls + router.cpu_calls) > 0 else 0.0,
            "decomposed": True,
            "iterations": total_iterations
        }
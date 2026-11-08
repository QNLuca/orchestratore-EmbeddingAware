from collections import deque
import json
import math
import time
from typing import Dict, Optional
from urllib.parse import urlparse
import dimod
from dotenv import load_dotenv
import dwave.graphs as dnx
import datetime as dt
import random

from dwave.system import DWaveSampler

from orchestrator import EmbeddingAwareOrchestrator
from subQUBO import SubQUBO
from policies.alwaysCPU import AlwaysCPUPolicy
from policies.alwaysQPU import AlwaysQPUEmbeddablePolicy
from policies.kerberos import KerberosPolicy
from policies.qbsolv import QBSolvPolicy
from policies.mqt import MQTQAOPolicy
from policies.onlyFeaturesBased import OnlyFeaturesBasedPolicy
from policies.onlyEABased import OnlyEABasedPolicy

import os
from pathlib import Path

import pandas as pd
from loadLp import LoadLp

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_KEY = os.getenv("DWAVE_API_TOKEN")

#coda globale che definisce l'ordine di rotazione delle policy
POLICY_DEQUE = deque([
    "embedding_aware",
    "always_cpu",
    "kerberos",
    "qbsolv",
    "mqt_qao",
    "always_qpu_embeddable",
    "only_features_based",
    "only_ea_based"
])

#funzione factory per generare correttamente istanze per ogni policy
def get_policy_instance(name: str, subproblem_size: int, max_iter: int, convergence: int, embedding_time: float):
    if name == "always_cpu":
        return AlwaysCPUPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time)
    elif name == "kerberos":
        return KerberosPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence)
    elif name == "qbsolv":
        return QBSolvPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence)
    elif name == "mqt_qao":
        return MQTQAOPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time)
    elif name == "always_qpu_embeddable":
        return AlwaysQPUEmbeddablePolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time)
    elif name == "only_features_based":
        return OnlyFeaturesBasedPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time)
    elif name == "only_ea_based":
        return OnlyEABasedPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time)
    else:
        raise ValueError(f"Policy sconosciuta: {name}")

def run_comparative_suite(bqm: dimod.BinaryQuadraticModel, orchestratorEA: EmbeddingAwareOrchestrator, meta: Optional[Dict], seed=None, 
                          qpu_identity = None, target_graph = None, max_iter: Optional[int] = 5, convergence: Optional[int] = 2):

    mybqm = SubQUBO(bqm)

    if seed is None:
        seed = random.randint(0,10)

    if DWAVE_API_KEY:
        sampler = DWaveSampler(topology__type="zephyr")
        qpu_identity = sampler.solver.identity
        target_graph = sampler.to_networkx_graph()
    else:
        target_graph = dnx.zephyr_graph(12)
        qpu_identity = None
        

    verbose = meta.get('verbose', False)

    max_global_size = 1337

    bqm_num_vars = mybqm.num_variables
    bqm_avg_degree = mybqm.average_degree
    bqm_density = mybqm.density

    is_qpu_ready = not ((bqm_avg_degree > 52 and bqm_num_vars > 192 and bqm_density > 0.052) 
                                  or (bqm_num_vars > 250 and bqm_density >= 0.052))

    print("Calcolo embedding globale per il BQM in corso...")
    start_time_emb = time.perf_counter()
    if mybqm.num_variables < max_global_size and is_qpu_ready:
        global_embedding, is_all_embeddable, _ = mybqm.compute_embedding(bqm, target_graph, seed)
    else:
        global_embedding = {}
        is_all_embeddable = False
    embedding_time = time.perf_counter() - start_time_emb
    print(f"Embedding globale completato in {embedding_time}s. Embeddabile interamente: {is_all_embeddable}\n")

    #size del sottoproblema cresce in maniera controllata e variabile-dipendente
    subproblem_size = int(math.ceil(2.0 * math.sqrt(len(bqm.variables))))

    print(f"AVVIO BENCHMARK COMPARATIVO ROUTING POLICY (BQM N={len(bqm.variables)})")

    results = []

    #itera sulla coda ruotata e istanzia dinamicamente la policy corrispondente
    for policy_name in POLICY_DEQUE:
        print(f"Valutazione Policy: [{policy_name}] ...")
        
        #istanziazione dinamica basata sul nome corrente nella deque
        if policy_name == "embedding_aware":
            res = orchestratorEA.solve(
                bqm, meta, 
                global_embedding_bench=global_embedding,
                is_all_embeddable_bench=is_all_embeddable,
                benchmark=True, 
                embedding_time=embedding_time, 
                seed=seed, 
                target_graph=target_graph, 
                qpu_identity=qpu_identity
            )
        else:
            #mappatura per istanziare la policy corretta con i parametri previsti
            policy_obj = get_policy_instance(
                policy_name, 
                subproblem_size=subproblem_size, 
                max_iter=max_iter, 
                convergence=convergence, 
                embedding_time=embedding_time
            )
            
            res = policy_obj.solve(
                meta=meta,
                bqm=bqm,
                global_embedding=global_embedding,
                is_all_embeddable=is_all_embeddable,
                target_graph=target_graph,
                seed=seed,
                qpu_identity=qpu_identity
            )
            
        results.append(res)

    #rotazione policy, avanti deterministicamente di 1
    POLICY_DEQUE.rotate(1)

    print(f"BENCHMARK COMPARATIVO COMPLETATO (BQM N={len(bqm.variables)})")

    if verbose:

        print("\n" + "="*110)
        print(f"{'POLICY':<22} | {'ENERGY':<12} | {'QPLIB REF':<12} | {'WALL TIME (s)':<13} | {'QPU CALLS':<10} | {'CPU CALLS':<10} | {'QPU %':<6}")
        print("="*110)

        for r in results:

            energy_str = f"{r['best_energy']:<12.4f}"
            qplib_ref = f"{r['qplib_reference']:<12.4f}"
            wall_str = f"{r['wall_clock_time']:<13.4f}"

            if math.isnan(r["qpu_calls"]):
                qpu_calls_str = "-"
                cpu_calls_str = "-"
                qpu_ratio_str = "-"
            else:
                qpu_calls_str = f"{int(r['qpu_calls']):<10}"
                cpu_calls_str = f"{int(r['cpu_calls']):<10}"
                qpu_ratio_str = f"{r['qpu_utilization_ratio']*100:<5.1f}%"

            print(f"{r['policy']:<22} | {energy_str:<12} | {qplib_ref:<12} | {wall_str:<13} | {qpu_calls_str:<10} | {cpu_calls_str:<10} | {qpu_ratio_str:<6}")

            print("="*110)

    return results

############# TEST FILE LOCALI ##################
def load_existing_results(parquet_path):
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()

def extract_instance_number(fpath):
    uri = str(fpath)
    parsed_url = urlparse(uri)
    path = parsed_url.path if parsed_url.scheme else uri
    
    base_name = os.path.basename(path).split(".")[0]
    lp_filename = base_name.split("_")[1] if "_" in base_name else base_name

    return lp_filename

def execute_files_benchmark(orchestratorEA: EmbeddingAwareOrchestrator, status_dict: Optional[dict] = None):
    lp_dir = os.path.join("test", "lp")

    if not os.path.exists(lp_dir):
        msg = f"Errore: La cartella '{lp_dir}' non esiste"
        print(msg)
        return {"status": "error", "message": msg}

    lp_files = list(Path(lp_dir).rglob("*.[lL][pP]"))
    
    if not lp_files:
        msg = f"Nessun file .lp trovato in '{lp_dir}'"
        print(msg)
        return {"status": "error", "message": msg}

    print(f"Trovati {len(lp_files)} file .lp in: {lp_dir}")

    results_dir = os.path.join("test", "results")
    os.makedirs(results_dir, exist_ok=True) #assicura che la cartella esista

    code_version = status_dict["code_version"]
    base_name = f"orchestrator_test_results.{code_version}"
    suffix = ".parquet"

    orchestrators_results = os.path.join(results_dir, f"{base_name}{suffix}")

    json_metadata_path = orchestrators_results.replace(".parquet", ".json")

    #resume automatico
    df_runs_existing = load_existing_results(orchestrators_results)

    already_done = (
        set(df_runs_existing["instance_id"].astype(str))
        if not df_runs_existing.empty
        and "instance_id" in df_runs_existing.columns
        else set()
    )

    todo_files = [
        f for f in lp_files if extract_instance_number(f) not in already_done
    ]

    print(f"Già processati: {len(already_done)}")
    print(f"Da processare: {len(todo_files)}")

    if status_dict is not None:
        status_dict["total"] = len(lp_files)
        status_dict["completed"] = len(already_done)

    max_iter = 10
    convergence = 3

    #in caso di ripresa dei test, recupero seed da file json già esistente
    if os.path.exists(json_metadata_path):
        with open(json_metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            seed = metadata.get("minorminer_seed")
        print(f"Ripristinato seed precedente dal file JSON: {seed}")
    else:
        seed = random.randint(0, 10)
        experiment_metadata = {
            "code_version": code_version,
            "timestamp_start": dt.datetime.now().isoformat(),
            "parameters": {
                "max_iter": max_iter,
                "convergence": convergence,
                "num_reads": 40,
            },
            "minorminer_seed": seed,
            "qplib_instances": [str(f.name) for f in lp_files],
            "hardware_backend": f"DWaveSampler: {qpu_identity}" if DWAVE_API_KEY else "None: Simulated Annealing"
        }
        with open(json_metadata_path, "w", encoding="utf-8") as f:
            json.dump(experiment_metadata, f, indent=4)
        print(f"Creato file di metadati JSON con nuovo seed ({seed}): {json_metadata_path}")

    target_graph=dnx.zephyr_graph(12)
    qpu_identity=None 

    #Advantage2_system1, graph_id: 01b03fc8dc
    #qpu_identity = {
    #"name": "Advantage2_system1",
    #"version": {"graph_id": "01b03fc8dc"},
    #}

    if DWAVE_API_KEY:
        if qpu_identity:
            sampler = DWaveSampler(solver=dict(identity=qpu_identity))
            target_graph = sampler.to_networkx_graph()
        else:
            sampler = DWaveSampler(topology__type="zephyr")
            qpu_identity = sampler.solver.identity
            target_graph = sampler.to_networkx_graph()

    for idx, filepath in enumerate(todo_files, 1):
        fname = os.path.basename(filepath)
        print(f"\n[{idx}/{len(todo_files)}] embedding su {fname}...")
        t0 = time.perf_counter()

        filepath_str = str(filepath)

        bqm, meta = LoadLp.load_qubo_lp(filepath_str, file = True)

        single_runs = run_comparative_suite(
            bqm=bqm, orchestratorEA=orchestratorEA, meta=meta, max_iter=max_iter, 
            convergence=convergence, qpu_identity=qpu_identity, target_graph=target_graph, seed=seed
        )

        t1 = time.perf_counter()
        print(f"lp risolto eseguita in {t1 - t0:.2f}s")

        #salva in modo incrementale sul parquet unico delle run
        df_runs_existing = pd.concat(
            [df_runs_existing, pd.DataFrame(single_runs)], ignore_index=True
        )
        df_runs_existing.to_parquet(
            orchestrators_results, engine="pyarrow", index=False
        )

        if status_dict is not None:
            status_dict["completed"] += 1

        print(f"Salvato aggiornamento per {fname}")


    print("\n=== ESECUZIONE COMPLETATA ===")
    return {"status": "completed", "processed_files": len(todo_files)}
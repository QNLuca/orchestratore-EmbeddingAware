import math
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse
import dimod
import dwave.graphs as dnx

from orchestrator import EmbeddingAwareOrchestrator
from subQUBO import SubQUBO
from policies.basePolicy import BasePolicy
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

def run_comparative_suite(bqm: dimod.BinaryQuadraticModel, orchestratorEA: EmbeddingAwareOrchestrator, meta: Optional[Dict], target_graph=None, max_iter: Optional[int] = 5, convergence: Optional[int] = 2):

    mybqm = SubQUBO(bqm)

    if target_graph is None:
        target_graph = dnx.zephyr_graph(12)

    verbose = meta.get('verbose', False)

    max_global_size= 1337

    bqm_num_vars = mybqm.num_variables
    bqm_avg_degree = mybqm.average_degree
    bqm_density = mybqm.density

    is_qpu_ready = not ((bqm_avg_degree > 52 and bqm_num_vars > 192 and bqm_density > 0.052) 
                                  or (bqm_num_vars > 250 and bqm_density >= 0.052))

    print("Calcolo embedding globale per il BQM in corso...")
    start_time_emb = time.perf_counter()
    if mybqm.num_variables < max_global_size and is_qpu_ready:
        global_embedding, is_all_embeddable, _ = mybqm.compute_embedding(bqm, target_graph)
    else:
        global_embedding = {}
        is_all_embeddable = False
    embedding_time = time.perf_counter() - start_time_emb
    print(f"Embedding globale completato in {embedding_time}s. Embeddabile interamente: {is_all_embeddable}\n")

    #size del sottoproblema cresce in maniera controllata e variabile-dipendente
    subproblem_size = int(math.ceil(2.0 * math.sqrt(len(bqm.variables))))

    #istanzio policy
    pipeline_policies: List[BasePolicy] = [
        AlwaysCPUPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time),
        KerberosPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
        QBSolvPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
        MQTQAOPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time),
        AlwaysQPUEmbeddablePolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time),
        OnlyFeaturesBasedPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence, embedding_time=embedding_time),
        OnlyEABasedPolicy(subproblem_size=subproblem_size,max_iter=max_iter, convergence=convergence, embedding_time=embedding_time)
    ]

    print(f"AVVIO BENCHMARK COMPARATIVO ROUTING POLICY (BQM N={len(bqm.variables)})")

    results = []

    resultMyPolicy = orchestratorEA.solve(bqm, meta, global_embedding_bench=global_embedding,is_all_embeddable_bench=is_all_embeddable,benchmark=True, embedding_time=embedding_time)
    
    results.append(resultMyPolicy)


    for policy in pipeline_policies:
        print(f"Valutazione Policy: [{policy.name}] ...")
        res = policy.solve(
            meta=meta,
            bqm=bqm,
            global_embedding=global_embedding,
            is_all_embeddable=is_all_embeddable,
            target_graph=target_graph
        )

        results.append(res)

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

    i = 1
    while os.path.exists(orchestrators_results):
        orchestrators_results = os.path.join(results_dir, f"{base_name}.{i}{suffix}")
        i += 1

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

    for idx, filepath in enumerate(todo_files, 1):
        fname = os.path.basename(filepath)
        print(f"\n[{idx}/{len(todo_files)}] embedding su {fname}...")
        t0 = time.perf_counter()

        filepath_str = str(filepath)

        bqm, meta = LoadLp.load_qubo_lp(filepath_str, file = True)

        single_runs = run_comparative_suite(
            bqm=bqm, orchestratorEA=orchestratorEA, meta=meta, max_iter=max_iter, 
            convergence=convergence
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
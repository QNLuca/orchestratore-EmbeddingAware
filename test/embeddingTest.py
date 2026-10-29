from pathlib import Path
import os
import sys
import time
import dwave.graphs as dnx
from minorminer import find_embedding
import numpy as np
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from loadLp import LoadLp
from subQUBO import SubQUBO


def run_embedding_test_for_file(filepath, target_graph, num_runs=10):
    bqm, _ = LoadLp.load_qubo_lp(filepath, file = True)
    mybqm = SubQUBO(bqm)

    density = mybqm.density
    avg_degree = mybqm.average_degree
    max_degree = mybqm.max_degree
    min_degree = mybqm.min_degree
    num_edges = len(bqm.quadratic)
    logical_vars = bqm.num_variables

    edges = list(bqm.quadratic.keys()) + [(v, v) for v in bqm.variables]

    thresholds = [4, 8, 12]
    single_runs = []

    for seed in range(num_runs):
        t0 = time.perf_counter()

        embedding = find_embedding(
            S=edges,
            T=target_graph.edges,
            timeout=300,
            random_seed=seed,
            threads=8,
            verbose=0,
        )

        t1 = time.perf_counter()
        elapsed = t1 - t0

        threshold_defaults = {
            f"pct_chains_gt_{t}": np.nan for t in thresholds
        }

        run_record = {
            "instance_id": os.path.basename(filepath),
            "num_run": seed+1,
            "success": False,
            "embedding_time": elapsed,
            "physical_qubits": np.nan,
            "max_chain": np.nan,
            "avg_chain": np.nan,
            "median_chain": np.nan,
            "std_chain": np.nan,
            "skewness": np.nan,
            "has_critical_chain": np.nan,
            **threshold_defaults,
            "num_vars": logical_vars,
            "density": density,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "min_degree": min_degree,
        }

        if embedding:
            chain_lengths = [len(chain) for chain in embedding.values()]
            used_qubits = sum(chain_lengths)

            max_c = np.max(chain_lengths)
            mean_c = np.mean(chain_lengths)
            median_c = np.median(chain_lengths)
            std_c = np.std(chain_lengths)

            num_chains_in_run = len(chain_lengths)
            skewness = max_c - median_c

            #soglie percentuali relative alla singola run
            run_thresholds = {
                f"pct_chains_gt_{t}": (
                    sum(1 for cl in chain_lengths if cl > t)
                    / num_chains_in_run
                )
                for t in thresholds
            }

            #aggiorno il record della singola run
            run_record.update(
                {
                    "success": True,
                    "physical_qubits": used_qubits,
                    "max_chain": int(max_c),
                    "avg_chain": float(mean_c),
                    "median_chain": float(median_c),
                    "std_chain": float(std_c),
                    "skewness": float(skewness),
                    "has_critical_chain": bool(max_c > 7),
                    **run_thresholds,
                }
            )

            print(
                f"   {seed+1}ª run completata con successo in {elapsed:.4f}s"
            )
        else:
            print(f"   {seed+1}ª run fallita in {elapsed:.4f}s")

        single_runs.append(run_record)

    return single_runs


def load_existing_results(parquet_path):
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def benchmark_zephyr_embedding():
    lp_dir = os.path.join(current, "lp")
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    if not os.path.exists(lp_dir):
        print(f"Errore: La cartella '{lp_dir}' non esiste.")
        sys.exit(1)

    lp_files = list(Path(lp_dir).rglob("*.[lL][pP]"))
    
    if not lp_files:
        print(f"Nessun file .lp trovato in '{lp_dir}'.")
        sys.exit(0)

    print(f"Trovati {len(lp_files)} file .lp in: {lp_dir}")

    target_graph = dnx.zephyr_graph(12)  #4800 qubits

    results_dir = os.path.join(current, "results")
    os.makedirs(results_dir, exist_ok=True)

    single_runs_path = os.path.join(
        results_dir, "embedding_runs_results.parquet"
    )

    #resume automatico basato sulle singole run
    df_runs_existing = load_existing_results(single_runs_path)

    already_done = (
        set(df_runs_existing["file_name"].astype(str))
        if not df_runs_existing.empty
        and "file_name" in df_runs_existing.columns
        else set()
    )

    todo_files = [
        f for f in lp_files if os.path.basename(f) not in already_done
    ]

    print(f"Già processati: {len(already_done)}")
    print(f"Da processare: {len(todo_files)}")

    for idx, filepath in enumerate(todo_files, 1):
        fname = os.path.basename(filepath)
        print(f"\n[{idx}/{len(todo_files)}] embedding su {fname}...")
        t0 = time.perf_counter()

        single_runs = run_embedding_test_for_file(
            filepath, target_graph, num_runs=num_runs
        )

        t1 = time.perf_counter()
        print(f"{num_runs} runs eseguite in {t1 - t0:.2f}s")

        #salva in modo incrementale sul parquet unico delle run
        df_runs_existing = pd.concat(
            [df_runs_existing, pd.DataFrame(single_runs)], ignore_index=True
        )
        df_runs_existing.to_parquet(
            single_runs_path, engine="pyarrow", index=False
        )

        print(f"Salvato aggiornamento per {fname}")

    print("\n=== ESECUZIONE COMPLETATA ===")


if __name__ == "__main__":
    benchmark_zephyr_embedding()

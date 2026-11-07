import pandas as pd

# Carica il file Parquet
file_path = "orchestrator_test_results_qpu_access_time.v1.0.parquet"
df = pd.read_parquet(file_path)

# Verifica se la colonna esiste ed è numerica
col_name = "estimated_leap_qpu_time_us"
if col_name in df.columns:
    # Calcoli statistici (ignorando eventuali NaN)
    mean_runtime = df[col_name].mean()
    total_runtime = df[col_name].sum()
    
    print(f"--- MEDIA ---")
    print(f"Media: {mean_runtime:.2f} µs")
    print(f"Media: {mean_runtime / 1000:.4f} ms")
    print(f"Media: {mean_runtime / 1000000:.6f} s\n")
    
    print(f"--- TOTALE ---")
    print(f"Totale stimato: {total_runtime:.2f} µs")
    print(f"Totale stimato: {total_runtime / 1000:.4f} ms")
    print(f"Totale stimato: {total_runtime / 1000000:.6f} s")
else:
    print(f"Colonna '{col_name}' non trovata. Colonne disponibili: {list(df.columns)}")

if col_name in df.columns and "qpu_calls" in df.columns:
    # Filtriamo o gestiamo eventuali chiamate QPU uguali a 0 per evitare divisioni per zero
    valid_df = df[df["qpu_calls"] > 0].copy()
    
    # Calcolo del tempo totale per singola riga diviso per le sue qpu_calls
    valid_df["time_per_qpu_call"] = valid_df[col_name] / valid_df["qpu_calls"]
    
    mean_per_call = valid_df["time_per_qpu_call"].mean()
    total_time = valid_df[col_name].sum()
    total_calls = valid_df["qpu_calls"].sum()
    
    # Alternativa globale: somma di tutti i tempi diviso la somma di tutte le chiamate QPU
    global_mean_per_call = total_time / total_calls if total_calls > 0 else 0
    
    print(f"--- MEDIA PER CHIAMATA QPU (Media delle medie per riga) ---")
    print(f"Media: {mean_per_call:,.2f} µs | {mean_per_call / 1000:,.4f} ms | {mean_per_call / 1_000_000:,.6f} s\n")
    
    print(f"--- MEDIA GLOBALE (Tempo totale / Totale chiamate QPU) ---")
    print(f"Totale chiamate QPU: {total_calls}")
    print(f"Media: {global_mean_per_call:,.2f} µs | {global_mean_per_call / 1000:,.4f} ms | {global_mean_per_call / 1_000_000:,.6f} s")
else:
    print(f"Colonne non trovate. Disponibili: {list(df.columns)}")

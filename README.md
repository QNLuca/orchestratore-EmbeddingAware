# QUBO Orchestrator

Orchestratore sviluppato con **FastAPI** e **Redis** per l'elaborazione e/o la decomposizione di problemi di ottimizzazione combinatoria BQM/QUBO derivati da file `.lp` (ad es. [QPLIB](https://qplib.zib.de)).

Il sistema coordina tramite dinamiche di routing risorse classiche (CPU/Tabu/SA) e Quantum Annealing (D-Wave QPU) adottando un innovativo sistema **embedding-aware** per valutare la migliore configurazione d'esecuzione per problemi o eventuali sottoproblemi sub-QUBO.

---

## Caratteristiche Principali

* **Embedding-Aware Partitioning**: Valutazione euristica e dinamica dell'embedding su topologie Quantum Annealer target (es. D-Wave Zephyr) per determinare l'instradamento ideale dei sub-QUBO.
* **Elaborazione Asincrona**: Gestione tramite job ID in background con persistenza e tracciamento dello stato su Redis.

---

## Architettura del Progetto

```text
.
├── policies/               # Strategie di risoluzione (Strategy Pattern)
│   ├── base.py             # Classe astratta BasePolicy
│   ├── alwaysQPU.py        # Policy basata su utilizzo QPU ogni volta che problema è embeddable
|   ├── alwaysCPU.py        # Policy basata su solo utilizzo CPU
|   ├── qbsolv.py           # Policy basata su algoritmo di Dwave qbsolv
|   ├── kerberos.py         # Policy basata su framework di Dwave kerberos
│   ├── mqt.py              # Policy con predizione ML (dal lavoro di Volpe D. et al, 2024)
│   └── ...                 # Altre policy classiche/ibride
├── modelsRF/               # Modelli Random Forest (.pkl)
├── orchestrator.py         # Orchestratore centrale della pipeline con policy embedding-aware
├── benchmark.py            # Runner per la suite benchmark comparativa
├── api.py                  # Semplice FastAPI Web Server
├── docker-compose.yml      # Orchestrazione dei container
└── .env.example            # Template delle variabili d'ambiente

```

---

## Prerequisiti

* **Docker** e **Docker Compose**
* **Python 3.10** (per sviluppo locale)

---

## Installazione e Configurazione

### 1. Clona il Repository

```bash
git clone https://github.com/QNLuca/orchestratore-EmbeddingAware.git
cd orchestratore-EmbeddingAware

```

### 2. Configura le Variabili d'Ambiente

Imposta le tue credenziali nel file .env:

```env
#Redis
REDIS_HOST=redis
REDIS_PORT=6379

#D-Wave QPU
DWAVE_API_TOKEN= your_api_key

```

### 3. Avvio tramite Docker Compose

```bash
docker compose up -d --build

```

La documentazione OpenAPI interattiva sarà consultabile su: `http://localhost:8000/docs`

---

## Utilizzo delle API

### 1. Risoluzione Asincrona di un problema LP (`POST /solve-lp`)

Avvia la risoluzione del problema specificato tramite l'URL del file `.lp`.

```bash
curl -X 'POST' \
  'http://localhost:8000/solve-lp' \
  -H 'Content-Type: application/json' \
  -d '{
  "lp_url": "https://example.com/problems/sample.lp",
  "lagrange_multiplier": 10.0,
  "max_iter": 5
}'

```

**Risposta (202 Accepted):**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PROCESSING",
  "message": "Il file LP è in fase di download e risoluzione."
}

```

---

### 2. Benchmark Comparativo (`POST /benchmark-lp`)

Esegue il problema `.lp` su **tutte le policy integrate** per confrontarne le metriche di prestazione.

```bash
curl -X 'POST' \
  'http://localhost:8000/benchmark-lp' \
  -H 'Content-Type: application/json' \
  -d '{
  "lp_url": "https://example.com/problems/sample.lp",
  "max_iter": 3
}'

```

---

### 3. Consultazione dello Stato / Risultati (`GET /status/{job_id}`)

Recupera lo stato di avanzamento o il payload finale del job elaborato.

```bash
curl -X 'GET' 'http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000'

```

**Esempio di Output (Benchmark Completato):**

```json
{
  "status": "COMPLETED",
  "type": "BENCHMARK",
  "lp_url": "https://example.com/problems/sample.lp",
  "num_variables": 120,
  "results": [
    {
      "policy": "embedding_aware",
      "best_energy": -145.50,
      "wall_clock_time": 1.12,
      "qpu_calls": 2,
      "cpu_calls": 3,
      "qpu_utilization_ratio": 0.4,
      "iterations": 3
    },
    {
      "policy": "mqt_qao (QA)",
      "best_energy": -142.10,
      "wall_clock_time": 0.89,
      "qpu_calls": 1,
      "cpu_calls": 4,
      "qpu_utilization_ratio": 0.2,
      "iterations": 3
    }
  ]
}

```
---

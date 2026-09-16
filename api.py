import json
import os
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, ConfigDict, Field
import redis

from benchmark import run_comparative_suite
from orchestrator import EmbeddingAwareOrchestrator

from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

app = FastAPI(
    title="QUBO Orchestrator API",
    description="API Asincrona per la risoluzione distribuita di problemi QUBO da file .lp"
)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

#inizializzo l'orchestratore emebedding-aware
orchestrator = EmbeddingAwareOrchestrator(
    policy_mode='embedding_aware',
    max_iter=5,
    convergence=2
)

### MODELLI DI RICHIESTA E RISPOSTA ###
class LPUrlRequest(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    lp_url: HttpUrl  = Field(      
        examples=["https://qplib.zib.de/lp/QPLIB_3852.lp"]
        )
    """
    URL al file .lp da elaborare.
    """
    lagrange_multiplier: Optional[float] = 10.0
    """
    Fattore di penalizzazione applicato per la conversione dei vincoli dell'istanza in termini quadratici di energia nel BQM.
    """
    max_iter: Optional[int] = 5
    """
    Numero massimo di iterazioni per l'orchestratore.
    """
    convergence: Optional[int] = 2
    """
    Soglia di convergenza all'energia minima oltre la quale terminare anticipatamente la risoluzione.
    """

class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str

def run_pipeline(job_id: str, req: LPUrlRequest):
    """Esegue la pipeline di download, parsing e risoluzione aggiornando lo stato su Redis."""
    try:
        #stato iniziale
        redis_client.set(f"job:{job_id}", json.dumps({"status": "PROCESSING", "progress": "Downloading and parsing .lp file..."}))

        #download e parsing del file .lp
        subqubo_obj, _, _ = orchestrator.load_qubo_lp(str(req.lp_url))

        if req.max_iter:
            orchestrator.max_iter = req.max_iter

        if req.convergence:
            orchestrator.convergence = req.convergence

        redis_client.set(f"job:{job_id}", json.dumps({"status": "PROCESSING", "progress": "Solving BQM via orchestrator..."}))

        #risoluzione embedding-aware
        result = orchestrator.solve(subqubo_obj)

        final_payload = {
            "status": "COMPLETED",
            "lp_url": str(req.lp_url),
            "num_variables": len(subqubo_obj.variables),
            #"bqm_serialized": subqubo_obj.to_serializable(),
            "result": result
        }

        #salvo risultato su Redis
        redis_client.setex(f"job:{job_id}", 86400, json.dumps(final_payload, default=str))

    except Exception as e:
        error_payload = {
            "status": "FAILED",
            "error": str(e)
        }
        redis_client.setex(f"job:{job_id}", 86400, json.dumps(error_payload))

def run_benchmark_pipeline(job_id: str, req: LPUrlRequest):
    """
    Esegue pipeline per scaricare file .lp ed eseguire il benchmark comparativo di tutte le policy implementate.
    """
    try:
        #stato iniziale
        redis_client.set(f"job:{job_id}", json.dumps({
            "status": "PROCESSING", 
            "progress": "Downloading and parsing .lp file for benchmarking..."
        }))

        #parsing del file .lp
        subqubo_obj, _, meta = orchestrator.load_qubo_lp(str(req.lp_url))

        if req.max_iter:
                orchestrator.max_iter = req.max_iter
        
        if req.convergence:
                orchestrator.convergence = req.convergence

        redis_client.set(f"job:{job_id}", json.dumps({
            "status": "PROCESSING", 
            "progress": f"Running comparative benchmark across all policies (Variables: {len(subqubo_obj.variables)})..."
        }))

        #eseguo benchmark comparativo presente in benchmark.py
        benchmark_results = run_comparative_suite(
            bqm=subqubo_obj,
            orchestratorEA=orchestrator,
            meta = meta
        )

        #formattazione e salvataggio dei risultati
        final_payload = {
            "status": "COMPLETED",
            "type": "BENCHMARK",
            "lp_url": str(req.lp_url),
            "num_variables": len(subqubo_obj.variables),
            "results": benchmark_results
        }

        #salva su Redis con TTL di 24 ore
        redis_client.setex(f"job:{job_id}", 86400, json.dumps(final_payload, default=str))

    except Exception as e:
        error_payload = {
            "status": "FAILED",
            "type": "BENCHMARK",
            "error": str(e)
        }
        redis_client.setex(f"job:{job_id}", 86400, json.dumps(error_payload))

### ENDPOINT API ###

@app.post("/solve-ea-lp", response_model=JobSubmitResponse, status_code=202)
def solve_lp_from_url(request: LPUrlRequest, background_tasks: BackgroundTasks):
    """
    Riceve l'URL di un file .lp e lo elabora con strategia embedding-aware.
    """

    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_pipeline, job_id, request)
    
    return JobSubmitResponse(
        job_id=job_id,
        status="PROCESSING",
        message="Il file .lp è in fase di download e risoluzione."
    )

@app.post("/benchmark-lp", response_model=JobSubmitResponse, status_code=202)
def benchmark_lp_from_url(request: LPUrlRequest, background_tasks: BackgroundTasks):
    """
    Riceve l'URL di un file .lp ed esegue il benchmark comparativo di ogni policy implementata.
    """

    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_benchmark_pipeline, job_id, request)
    
    return JobSubmitResponse(
        job_id=job_id,
        status="PROCESSING",
        message="Il file .lp è in fase di elaborazione per il benchmark comparativo."
    )


@app.get("/status/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Recupera lo stato corrente o il risultato di un job inviato in precedenza.
    """

    raw_data = redis_client.get(f"job:{job_id}")
    
    if not raw_data:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' non trovato o scaduto.")
    
    return json.loads(raw_data)
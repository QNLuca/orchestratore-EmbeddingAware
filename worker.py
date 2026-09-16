import os
import json
import time
import redis
import dimod
import tabu
import hybrid

from dimod.serialization.json import DimodEncoder, DimodDecoder

from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)


def solve_subqubo(sub_bqm: dimod.BinaryQuadraticModel, sampler_type: str) -> dimod.SampleSet:
        """
        Risolve direttamente il sub-QUBO con il sampler indicato.
        """
        init_state = hybrid.State(subproblem=sub_bqm)

        if sampler_type == "CPU":
            response = hybrid.TabuSubproblemSampler(num_reads=20).run(sub_bqm).result()
        else: #QPU
            final_state = hybrid.SimulatedAnnealingSubproblemSampler(num_reads=100).run(init_state).result()

        response = final_state.subsamples

        return response


def main():
    queues = ["queue:QPU", "queue:CPU"]
    print(f"Worker avviato")

    while True:
        try:
            item = r.blpop(queues, timeout=2)
            if item:
                queue_name, payload = item
                worker_type = queue_name.split(":")[-1]
                data = json.loads(payload)

                job_id = data["job_id"]
                task_id = data["task_id"]

                sub_bqm = dimod.BinaryQuadraticModel.from_serializable(data["bqm"])

                print(f"[{worker_type}] Prelevato task {task_id}...")
                
                sampleset = solve_subqubo(sub_bqm, worker_type)
                
                response_payload = {
                    "task_id": task_id,
                    "sampleset": sampleset.to_serializable()  #serializzazione con primitiva Python per incompatibilità metodo SDK con BQM
                }

                response_json = json.dumps(response_payload)

                response_channel = f"results:{job_id}"

                r.publish(response_channel, response_json)
                print(f"[{worker_type}] Task {task_id} completato e inviato.")
        except Exception as e:
            print(f"Errore worker: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    main()
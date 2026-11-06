import os

import hybrid
import dimod
import time
from typing import Dict, Any

from utils.cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dotenv import load_dotenv

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_KEY = os.getenv("DWAVE_API_TOKEN")

class RaceCounter(hybrid.Runnable):
    """
    Runnable custom compatibile con hybrid.Race e thread asincroni.
    Traccia l'esecuzione incrementando il contatore senza alterare lo State.
    """
    def __init__(self, counter_dict, key_name, **runopts):
        super().__init__(**runopts)
        self.counter_dict = counter_dict
        self.key_name = key_name

    def next(self, state, **runopts):

        self.counter_dict[self.key_name] += 1
        return state

class QBSolvPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="qbsolv", **kwargs)

    def solve(self, meta, bqm, global_embedding,
                is_all_embeddable,seed, qpu_identity, target_graph=None) -> Dict[str, Any]:
        lp_filename = meta.get("lp_filename", None)
        bestknown_solution = float(meta.get("bestknown_solution",None))

        invert_func = meta.get("invert_func", None)
                
        cqm = meta.get("cqm",None)

        is_maximize = meta.get("is_maximize", False)

        calls = {
            "cpu_calls": 0,
            "qpu_calls": 0
        }

        if DWAVE_API_KEY:
            if qpu_identity:
                qpu_sampler = DWaveSampler(solver=dict(identity=qpu_identity))
            else:
                qpu_sampler = DWaveSampler(solver={'topology__type': 'zephyr'})

            qpu_branch = hybrid.QPUSubproblemAutoEmbeddingSampler(
                            qpu_sampler=qpu_sampler,
                            num_reads=40,
                            sampling_params={'annealing_time': 20}
                            )
        else:
            qpu_branch = hybrid.SimulatedAnnealingSubproblemSampler()
        

        iteration = hybrid.Race(
            RaceCounter(calls, "cpu_calls")
            | hybrid.InterruptableTabuSampler(),
            hybrid.EnergyImpactDecomposer(size=min(self.subproblem_size, len(bqm.variables)), rolling=True, rolling_history=0.15)
            | hybrid.Const(subsamples=None) #reset subsamples
            | RaceCounter(calls, "qpu_calls")
            | qpu_branch
            | hybrid.SplatComposer()
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True)

        main = hybrid.Loop(iteration, max_iter=self.max_iter, convergence=self.convergence)
        
        start_time = time.perf_counter()
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)
        final_state = main.run(init_state).result()

        total_wall_clock = time.perf_counter() - start_time

        total_qpu = calls["qpu_calls"]
        total_cpu = calls["cpu_calls"]
        total_calls = total_qpu + total_cpu

        best_energy = float(final_state.samples.first.energy)

        if is_maximize:
            best_energy = -best_energy

        rel_gap = abs(best_energy - bestknown_solution) / abs(bestknown_solution) if bestknown_solution else None
        abs_gap = abs(best_energy - bestknown_solution) if bestknown_solution is not None else None

        best_sample = dict(final_state.samples.first.sample)

        cqmResults = {}
        
        if len(cqm.constraints)>0:
            cqmResults = CqmUtilities.evaluate_cqm_feasibility(best_sample,invert_func, cqm)
        
        return {
            "instance_id": lp_filename,
            "seed": seed,
            "is_maximization_objective": is_maximize,
            "policy": self.name,
            "best_energy": best_energy,
            "qplib_reference": bestknown_solution,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "wall_clock_time": total_wall_clock,
            "qpu_calls": total_qpu,
            "cpu_calls": total_cpu,
            "qpu_utilization_ratio": total_qpu/total_calls if total_calls > 0 else 0.0,
            
            **cqmResults
        }
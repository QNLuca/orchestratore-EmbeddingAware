import os

import hybrid
import dimod
import time
from typing import Dict, Any

from subQUBO import SubQUBO
from utils.cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dotenv import load_dotenv

from utils.mockSolverOffline import OfflineQPUSolver

#carica le variabili contenute nel file .env
load_dotenv()

DWAVE_API_KEY = os.getenv("DWAVE_API_TOKEN")

class QPUTimeEstimator(hybrid.Runnable):
    """Intercetta i sub-problemi nel ramo QPU e accumula il tempo stimato."""
    def __init__(self, time_accumulator_dict, key_name, num_reads=40, annealing_time=20, **runopts):
        super().__init__(**runopts)
        self.acc_dict = time_accumulator_dict
        self.key_name = key_name
        self.num_reads = num_reads
        self.annealing_time = annealing_time

    def next(self, state, **runopts):
        sub_bqm = getattr(state, 'subproblem', None)
        if sub_bqm and len(sub_bqm.variables) > 0:
            num_qubits = len(sub_bqm.variables)
            #calcola il tempo con solver fittizio
            estimated_us = OfflineQPUSolver.estimate_qpu_access_time(
                num_qubits=num_qubits,
                num_reads=self.num_reads,
                annealing_time=self.annealing_time
            )
            self.acc_dict[self.key_name] += estimated_us
            
        return state

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

        mybqm = SubQUBO(bqm)
        
        bqm_num_vars = mybqm.num_variables
        bqm_avg_degree = mybqm.average_degree
        bqm_density = mybqm.density

        calls = {
            "cpu_calls": 0,
            "qpu_calls": 0,
            "estimated_leap_qpu_time_us": 0.0
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
            | QPUTimeEstimator(calls, "estimated_leap_qpu_time_us", num_reads=40, annealing_time=20)
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
            "num_vars": bqm_num_vars,
            "avg_degree": bqm_avg_degree,
            "density": bqm_density,
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
            
            **cqmResults,
            "estimated_leap_qpu_time_us": calls["estimated_leap_qpu_time_us"],
        }
import time
from typing import Any, Dict
import dimod
import hybrid

from cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy
from policies.baseRouter import BaseRouter

class AlwaysCPURouter(BaseRouter):
    """Policy baseline: invia tutti i sotto-problemi esclusivamente al solutore classico CPU."""
    def __init__(self, qpu_sampler, cpu_sampler, **kwargs):
        super().__init__(qpu_sampler=qpu_sampler, cpu_sampler=cpu_sampler, **kwargs)
    
    def next(self, state, **kwargs):

        res_state = self._execute_route(state, route_to_qpu=False)

        return state.updated(subsamples=res_state)


class AlwaysCPUPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="always_cpu", **kwargs)

    def solve(self, meta, bqm, global_embedding, is_all_embeddable, target_graph=None) -> Dict[str, Any]:
        lp_filename = meta.get("lp_filename", None)
        bestknown_solution = float(meta.get("bestknown_solution",None))

        invert_func = meta.get("invert_func", None)
                
        cqm = meta.get("cqm",None)

        is_maximize = meta.get("is_maximize", False)

        start_time = time.perf_counter()
        
        cpu_sampler = hybrid.TabuSubproblemSampler(num_reads=20)

        router = AlwaysCPURouter(
            qpu_sampler=None,
            cpu_sampler=cpu_sampler
        );

        decomposer = hybrid.Unwind(
            hybrid.EnergyImpactDecomposer(
                size=min(self.subproblem_size, len(bqm.variables)),
                rolling=True,
                rolling_history=0.3,
                traversal='bfs'
            )
        )

        subproblem_pipeline = (
            decomposer
            #| hybrid.Const(subsamples=None)
            | hybrid.Map(router)
            | hybrid.Reduce(hybrid.Lambda(self._merge_substates))
            | hybrid.SplatComposer()
        )

        main = hybrid.Loop(subproblem_pipeline, max_iter=self.max_iter, convergence=self.convergence)
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)

        final_state = main.run(init_state).result()
        total_wall_clock = time.perf_counter() - start_time

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
            "is_maximization_objective": is_maximize,
            "policy": self.name,
            "best_energy": best_energy,
            "qplib_reference": bestknown_solution,
            "abs_gap": abs_gap,
            "rel_gap": rel_gap,
            "wall_clock_time": total_wall_clock,
            "qpu_calls": 0,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": 0.0,
            
            **cqmResults
        }
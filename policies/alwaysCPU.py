import time
from typing import Any, Dict, Optional
import dimod
import hybrid

from policies.routerPolicies import ConfigurableRouterSampler
from policies.base import BasePolicy

class AlwaysCPUPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="always_cpu", **kwargs)

    def solve(self, bqm, global_embedding, is_all_embeddable, target_graph=None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)
        router = ConfigurableRouterSampler(
            mode=self.name,
            global_embedding=global_embedding,
            is_all_embeddable=is_all_embeddable,
            qpu_sampler=None,
            cpu_sampler=cpu_fallback,
            target_graph=target_graph
        )

        decomposer = hybrid.Unwind(
            hybrid.EnergyImpactDecomposer(
                size=min(self.subproblem_size, len(bqm.variables)),
                rolling=True,
                rolling_history=1,
                traversal='bfs'
            )
        )

        subproblem_pipeline = (
            decomposer
            | hybrid.Map(router)
            | hybrid.Reduce(hybrid.Lambda(self._merge_substates))
            | hybrid.SplatComposer()
        )

        main = hybrid.Loop(subproblem_pipeline, max_iter=self.max_iter, convergence=self.convergence)
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)

        final_state = main.run(init_state).result()
        total_wall_clock = time.perf_counter() - start_time

        return {
            "policy": self.name,
            #"best_sample": dict(final_state.samples.first.sample),
            "best_energy": float(final_state.samples.first.energy),
            "wall_clock_time": total_wall_clock,
            "qpu_calls": router.qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": 0.0,
            "iterations": final_state.get('loop_iter', self.max_iter)
        }
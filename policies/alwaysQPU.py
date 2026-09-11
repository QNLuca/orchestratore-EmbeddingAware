import hybrid
import dimod
import time
from typing import Dict, Any

from policies.base import BasePolicy
from policies.routerPolicies import ConfigurableRouterSampler

class AlwaysQPUEmbeddablePolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="always_qpu_embeddable", **kwargs)

    def solve(self, bqm, global_embedding, is_all_embeddable, target_graph=None) -> Dict[str, Any]:
        qpu_target_global = hybrid.InterruptableSimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)

        if is_all_embeddable:
            start_time = time.perf_counter()
            init_state = hybrid.State.from_sample(
                hybrid.min_sample(bqm), 
                bqm
            ).updated(embedding=global_embedding)
            
            res_state = qpu_target_global.run(init_state).result()
            total_wall_clock = time.perf_counter() - start_time
            sampleset = res_state.subsamples if res_state.subsamples is not None else res_state.samples

            return {
                "policy": self.name,
                "best_energy": float(sampleset.first.energy),
                "wall_clock_time": total_wall_clock,
                "qpu_calls": 1,
                "cpu_calls": 0,
                "qpu_time": total_wall_clock,
                "cpu_time": 0.0,
                "qpu_utilization_ratio": 1.0,
                "decomposed": False,
                "iterations": 1
            }

        start_time = time.perf_counter()
        qpu_target = hybrid.InterruptableSimulatedAnnealingSubproblemSampler(num_reads=20, num_sweeps=1000)
        cpu_fallback = hybrid.TabuSubproblemSampler(num_reads=20)

        router = ConfigurableRouterSampler(
            mode=self.name,
            global_embedding=global_embedding,
            is_all_embeddable=is_all_embeddable,
            qpu_sampler=qpu_target,
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

        total_calls = router.qpu_calls + router.cpu_calls
        return {
            "policy": self.name,
            #"best_sample": dict(final_state.samples.first.sample),
            "best_energy": float(final_state.samples.first.energy),
            "wall_clock_time": total_wall_clock,
            "qpu_calls": router.qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": router.qpu_calls / total_calls if total_calls > 0 else 0.0,
            "iterations": final_state.get('loop_iter', self.max_iter)
        }
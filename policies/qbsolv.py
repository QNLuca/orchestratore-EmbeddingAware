import hybrid
import dimod
import time
from typing import Dict, Any

from policies.basePolicy import BasePolicy

class QBSolvPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="qbsolv", **kwargs)

    def solve(self, bqm, global_embedding,
                is_all_embeddable,target_graph=None) -> Dict[str, Any]:
        start_time = time.perf_counter()

        iteration = hybrid.Race(
            hybrid.InterruptableTabuSampler(),
            hybrid.EnergyImpactDecomposer(size=self.subproblem_size, rolling=True, rolling_history=0.15)
            #| hybrid.QPUSubproblemAutoEmbeddingSampler() #API key
            | hybrid.SimulatedAnnealingSubproblemSampler()
            | hybrid.SplatComposer()
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True)

        main = hybrid.Loop(iteration, max_iter=self.max_iter, convergence=self.convergence)
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)
        final_state = main.run(init_state).result()

        total_wall_clock = time.perf_counter() - start_time
        return {
            "policy": self.name,
            "best_energy": float(final_state.samples.first.energy),
            "wall_clock_time": total_wall_clock,
            "qpu_calls": float('nan'),
            "cpu_calls": float('nan'),
            "qpu_utilization_ratio": float('nan'),
            "iterations": final_state.get('loop_iter', self.max_iter)
        }
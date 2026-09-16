import time
from typing import Any, Dict
import dimod
import hybrid

from policies.basePolicy import BasePolicy

class KerberosPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="kerberos", **kwargs)

    def solve(self, bqm, global_embedding,
                is_all_embeddable,target_graph=None) -> Dict[str, Any]:
        start_time = time.perf_counter()

        #API key necessaria
        #kerberos = hybrid.Kerberos(max_iter=self.max_iter,convergence=self.convergence, sa_reads=20,sa_sweeps=1000,tabu_timeout=200,max_subproblem_size=self.subproblem_size)

        subproblem_solver = (
            hybrid.EnergyImpactDecomposer(
                size=min(self.subproblem_size, len(bqm.variables)), 
                rolling=True, 
                rolling_history=0.3, 
                traversal='bfs'
            )
            | hybrid.SimulatedAnnealingSubproblemSampler(num_reads=10, num_sweeps=500)
            | hybrid.SplatComposer()
        )

        kerberos_offline = hybrid.Loop(
            hybrid.Race(
                hybrid.BlockingIdentity(),
                hybrid.InterruptableTabuSampler(timeout=500),
                hybrid.SimulatedAnnealingProblemSampler(num_reads=1, num_sweeps=10000),
                subproblem_solver
            ) | hybrid.ArgMin() | hybrid.TrackMin(output=True),
            max_iter=self.max_iter,
            convergence=self.convergence
        )

        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)
        final_state = kerberos_offline.run(init_state).result()

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
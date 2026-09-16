import time
from typing import Any, Dict
import dimod
import hybrid
import dwave.graphs as dnx

from subQUBO import SubQUBO
from policies.basePolicy import BasePolicy
#from policies.routerPolicies import ConfigurableRouterSampler

from policies.baseRouter import BaseRouter

class OnlyMyFeaturesRouter(BaseRouter):
    """Invia a QPU ogni sub-problema per cui si riesce a calcolare un embedding valido."""
    def __init__(self, qpu_sampler, cpu_sampler, target_graph=None, max_size=40, max_density=0.5, max_degree=15, **kwargs):
        super().__init__(qpu_sampler=qpu_sampler, cpu_sampler=cpu_sampler, target_graph=target_graph, **kwargs)
        self.target_graph = target_graph
        self.max_size = max_size
        self.max_density = max_density
        self.max_degree = max_degree

    def next(self, state, **kwargs):
        sub_bqm = state.subproblem
        num_vars = len(sub_bqm.variables)
        sub_qubo = SubQUBO(bqm=sub_bqm)
        density = sub_qubo.density
        avg_degree = sub_qubo.average_degree

        embedding, is_embeddable = self.find_embedding(sub_bqm, self.target_graph)

        route_to_qpu = is_embeddable and (num_vars <= self.max_size) and (density <= self.max_density) and avg_degree < self.max_degree

        return self._execute_route(state, route_to_qpu=route_to_qpu, embedding=embedding)

class OnlyMyFeaturesBasedPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="only_my_features_based", **kwargs)

    def solve(self, bqm, global_embedding, is_all_embeddable, target_graph=None) -> Dict[str, Any]:
        mybqm = SubQUBO(bqm)
        max_density = 0.5
        max_size = 231

        qpu_target_global = hybrid.InterruptableSimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)

        if is_all_embeddable and (mybqm.num_variables <= max_size) and (mybqm.density <= max_density) and mybqm.average_degree <= 15:
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

        #router = ConfigurableRouterSampler(
        #    mode=self.name,
        #    global_embedding=global_embedding,
        #    is_all_embeddable=is_all_embeddable,
        #    qpu_sampler=qpu_target,
        #    cpu_sampler=cpu_fallback,
        #    target_graph=target_graph,
        #    max_size=max_size,
        #    max_density=max_density
        #)

        router = OnlyMyFeaturesRouter(
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=target_graph,
            max_size=max_size,
            max_density=max_density,
            max_degree=15
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
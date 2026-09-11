from importlib import resources
import os
import hybrid
import dimod
import time
from typing import Dict, Any
import pickle
import pandas as pd
import numpy as np

from policies.base import BasePolicy
from policies.routerPolicies import ConfigurableRouterSampler

#per sopprimere warning sklearn, tornare indietro di versione romperebbe l'attuale versione di SDK dwave ocean
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class MQTQAOPolicy(BasePolicy):
    """
    Policy basata sulla selezione predittiva del solutore tramite Machine Learning (modello Random Forest).

    Lavori a riguardo:
        - Volpe, D., et al. (2024). "A Predictive Approach for Selecting the Best Quantum Solver 
          for an Optimization Problem". IEEE QCE. arXiv:2408.03613.
        - Volpe, D., et al. (2024). "Towards an Automatic Framework for Solving Optimization 
          Problems with Quantum Computers". IEEE QSW. arXiv:2406.12840.
    """
    def __init__(self, **kwargs):
        super().__init__(name="mqt_qao", **kwargs)

    def extract_qubo_features(self,bqm: dimod.BinaryQuadraticModel) -> pd.DataFrame:
        num_vars = len(bqm.variables)
        linear_coeffs = list(bqm.linear.values())
        num_a = len(linear_coeffs)
        mean_a = float(np.mean(linear_coeffs)) if num_a > 0 else 0.0
        var_a = float(np.var(linear_coeffs)) if num_a > 0 else 0.0

        quad_coeffs = list(bqm.quadratic.values())
        num_b = len(quad_coeffs)
        mean_b = float(np.mean(quad_coeffs)) if num_b > 0 else 0.0
        var_b = float(np.var(quad_coeffs)) if num_b > 0 else 0.0

        all_coeffs = linear_coeffs + quad_coeffs
        mean_all = float(np.mean(all_coeffs)) if len(all_coeffs) > 0 else 0.0
        var_all = float(np.var(all_coeffs)) if len(all_coeffs) > 0 else 0.0

        feature_names = [
            "var qubo", "n 1", "n 2", "avg 1", "avg 2", 
            "var 1", "var 2", "total average", "total variance"
        ]
        features = [[num_vars, num_a, num_b, mean_a, mean_b, var_a, var_b, mean_all, var_all]]
        return pd.DataFrame(features, columns=feature_names)

    def predict_mqt_best_solver(self,bqm: dimod.BinaryQuadraticModel) -> str:
        try:
            #importato modelli dal lavoro di Volpe D. et al, 2024: https://github.com/cda-tum/mqt-qao/tree/main/src/mqt/qao/model/RandomForest
            with open(os.path.join("modelsRF", "model.pkl"), "rb") as f:
                model = pickle.load(f)
            with open(os.path.join("modelsRF", "Scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)
            with open(os.path.join("modelsRF", "ScalerKCross.pkl"), "rb") as f:
                scalerk = pickle.load(f)

            features = self.extract_qubo_features(bqm)
            features_scaled = scalerk.transform(features)
            features_scaled = scaler.transform(features_scaled)

            prediction = int(model.predict(features_scaled)[0])
            label_map = {0: 'QA', 1: 'QAOA', 2: 'VQE', 3: 'GAS', 4: 'SA'}
            return label_map.get(prediction, "SA")
        except Exception as e:
            print(e)
            return "SA"

    def solve(self, bqm, global_embedding, is_all_embeddable, target_graph=None) -> Dict[str, Any]:
        predicted_solver = self.predict_mqt_best_solver(bqm)

        policy_label = f"{self.name} ({predicted_solver})"

        qpu_target_global = hybrid.InterruptableSimulatedAnnealingProblemSampler(num_reads=20, num_sweeps=1000)

        if predicted_solver == "QA" and is_all_embeddable:
            start_time = time.perf_counter()
            init_state = hybrid.State.from_sample(
                hybrid.min_sample(bqm), 
                bqm
            ).updated(embedding=global_embedding)
            
            res_state = qpu_target_global.run(init_state).result()
            total_wall_clock = time.perf_counter() - start_time
            sampleset = res_state.subsamples if res_state.subsamples is not None else res_state.samples

            return {
                "policy": policy_label,
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
            predicted_solver_mqt=predicted_solver,
            predict_fn=self.predict_mqt_best_solver,
            qpu_sampler=qpu_target,
            cpu_sampler=cpu_fallback,
            target_graph=target_graph,
            verbose=True
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

        iteration = hybrid.Race(
            hybrid.InterruptableTabuSampler(timeout=200),
            subproblem_pipeline
        ) | hybrid.ArgMin() | hybrid.TrackMin(output=True)

        main = hybrid.Loop(iteration, max_iter=self.max_iter, convergence=self.convergence)
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)

        final_state = main.run(init_state).result()
        total_wall_clock = time.perf_counter() - start_time

        return {
            "policy": policy_label,
            #"best_sample": dict(final_state.samples.first.sample),
            "best_energy": float(final_state.samples.first.energy),
            "wall_clock_time": total_wall_clock,
            "qpu_calls": router.qpu_calls,
            "cpu_calls": router.cpu_calls,
            "qpu_utilization_ratio": 0.0,
            "iterations": final_state.get('loop_iter', self.max_iter)
        }
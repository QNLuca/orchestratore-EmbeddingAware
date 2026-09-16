import math
from typing import Dict, List, Optional, Union
import dimod
import dwave.graphs as dnx

from orchestrator import EmbeddingAwareOrchestrator
from subQUBO import SubQUBO
from policies.basePolicy import BasePolicy
from policies.alwaysCPU import AlwaysCPUPolicy
from policies.alwaysQPU import AlwaysQPUEmbeddablePolicy
from policies.kerberos import KerberosPolicy
from policies.qbsolv import QBSolvPolicy
from policies.mqt import MQTQAOPolicy
from policies.onlyMyFeaturesBased import OnlyMyFeaturesBasedPolicy

def run_comparative_suite(bqm: Union[dimod.BinaryQuadraticModel, SubQUBO], orchestratorEA: EmbeddingAwareOrchestrator, meta: Optional[Dict], target_graph=None, max_iter: Optional[int] = 5, convergence: Optional[int] = 2):
    if isinstance(bqm, SubQUBO):
        mybqm = bqm.bqm
    else:
        mybqm = SubQUBO(bqm)

    if target_graph is None:
        target_graph = dnx.zephyr_graph(15)

    print("Calcolo embedding globale per il BQM in corso...")
    if mybqm.num_variables < 231:
        global_embedding, is_all_embeddable = mybqm.compute_embedding(bqm, target_graph)
    else:
        global_embedding = {}
        is_all_embeddable = False
    print(f"Embedding globale completato. Embeddabile interamente: {is_all_embeddable}\n")

    subproblem_size = int(math.ceil(2.0 * math.sqrt(len(bqm.variables))))


    #istanzio policy
    pipeline_policies: List[BasePolicy] = [
        AlwaysCPUPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
        #KerberosPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
        #QBSolvPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
        #MQTQAOPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
        #AlwaysQPUEmbeddablePolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
        #OnlyMyFeaturesBasedPolicy(subproblem_size=subproblem_size, max_iter=max_iter, convergence=convergence),
    ]

    print(f"AVVIO BENCHMARK COMPARATIVO ROUTING POLICY (BQM N={len(bqm.variables)})")

    results = []

    resultMyPolicy = orchestratorEA.solve(bqm)

    results.append(resultMyPolicy)

    for policy in pipeline_policies:
        print(f"Valutazione Policy: [{policy.name}] ...")
        res = policy.solve(
            bqm=bqm,
            global_embedding=global_embedding,
            is_all_embeddable=is_all_embeddable,
            target_graph=target_graph
        )

        if meta.get("is_maximize", False):
            res["best_energy"] = -res["best_energy"]

        results.append(res)

    print(f"BENCHMARK COMPARATIVO COMPLETATO (BQM N={len(bqm.variables)})")

    return results
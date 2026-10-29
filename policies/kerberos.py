import time
from typing import Any, Dict
import dimod
import hybrid

from cqmUtilities import CqmUtilities
from policies.basePolicy import BasePolicy

#class TrackingArgMin(hybrid.ArgMin):
#    """
#    ArgMin custom che confronta gli States in arrivo da hybrid.Race,
#    seleziona il migliore e aggiorna un dizionario di conteggio.
#    """
#    def __init__(self, counter_dict: dict, branch_keys: list = None, **runopts):
#        super().__init__(**runopts)
#        self.counter_dict = counter_dict
#        #nomi delle chiavi da incrementare nel dizionario in base all'indice del ramo
#        self.branch_keys = branch_keys or ["tabu", "sa", "qpu"]
#
#    def next(self, states, **runopts):
#        #esegue il confronto e calcola lo State migliore nativamente
#        best_state = super().next(states, **runopts)
#        best_energy = float(best_state.samples.first.energy)
#
#        #individua l'indice del ramo che ha generato l'energia minima
#        for idx, state in enumerate(states):
#            energy = float(state.samples.first.energy)
#            #confronto con tolleranza per numeri in virgola mobile
#            if abs(energy - best_energy) < 1e-9:
#                if idx < len(self.branch_keys):
#                    key = self.branch_keys[idx]
#                    self.counter_dict[key] = self.counter_dict.get(key, 0) + 1
#                break #registra il primo ramo che ottiene il valore minimo
#
#        return best_state

#class TagState(hybrid.Runnable):
#    """Aggiunge una firma nei metadati dello State per identificare il ramo di appartenenza."""
#    def __init__(self, branch_name: str, **runopts):
#        super().__init__(**runopts)
#        self.branch_name = branch_name
#
#    def next(self, state, **runopts):
#        # Aggiunge/aggiorna il campo 'winning_branch' negli informativi dello State
#        return state.updated(winning_branch=self.branch_name)

#class WinningBranchTracer(hybrid.Runnable):
#    """Legge il tag dello State vincente selezionato da ArgMin e incrementa il contatore relativo."""
#    def __init__(self, counter_dict: dict, **runopts):
#        super().__init__(**runopts)
#        self.counter_dict = counter_dict
#
#    def next(self, state, **runopts):
#        #recupera il nome del ramo vincente salvato nello State da TagState
#        winner = state.get('winning_branch', 'unknown')
#        if winner in self.counter_dict:
#            self.counter_dict[winner] += 1
#        return state 

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

class KerberosPolicy(BasePolicy):
    def __init__(self, **kwargs):
        super().__init__(name="kerberos", **kwargs)

    def solve(self, meta, bqm, global_embedding,
                is_all_embeddable,target_graph=None) -> Dict[str, Any]:
        lp_filename = meta.get("lp_filename", None)
        bestknown_solution = float(meta.get("bestknown_solution",None))

        invert_func = meta.get("invert_func", None)
                
        cqm = meta.get("cqm",None)

        is_maximize = meta.get("is_maximize", False)
        
        start_time = time.perf_counter()

        #API key necessaria
        #kerberos = hybrid.Kerberos(max_iter=self.max_iter,convergence=self.convergence, sa_reads=20,sa_sweeps=1000,tabu_timeout=200,max_subproblem_size=self.subproblem_size)

        # Dizionario per tracciare le invocazioni esatte di ciascun ramo
        calls = {
            "cpu_tabu_calls": 0,
            "cpu_sa_calls": 0,
            "qpu_calls": 0
        }

        qpu_branch = (
            hybrid.EnergyImpactDecomposer(
                size=min(self.subproblem_size, len(bqm.variables)),
                rolling=True,
                rolling_history=0.3,
                traversal='bfs'
            )
            | hybrid.Const(subsamples=None) #reset subsamples
            | RaceCounter(calls, "qpu_calls")
            | hybrid.SimulatedAnnealingSubproblemSampler(num_reads=10, num_sweeps=500)
            | hybrid.SplatComposer()
        )

        # Rami CPU
        tabu_branch = (  
            RaceCounter(calls, "cpu_tabu_calls")
            | hybrid.InterruptableTabuSampler(timeout=500)
        )

        sa_global_branch = (
            RaceCounter(calls, "cpu_sa_calls")
            | hybrid.SimulatedAnnealingProblemSampler(num_reads=1, num_sweeps=10000)
        )

        kerberos_offline = hybrid.Loop(
            hybrid.Race(
                hybrid.BlockingIdentity(),
                tabu_branch,
                sa_global_branch,
                qpu_branch
            ) | hybrid.ArgMin() | hybrid.TrackMin(output=True),
            max_iter=self.max_iter,
            convergence=self.convergence
        )
    
        init_sample = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads=1).first.sample
        init_state = hybrid.State.from_sample(init_sample, bqm)
        final_state = kerberos_offline.run(init_state).result()

        total_wall_clock = time.perf_counter() - start_time

        total_qpu = calls["qpu_calls"]
        total_cpu = calls["cpu_tabu_calls"] + calls["cpu_sa_calls"]
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
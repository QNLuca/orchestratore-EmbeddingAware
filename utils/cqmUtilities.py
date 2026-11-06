import dimod
from typing import Dict, Any

class CqmUtilities:

    @staticmethod
    def evaluate_cqm_feasibility(
        best_sample_bqm: dict, 
        invert_func, 
        cqm: dimod.ConstrainedQuadraticModel
    ) -> Dict[str, Any]:
        """
        Riconverte il sample BQM nel formato del CQM originale e ne calcola l'ammissibilità.
        """
        #mappa la soluzione dal BQM di nuovo allo spazio di variabili del CQM
        cqm_sample = invert_func(best_sample_bqm)

        #verifica se la soluzione rispetta TUTTI i vincoli del CQM
        is_feasible = cqm.check_feasible(cqm_sample)

        #calcola il dettaglio delle violazioni dei vincoli
        #restituisce un dizionario {nome_vincolo: valore_violazione}
        violations = cqm.violations(cqm_sample)

        #metriche aggregate sulle violazioni
        num_violated_constraints = sum(1 for v in violations.values() if v > 0)
        total_violation_magnitude = sum(v for v in violations.values() if v > 0)
        max_violation = max(violations.values()) if violations else 0.0

        #calcola l'energia effettiva della funzione obiettivo originale (senza penalità di Lagrange)
        cqm_energy = cqm.objective.energy(cqm_sample)

        return {
            "is_feasible": is_feasible,
            "num_constraints": len(cqm.constraints),
            "num_violated_constraints": num_violated_constraints,
            "feasibility_ratio": (len(cqm.constraints) - num_violated_constraints) / len(cqm.constraints) if len(cqm.constraints) > 0 else 1.0,
            "total_violation_magnitude": float(total_violation_magnitude),
            "max_violation": float(max_violation),
            "cqm_objective_energy": float(cqm_energy),
        }
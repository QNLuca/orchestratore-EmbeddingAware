import dimod
from typing import Dict, Any

class CqmUtilities:

    @staticmethod
    def evaluate_cqm_feasibility( 
        all_samples,
        invert_func, 
        cqm: dimod.ConstrainedQuadraticModel,
        is_maximize: bool = False
    ) -> Dict[str, Any]:
        """
        Riconverte il sample BQM nel formato del CQM originale e, se disponibile,
        seleziona il campione ammissibile a minore energia, altrimenti ritorna il
        campione a minore energia segnalandone la non ammissibilità.
        """
        best_Cqm_sample = None
        is_feasible = False

        for s in all_samples.data(sorted_by='energy'): #anche se default esplicito l'odinamento per chiarezza
            #solo campioni che giungono dal campionamento diretto hanno più di un singolo sample.
            #prediligo, in quel caso, se disponibile, una soluzione ammissibile piuttosto che quella a più bassa energia.
            candidate_Bqm_sample = dict(s.sample)
            #riconverto BQM nello spazio CQM
            candidate_Cqm_sample = invert_func(candidate_Bqm_sample)
            #verifica se candidate_cqm_sample è ammissibile
            if cqm.check_feasible(candidate_Cqm_sample):
                #calcolo l'energia dell'obiettivo originale, essendo ordinati in modo crescente sull'energia prendo il primo ammissibile
                best_Cqm_sample = candidate_Cqm_sample
                raw_energy = float(cqm.objective.energy(best_Cqm_sample))
                is_feasible = True
                break

        #fallback: se nessun campione rispetta i vincoli, si prende comunque il primo in assoluto e si segnala la non ammissibilità
        if best_Cqm_sample is None:
            first_row = next(all_samples.data(sorted_by='energy'))
            best_Bqm_sample = dict(first_row.sample)
            best_Cqm_sample = invert_func(best_Bqm_sample)
            raw_energy = float(cqm.objective.energy(best_Cqm_sample))
            is_feasible = False

        #calcola il dettaglio delle violazioni dei vincoli
        #restituisce un dizionario {nome_vincolo: valore_violazione}
        violations = cqm.violations(best_Cqm_sample)

        #metriche aggregate sulle violazioni
        num_violated_constraints = sum(1 for v in violations.values() if v > 0)
        total_violation_magnitude = sum(v for v in violations.values() if v > 0)
        max_violation = max(violations.values()) if violations else 0.0

        #se funzione obiettivo è di massimizzazione, inverto segno energia
        best_energy = -raw_energy if is_maximize else raw_energy

        return {
            "is_feasible": is_feasible,
            "num_constraints": len(cqm.constraints),
            "num_violated_constraints": num_violated_constraints,
            "feasibility_ratio": (len(cqm.constraints) - num_violated_constraints) / len(cqm.constraints) if len(cqm.constraints) > 0 else 1.0,
            "total_violation_magnitude": float(total_violation_magnitude),
            "max_violation": float(max_violation),
            "cqm_objective_energy": best_energy,
        }


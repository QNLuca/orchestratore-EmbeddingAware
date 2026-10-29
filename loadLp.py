import os
from typing import Tuple
import urllib.request
import dimod
from typing import Dict, List, Optional, Tuple
import io
from urllib.parse import urlparse
from pathlib import Path

class LoadLp:

    @staticmethod
    def load_qubo_lp(
    uri: str,
    lagrange_multiplier: float = None, #per convertire i vincoli di un problema matematico in penalità energetiche, default a None per gestione automatica
    file: bool = False
    ) -> Tuple[dimod.BinaryQuadraticModel, Dict]:
        if not file:
            req = urllib.request.Request(uri, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                content = response.read()
        else:
            # Lettura da file locale
            with open(uri, "rb") as f:
                content = f.read()

        #rileva se la funzione obiettivo richiede di massimizzare o minimizzare
        text = content.decode("utf-8", errors="ignore").lower()
        is_maximize = False
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("\\") or line.startswith("/"):
                continue
            if line.startswith("maximize") or line.startswith("max"):
                is_maximize = True
                break
            if line.startswith("minimize") or line.startswith("min"):
                is_maximize = False
                break

        bestknown_solution = LoadLp.load_bestknown_result(uri)

        parsed_url = urlparse(uri)
        path = parsed_url.path if parsed_url.scheme else uri
    
        # Gestione sicura del nome file
        base_name = os.path.basename(path).split(".")[0]
        lp_filename = base_name.split("_")[1] if "_" in base_name else base_name


        cqm = dimod.lp.load(io.BytesIO(content))

        try:
            #invert è una funzione che consente di riottenere dei sample da confrontare con l'istanza originaria del cqm
            bqm, invert = dimod.cqm_to_bqm(cqm, 
                                           lagrange_multiplier=lagrange_multiplier #se è None viene gestito di default come 10x del max_bias
                                           )

            print(f"BQM variables: {len(bqm.variables)}")
        except ValueError as e:
            if "quadratic" in str(e).lower():
                print(f"Impossibile convertire in BQM: l'istanza contiene vincoli quadratici.")
                raise ValueError("L'istanza contiene vincoli quadratici e non può essere ridotta a BQM/QUBO direttamente.") from e

        meta = {
            "uri": uri,
            "lp_filename": lp_filename,
            "invert_func": invert,
            "cqm": cqm,
            "bestknown_solution": bestknown_solution,
            "n_variables_original": len(cqm.variables),
            "n_constraints": len(cqm.constraints),
            "is_maximize": is_maximize,
        }

        return bqm, meta #invert, meta, cqm


    @staticmethod
    def load_bestknown_result(
        instance: str,
        solutions_filepath: str = None
    ) -> Optional[float]:

        parsed_url = urlparse(instance)
        path = parsed_url.path if parsed_url.scheme else instance

        #pulisce il nome dell'istanza mantenendo solo il nome di base (es. 'test/lp/QPLIB_0018.lp' -> 'QPLIB_0018')
        clean_instance = os.path.basename(path).split(".")[0]
        
        if solutions_filepath is None:
            solutions_filepath = Path(__file__).resolve().parent / "test" / "lp" / "solutions.txt"
        else:
            solutions_filepath = Path(solutions_filepath)

        if not solutions_filepath.exists():
            raise FileNotFoundError(f"File delle soluzioni non trovato in: {solutions_filepath}")

        with open(solutions_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                status = parts[0]
                problem_name = parts[1]

                if problem_name == clean_instance:
                    if status == "=best=" and len(parts) >= 3:
                        return float(parts[2])
                    elif status == "=unkn=":
                        return None

        return None
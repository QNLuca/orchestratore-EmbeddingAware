from typing import Optional

import numpy as np

# https://github.com/dwavesystems/dwave-cloud-client/blob/0.14.8/dwave/cloud/solver.py#L1587
class OfflineQPUSolver:
    """
    Solver fittizio offline che incapsula la logica ufficiale di stima 
    del tempo QPU basata sulle specifiche di D-Wave Advantage2_System1.
    """
    def __init__(self):
        self.properties = { #Advantage2_system1 - graph_id: 01b03fc8dc
            'problem_timing_data': {
                'version': '1.0.0',
                'typical_programming_time': 33631.59,
                'reverse_annealing_with_reinit_prog_time_delta': 0.0,
                'reverse_annealing_without_reinit_prog_time_delta': 31.33,
                'default_programming_thermalization': 1000.0,
                'default_annealing_time': 20.0,
                'readout_time_model': 'pwl_log_log',
                'readout_time_model_parameters': [0,0.7118156236873552,1.622919864592155,2.5460233863267376,3.6610550848533783,1.3562242666114566,1.6328679675785214,1.7857781469507206,1.99055000334358,1.9905608299940198], 
                'qpu_delay_time_per_sample': 60.57,
                'reverse_annealing_with_reinit_delay_time_delta': -44.5,
                'reverse_annealing_without_reinit_delay_time_delta': -41.5,
                'decorrelation_max_nominal_anneal_time': 2000.0,
                'decorrelation_time_range': [0.0, 0.0],
                'default_readout_thermalization': 0.0
            }
        }

    def estimate_qpu_access_time(self,
                                 num_qubits: int,
                                 num_reads: int = 1,
                                 annealing_time: Optional[float] = None,
                                 anneal_schedule: Optional[list[tuple[float, float]]] = None,
                                 initial_state: Optional[list[tuple[float, float]]] = None,
                                 reverse_anneal: bool = False,
                                 reinitialize_state: bool = False,
                                 programming_thermalization: Optional[float] = None,
                                 readout_thermalization: Optional[float] = None,
                                 reduce_intersample_correlation: bool = False,
                                 **kwargs) -> float:
        
        if anneal_schedule and annealing_time:
            raise ValueError("set only one of ``anneal_schedule`` or ``annealing_time``")

        if anneal_schedule and anneal_schedule[0][1] == 1 and not initial_state:
            raise ValueError("reverse annealing requested (``anneal_schedule`` "
                             "starts with s==1) but ``initial_state`` not provided")

        try:
            problem_timing_data = self.properties['problem_timing_data']
        except:
            raise KeyError("selected solver is missing required property ``problem_timing_data``")

        try:
            version_timing_model = problem_timing_data['version']
        except:
            raise KeyError("selected solver is missing ``problem_timing_data`` field ``version``")

        try:
            typical_programming_time = problem_timing_data['typical_programming_time']
            ra_with_reinit_prog_time_delta = problem_timing_data['reverse_annealing_with_reinit_prog_time_delta']
            ra_without_reinit_prog_time_delta = problem_timing_data['reverse_annealing_without_reinit_prog_time_delta']
            default_programming_thermalization = problem_timing_data['default_programming_thermalization']
            default_annealing_time = problem_timing_data['default_annealing_time']
            readout_time_model = problem_timing_data['readout_time_model']
            readout_time_model_parameters = problem_timing_data['readout_time_model_parameters']
            qpu_delay_time_per_sample = problem_timing_data['qpu_delay_time_per_sample']
            ra_with_reinit_delay_time_delta = problem_timing_data['reverse_annealing_with_reinit_delay_time_delta']
            ra_without_reinit_delay_time_delta = problem_timing_data['reverse_annealing_without_reinit_delay_time_delta']
            decorrelation_max_nominal_anneal_time = problem_timing_data['decorrelation_max_nominal_anneal_time']
            decorrelation_time_range = problem_timing_data['decorrelation_time_range']
            default_readout_thermalization = problem_timing_data['default_readout_thermalization']

        except KeyError as err:
            err_msg = f"selected solver is missing ``problem_timing_data`` field " + \
                      f"{err} required by timing model version {version_timing_model}"
            raise ValueError(err_msg)

        if not version_timing_model.startswith("1.0."):
            raise ValueError(f"``estimate_qpu_access_time`` does not currently "
                             f"support timing model {version_timing_model} "
                             f"used by the selected solver")

        ra_programming_time = 0
        ra_delay_time = 0
        if anneal_schedule and anneal_schedule[0][1] == 1:
            if reinitialize_state:
                ra_programming_time = ra_with_reinit_prog_time_delta
                ra_delay_time = ra_with_reinit_delay_time_delta
            else:
                ra_programming_time = ra_without_reinit_prog_time_delta
                ra_delay_time = ra_without_reinit_delay_time_delta

        if programming_thermalization:
            programming_thermalization_time = programming_thermalization
        else:
            programming_thermalization_time = default_programming_thermalization

        programming_time = typical_programming_time + ra_programming_time + \
                           programming_thermalization_time

        anneal_time = default_annealing_time
        if annealing_time:
            anneal_time = annealing_time
        elif anneal_schedule:
            anneal_time = anneal_schedule[-1][0]

        n = len(readout_time_model_parameters)
        if n % 2:
             raise ValueError(f"for the selected solver, ``problem_timing_data`` "
                              f"property field ``readout_time_model_parameters`` "
                              f"is not of an even length as required by timing "
                              f"model version {version_timing_model}")

        q = readout_time_model_parameters[:n//2]
        t = readout_time_model_parameters[n//2:]
        if readout_time_model == 'pwl_log_log':
            readout_time = pow(10, np.interp(np.emath.log10(num_qubits), q, t))
        elif readout_time_model == 'pwl_linear':
            readout_time = np.interp(num_qubits, q, t)
        else:
            raise ValueError("``estimate_qpu_access_time`` does not support "
                             f"``readout_time_model`` value {readout_time_model} "
                             f"in version {version_timing_model}")

        if readout_thermalization:
            readout_thermalization_time = readout_thermalization
        else:
            readout_thermalization_time = default_readout_thermalization

        decorrelation_time = 0
        if reduce_intersample_correlation:
            r_min = decorrelation_time_range[0]
            r_max = decorrelation_time_range[1]
            decorrelation_time = anneal_time/decorrelation_max_nominal_anneal_time * (r_max - r_min) + r_min

        sampling_time_per_read = anneal_time + readout_time + qpu_delay_time_per_sample + \
            ra_delay_time + max(readout_thermalization_time, decorrelation_time)

        sampling_time = num_reads * sampling_time_per_read

        return sampling_time + programming_time
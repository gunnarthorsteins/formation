import json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
import voltaiq_studio as vs

import figures

def load_data_by_experiment(experiment: vs.test_record.test_record.TestRecord, trace_keys: list) -> pd.DataFrame:
    """Loads in experiment as pandas dataframe.
    
    Args:
        experiment (vs.test_record.test_record.TestRecord): Experiment data
        trace_keys (list): Data features to be loaded from dataset
        
    Returns:
        pd.DataFrame: Experiment time-series.
    """

    reader = experiment.make_time_series_reader()

    for trace_key in trace_keys:
        reader.add_trace_keys(trace_key)

    df = reader.read_pandas()
    df.set_index('h_test_time', inplace=True)

    return df


def get_json(filename: str) -> dict:
    """Fetches data from a json file.

    Args:
        filename (str): Filename, w/o file-ending.

    Returns:
        dict: JSON data as a dict.
    """

    with open(f"{filename}.json") as f:
        json_ = json.load(f)

    return json_


def generate_exp_id(name: str) -> str:
    """Generates experiment ID.
    
    For matching cycling with formation. Refer to
    match_cycling_with_formation() for further info.
    
    Args:
        name (str): Raw experiment name
        
    Returns:
        str: Parsed experiment ID
        
    Example
        >> name = 'UM Internal 0620 - Form - 13.014'
        >> id_ = generate_exp_id(name=name)
        >> print(id_)
        >> 13
    """

    return str(abs(int(name.name.split(' ')[-1].split('.')[0])))


def get_p_val(var_1: list, var_2: list) -> float:
    """Calculate p-value of two variables.
    
    Args:
        var_1 (list): First variable.
        var_2 (list): Second variable.
    
    Returns:
        float: p-value.
    """
    
    return stats.ttest_ind(var_1, var_2).pvalue


def zero_shift_formation(df: pd.DataFrame, V_min: float = 2.9) -> pd.DataFrame:
    """Zero-shifts formation dataframes.
    
    The raw formation data starts with a ~24 rest period.
    
    Args:
        df (pd.DataFrame): Raw formation dataframe.
        V_min (float, optional): Minimum voltage for formation.
            Defaults to 2.9.
        
    Returns:
        pd.DataFrame: Data with inital rest period removed.
    """

    df = df[df.h_potential > V_min]
    df.index -= df.index[0]

    return df


def match_cycling_with_formation(categorized_experiments: dict, formations: dict) -> dict:
    """Merges formation with cycling.
    
    Formation and cycling data are in two separate files
    for each experiment. This finds a match beteen the
    two exp IDs and merges the former into the latter.
    
    Args:
        categorized_experiments (dict): Entire cycling dataset,
            i.e. excluding formation data.
        formations (dict): Formation data.
    
    Returns:
        dict: Dataset as a nice, loopable dictionary.
    """

    for exp_serial_no in categorized_experiments:
        for form_serial_no, df in formations.items():
            if exp_serial_no == form_serial_no:
                categorized_experiments[exp_serial_no]['formation'] = df
                continue
        continue

    return categorized_experiments


def filter_HPPC(by_cycle: pd.DataFrame, column: str = 'h_discharge_capacity', threshold: float = 0.01) -> list:
    """Removes HPPC and embedded reference performance tests.
        
    Args:
        by_cycle (pd.DataFrame): Experiment grouped by cycle.
        column (str, optional): Column to be filtered on.
            Defaults to 'h_discharge_capacity'.
        threshold (float): Remove value if the difference between
            two consecutive values exceeds it.
            
    Returns:
        list: Indices where threshold is passed.
    """

    original = by_cycle[column].values
    shifted = by_cycle[column].shift(periods=1).values
    diff = original - shifted

    indices = [i for i, element in enumerate(diff) if abs(element) > threshold]

    return indices



def _get_V_indices(current: np.array, I_min: float = -1.5) -> list[list, list]:
    """Gets voltage indices at start of and end of (negative) current pulses.
    
    Helper function for get_resistance().
    
    Args:
        current (np.array): Current time series.
        I_min (float, optional): Upper current cutoff
            for HPPC. Defaults to -1.5.
        
    Returns:
        V_max_indices (list): Index of last
            voltage value for each high-current pulse.
        V_min_indices (list): Index of last
            voltage value before current pulse for
            each high-current pulse.
    """

    indices = current < I_min
    V_max_indices = [False] * len(indices)
    V_min_indices = [False] * len(indices)
    for i in range(len(indices)):
        if indices[i]:
            if not indices[i - 1]:
                V_min_indices[i - 1] = True
        if not indices[i]:
            if indices[i - 1]:
                V_max_indices[i - 1] = True

    return V_max_indices, V_min_indices


def get_resistance(cycling: pd.DataFrame, to_plot: int, cycle_no: int = 3) -> list:
    """Gets the resistance from the HPPC protocol.
    
    Args:
        cycling (pd.DataFrame): Cycling dataframe.
        to_plot (int): Whether to plot explanatory fig
            or not. Plots if value passed is 0, else not
            (I know, awful logic!).
        cycle_no (int, optional): Cycle number where HPPC
            protocol is performed. Defaults to 3.
            
    Returns:
        R (list): HPPC resistance values.
    """

    hppc_cycle = cycling[cycling.h_cycle == cycle_no]
    V = hppc_cycle.h_potential.values
    I = hppc_cycle.h_current.values

    V_max_indices, V_min_indices = _get_V_indices(
        current=hppc_cycle.h_current.values)

    if to_plot == 0:
        figures._explanatory_fig(
            df=hppc_cycle,
            indices=[V_min_indices, V_max_indices]
        )

    delta_V = V[V_max_indices] - V[V_min_indices]

    R = delta_V / I[V_max_indices] * 1000

    return R[:-2]  # Don't know why


def get_capacities(formation: pd.DataFrame) -> list[np.array]:
    """Gets the (dis)charge capacity.
    
    Following the original work, they use charge capacity
    on the first formation cycle and discharge capacity
    and the end of formation to calculate diagnostic signals.
    
    Args:
        formation (pd.DataFrame): Dataframe, grouped by cycle
            [using max()].
            
    Returns:
        np.array: charge capacity of _first_ formation cycle.
        np.array: discharge capacity of _last_ formation cycle.
    """

    index = 6 if max(
        formation.index) == 7 else -1
    discharge_capacity = formation.h_discharge_capacity.iloc[index]

    charge_capacity = formation.h_charge_capacity.iloc[0]

    return charge_capacity, discharge_capacity


def get_R5SoC(Rs: np.ndarray, SoC: list = [3.5, 7.8], desired_SoC: int = 5) -> float:
    """Interpolates resistances at two SoCs to get R_5SoC
    
    Args:
        Rs (np.ndarray): Resistance at two SoCs.
        SoC (list): Corresponding State-of-Charge.
        desired_SoC (int): The SoC we want.
    
    Returns:
        float: The interpolates resistance.
    """

    set_val = 5
    y_interp = interp1d(SoC, Rs)
    R_5 = y_interp(set_val)

    return R_5.item()
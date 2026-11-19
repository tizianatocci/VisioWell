import pandas as pd
import numpy as np
import os
import warnings
    
def LoadCsvFile(dir_results, result_file, sep=";"):
    """
    Load .csv file into a pd dataframe and print columns.

    Parameters
    ----------
    dir_results : string.
    result_file : string.
    sep : string, optional
        The default is ";".

    Raises
    ------
    TypeError if the arguments are not strings.

    Returns
    -------
    df : pd dataframe.

    """
    if not isinstance(dir_results, str):
        raise TypeError(f"'dir_results' must be string but got {type(dir_results)}")
    if not isinstance(result_file, str):
        raise TypeError(f"'result_file' must be string but got {type(result_file)}")
    if not isinstance(sep, str):
        raise TypeError(f"'sep' must be string but got {type(sep)}")
    df = pd.read_csv(os.path.join(dir_results, result_file), sep=sep)  # load file
    print(f"The file {result_file} contains the following columns:\n{list(df.columns)}")
    
    return df
    

def FindColonySize(df):
    """
    Function to retrieve colony size from result .csv file.

    Parameters
    ----------
    df : pd dataframe.

    Raises
    ------
    TypeError
        if df is not a pd dataframe.
    KeyError
        if the column from which retrieve colony size is not in df.

    Returns
    -------
    output: numeric.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}.")

    column_name = "Correction"  # column name from which the colony size has to be retrieved
    if column_name not in df.columns:
        raise KeyError(f"'column_name' is not in df.columns. The columns are: {list(df.columns)}")
    print("Retrieving colony size")
        
    return df[column_name].dropna().iloc[-1]  # last value not NA


def set_additional_T3s_to_nan(values_dict, n_to_delete):
    """
    Set T3.x to np.nan.

    Parameters
    ----------
    values_dict : dict.
    n_to_delete : int.

    Raises
    ------
    TypeError
        if inputs are not the expected format.
    ValueError.

    Returns
    -------
    values_dict : dict.

    """
    if not isinstance(values_dict, dict):
        raise TypeError(f"values_dict expected to be dictionary but got {type(values_dict)}")
    if not isinstance(n_to_delete, int):
        raise TypeError(f"n_to_delete expected to be dictionary but got {type(n_to_delete)}")
    
    keys = list(values_dict.keys())    # find the last key that is not NaN
    for i in reversed(range(len(keys))):
        key = keys[i]
        if not np.isnan(values_dict[key]):  # Check if the value is not NaN
            #print(f"Last not np.nan key: {key}, Value: {values_dict[key]}")
            last_non_nan_index=i
            break  # Stop once we find the last not np.nan key
        
    
    deleted_count = 0
    if n_to_delete > 0:
        for i in reversed(range(last_non_nan_index+1)):
            key = keys[i]
            #print(key)
            if not np.isnan(values_dict[key]):  
                values_dict[key] = np.nan
                deleted_count += 1
            else:
               raise ValueError("It should not be np.nan")
            if deleted_count == n_to_delete:
                break
    return values_dict

def fill_dictionary(values_dict, height_jump, n_frame):
    """
    Fill dictionary of times of division.

    Parameters
    ----------
    values_dict : dict.
    height_jump : int.
    n_frame : int.

    Raises
    ------
    TypeError
        if arguments are not the expected types.

    Returns
    -------
    values_dict : dict.

    """
    if not isinstance(values_dict, dict):
        raise TypeError(f"values_dict expected to be dictionary but got {type(values_dict)}")
    if not isinstance(height_jump, int):
        raise TypeError(f"height_jump expected to be an integer but got {type(height_jump)}")
    if not isinstance(n_frame, int):
        raise TypeError(f"n_frame expected to be an integer but got {type(n_frame)}")
     
    keys = list(values_dict.keys())
    
    added_count = 0
    for i in range(len(keys)):
        key=keys[i]
        if values_dict[key]==0:
            values_dict[key]=n_frame  # +1 if human notation?
            added_count += 1
        if added_count == height_jump:
            break
    return values_dict
    
def ConvertFrameInTime(result, imaging_frequence):
    """
    Convert number of frame in time based on imaging frequence.

    Parameters
    ----------
    result : dict.
    imaging_frequence : float.

    Raises
    ------
    TypeError if input is not as expected.

    Returns
    -------
    result : dict

    """
    if not isinstance(result, dict):
        raise TypeError(f"'result' should be a dictionary but got {type(result)}")
    if not isinstance(imaging_frequence, float):
        raise TypeError(f"'imaging_frequence' should be a float but got {type(imaging_frequence)}")
    for key, value in result.items():
        result[key] = result[key]/imaging_frequence
    return result
    
def FindDivisionTime(df):
    """
    WRONG FUNCTION
    Retrieve division times.

    Parameters
    ----------
    df : pd dataframe.

    Raises
    ------
    TypeError
        if df is not a pandas df.
    KeyError
        if "Correction" is not a column in df.

    Returns
    -------
    result : dict.
    change_indices : list.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}.")

    # convert the column to a list
    column_name = "Correction"  # column name from which the colony size has to be retrieved
    if column_name not in df.columns:
        raise KeyError(f"'column_name' is not in df.columns. The columns are: {list(df.columns)}")
    correction_values = df[column_name].tolist()
    
    
    keys = ["T1", "T2_1", "T2_2", "T3_1", "T3_2", "T3_3", "T3_4"]
    result = {key: 0 for key in keys}
    # list of indexes where there is a change of value
    change_indices = []
    #keys_lost=[]
    for i in range(1, len(correction_values)):  # On commence à 1 pour comparer avec i-1
        flag_increasing = correction_values[i] > correction_values[i - 1]
        flag_decreasing = correction_values[i] < correction_values[i - 1]
        #if i==135:
        #    breakpoint()
        if flag_increasing and correction_values[i]>1:  # Si la valeur augmente
            height_jump = int(correction_values[i] - correction_values[i - 1])
            # the index is sorted as many times as the height jump
            # for example, if I pass from 2 to 4 cells: I have to assign two times of division
            change_indices.extend([i] * int(height_jump)) # i+1 to count in human notation and not python notation
            result=fill_dictionary(result, height_jump, i)
        elif flag_decreasing and any(value == 0 for value in result.values()):
            #if i>135:
            #    breakpoint()
            height_jump = int(abs(correction_values[i] - correction_values[i - 1]))
            flag_assigned= result["T1"] !=0 and\
                all(v == 0 for k, v in result.items() if k not in ("T1", "T2.1"))
            # if T1 was assigned or if both T1, T2.1 were assigned
            if flag_assigned: 
                #breakpoint()
                # always lost T2.2
                #keys_lost.append("T2_2", "T3_3", "T3_4") 
                result["T2_2"]=np.nan
                result["T3_3"]=np.nan
                result["T3_4"]=np.nan
                # eventually losing another T3.x
                n_add_lost_times = height_jump - 1
                result=set_additional_T3s_to_nan(result, n_add_lost_times)
             
                
            else: 
            # if T1, T2.1, T2.2 were assigned
            # or T1, T.2.1, T.3.1 were assigned
            # keys_lost.append("T3_4") # t3.4 if nothing was already lost else T3.2
            # else t3.2
            # if T1, T.2.1, T.2.2, T3.1 were assigned
            # if T1, T.2.1, T.2.2, T3.1, T3.2 were assigned
            # if T1, T.2.1, T.2.2, T3.1, T3.2, T3.3 were assigned
            # i lost the last T3.x key that is not np.nan
                result=set_additional_T3s_to_nan(result, height_jump)
                
                
    # Détecter les changements de valeur dans la colonne correction
    #keys = ["T1", "T2_1", "T2_2", "T3_1", "T3_2", "T3_3", "T3_4"]
    #result = {key: np.nan for key in keys}
    
    #for key, value in zip(result.keys(), change_indices):
    #    result[key] = value
    
    for key, value in result.items():
        if value == 0:
            result[key] = np.nan
    print("Retrieving times of division")
    
    return result, change_indices


def FindQualityIndexes(df):
    """
    Retrieve quality indexes.

    Parameters
    ----------
    df : pandas dataframe.

    Raises
    ------
    TypeError
        if df is not pandas dataframe.
    ValueError
        If there is an issue with the indexes.

    Returns
    -------
    row_dict : dict.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}.")
    list_columns = ['Indic_exact', 'Indic_exact-9', 'Indic_std', 'Indic_std-9',\
       'Osc_index','Osc_height_index', 'Osc_index-9', 'Osc_height_index-9']
    column_contained = np.array([col in df.columns for col in list_columns])
    valid_columns = np.array(list_columns)[column_contained]
    part_df = df[valid_columns]
    first_row = part_df.iloc[0].values
    flag_equal_rows = all(np.array_equal(row, first_row, equal_nan=True) for row in part_df.iloc[1:].values)
    if not flag_equal_rows:
        raise ValueError("The quality indexes should have the same values in the dataframe.")
    row_dict = part_df.iloc[0].to_dict()
    
    if np.any(~column_contained):
        inexistent_columns = np.array(list_columns)[~column_contained]
        warnings.warn(f"The following columns are missing from the DataFrame: {inexistent_columns}")
        row_dict.update({col: np.nan for col in inexistent_columns if col not in df.columns})
        row_dict = {col: row_dict[col] for col in list_columns}
    
    print("Retrieving quality indexes")
    
    return row_dict


def FindDiscrepancies(result: dict, time_interval_discrepancy:int) -> bool:
    """
    Checks for discrepancies between measurements in 'result', ignoring np.nan values.
    
    Args:
        result (dict): Dictionary with keys 'T1', 'T2_1', 'T2_2', 'T3_1'.
        time_interval_discrepancy (int): The time interval to check for discrepancies.
    
    Returns:
        bool: True if there is a discrepancy (difference < time_interval_discrepancy), False otherwise.
    
    Raises:
        KeyError: If any required key is missing.
    """
    required_keys = ["T1", "T2_1", "T2_2", "T3_1"]
    
    # Check that all required keys are present
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")

    t1 = result["T1"]
    t2_1 = result["T2_1"]
    t2_2 = result["T2_2"]
    t3_1 = result["T3_1"]

    # Check first pair only if neither is np.nan
    if not (np.isnan(t1) or np.isnan(t2_1)):
        if t2_1 - t1 < time_interval_discrepancy:
            return True

    # Check second pair only if neither is np.nan
    if not (np.isnan(t2_2) or np.isnan(t3_1)):
        if t3_1 - t2_2 < time_interval_discrepancy:
            return True

    return False


def ComputeTrustIndex(oscillation_indicator: float, flag_discrepancy: bool, colony_size: int, colony_size_limit: int) -> float:
    """
    Computes a trust index based on oscillation, discrepancies, and colony size.
    It applies a penalty equal to oscillation_indicator, 
    a penalty of 0.3 if discrepancies are found, 
    and a penalty of 0.2 if the colony size exceeds a specified limit.

    Args:
        oscillation_indicator (float): Measure of oscillation.
        flag_discrepancy (bool): True if discrepancies were found.
        colony_size (int): Number of elements in the colony.
        colony_size_limit (int): Threshold for colony size to apply penalty.
    
    Returns:
        float: Calculated trust index (can be less than 0 if penalties are high).

    Raises:
        ValueError: If inputs are not of expected types or out of expected ranges.
    """
    # Input validation
    if not isinstance(oscillation_indicator, (int, float)):
        raise ValueError("oscillation_indicator must be a number.")
    if not isinstance(flag_discrepancy, bool):
        raise ValueError("flag_discrepancy must be a boolean.")


    trust_score = 1 - oscillation_indicator

    # Subtract penalty for discrepancies
    if flag_discrepancy:
        trust_score -= 0.3

    # Subtract penalty for large colony size
    if colony_size > colony_size_limit:
        trust_score -= 0.2

    return trust_score



def is_unassigned(event):
    """
    Return True if the event is not yet assigned (np.nan)
    """
    return pd.isna(event)


def chronological_model(df):
    """
    Chronological model handling total death to zero.
    """
    BLOCKED = "blocked"
    column_name = "Correction"
    correction_values = df[column_name].tolist()
    event_keys = ["T1", "T2_1", "T2_2", "T3_1", "T3_2", "T3_3", "T3_4"]
    max_events = len(event_keys)
    events = [np.nan] * max_events
    current_cells = correction_values[0]

    if current_cells != 1:
        raise KeyError(f"First cell is not 1 but it is {current_cells}!")

    for frame_index in range(1, len(correction_values)):
        prev_cells = current_cells
        curr_cells = correction_values[frame_index]

        # --- GROWTH
        if curr_cells > prev_cells:
            height_jump = curr_cells - prev_cells
            for i in range(max_events):
                if height_jump == 0:
                    break
                if is_unassigned(events[i]):
                    events[i] = frame_index
                    height_jump -= 1

        # --- DEATH
        elif curr_cells < prev_cells:
            lost_cells = prev_cells - curr_cells

            # total death
            if curr_cells == 0:
                for i in range(max_events):
                    if is_unassigned(events[i]):
                        events[i] = BLOCKED
            else:
                # Block T2_2 if free
                t2_2_index = 2
                if is_unassigned(events[t2_2_index]):
                    events[t2_2_index] = BLOCKED
                # Block lost cells in T3
                for _ in range(lost_cells):
                    for t3_idx in reversed(range(3, 7)):
                        if is_unassigned(events[t3_idx]):
                            events[t3_idx] = BLOCKED
                            break

        current_cells = curr_cells

    # --- FINAL ASSIGNMENT
    result = {key: np.nan for key in event_keys}
    for i, key in enumerate(event_keys):
        if events[i] != BLOCKED:
            result[key] = events[i]

    return result




def chronological_model_block_lineage(df):
    """
    Chronological model handling total death to zero and if T2.2  is delete then
    T3.3 and T3.4 are deleted too
    """

    BLOCKED = "blocked"
    column_name = "Correction"
    correction_values = df[column_name].tolist()
    event_keys = ["T1", "T2_1", "T2_2", "T3_1", "T3_2", "T3_3", "T3_4"]
    max_events = len(event_keys)
    events = [np.nan] * max_events
    current_cells = int(correction_values[0])

    if current_cells != 1:
        raise KeyError(f"First cell is not 1 but it is {current_cells}!")

    for frame_index in range(1, len(correction_values)):
        prev_cells = int(current_cells)
        curr_cells = int(correction_values[frame_index])

        # --- GROWTH
        if curr_cells > prev_cells:
            height_jump = curr_cells - prev_cells
            for i in range(max_events):
                if height_jump == 0:
                    break
                if is_unassigned(events[i]):
                    events[i] = frame_index
                    height_jump -= 1
            # --- DEATH
        elif curr_cells < prev_cells:
            lost_cells = prev_cells - curr_cells

            # total death
            if curr_cells == 0:
                for i in range(max_events):
                    if is_unassigned(events[i]):
                        events[i] = BLOCKED
            else:
                # Block T2_2 if free and corresponding T3
                t2_2_index = 2
                t3_indices = [3, 4, 5, 6]  
                if is_unassigned(events[t2_2_index]):
                    events[t2_2_index] = BLOCKED
                    for t3_idx in [5, 6]:
                        if is_unassigned(events[t3_idx]):
                            events[t3_idx] = BLOCKED
                    lost_cells -= 1
                # Block lost cells in T3
                for _ in range(lost_cells):
                    for t3_idx in reversed(t3_indices):
                        if is_unassigned(events[t3_idx]):
                            events[t3_idx] = BLOCKED
                            break
        current_cells = curr_cells

    # --- FINAL ASSIGNMENT
    result = {key: np.nan for key in event_keys}
    for i, key in enumerate(event_keys):
        if events[i] != BLOCKED:
            result[key] = events[i]

    return result



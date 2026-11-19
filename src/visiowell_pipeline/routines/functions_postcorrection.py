# -*- coding: utf-8 -*-
"""
Functions for step2_curvecorrection.py
Author: Tiziana Tocci
"""
import numpy as np
import ruptures as rpt
from scipy import stats
import pandas as pd
import re
import os


def algorithm_change_point_detection(signal, which_algorithm, which_model, penalty):
    """
    Choose and apply the proper change point detection function,

    Parameters
    ----------
    signal : np array.
    which_algorithm : str. "pelt" or "binseng" or "bottomup".
    which_model : "l1" or "l2" or "rbf"....
    penality : int. Pentalty value.

    Returns
    -------
    result : list. Changing points.

    """
    if not isinstance(signal, np.ndarray):
        raise ValueError("'signal' must be a np array.")
    if which_algorithm=="pelt":
        algo = rpt.Pelt(model=which_model).fit(signal)
        result = algo.predict(pen=penalty)
    elif which_algorithm=="binseng":
        algo = rpt.Binseg(model=which_model).fit(signal)
        result = algo.predict(pen=penalty)
    elif which_algorithm=="bottomup":
        algo = rpt.BottomUp(model="which_model").fit(signal)
        result = algo.predict(pen=penalty)
    else:
        raise ValueError(f"No algorithm found corresponding to {which_algorithm}.\n Valid options are: 'pelt', 'binseng', 'bottomup'.")
    return result


def curve_correction_v2(all_numbers):
    """
    This function does a shallowed correction on the input array.
    Five points and four points are considered for the correction.
    The cases handled are:
        5p: steady - steady - error (too big or too small) - steady - steady
        4p: steady - steady - error (too big or too small) - steady
        4p: steady - error (too big or too small) - steady - steady
    In all of thiese cases, the error is set to the values around it.

    Parameters
    ----------
    all_numbers : np array with shape[0] of at least 5.
    
    Raise
    -----
    ValueError: if the input is not a numpy array or if it is a numpy array with not enough elements.
    
    Returns
    -------
    new_numbers : np array.

    """
    # output
    if not isinstance(all_numbers, np.ndarray) or all_numbers.shape[0]<5:
        raise ValueError("'all_numbers' must be a np array with shape[0]>=5.")
    new_numbers = all_numbers.copy()
    for i in range(2,len(new_numbers)-2): 
        # 5p: steady - steady - error (too big or too small) - steady - steady
        cond_5p=(new_numbers[i-1] == new_numbers[i+1]) and (new_numbers[i-2]==new_numbers[i-1]) and  (new_numbers[i+1]==new_numbers[i+2])\
            and (new_numbers[i]!=new_numbers[i-1])
        # 4p: steady - steady - error (too big or too small) - steady
        cond_4p_v1=(new_numbers[i] != new_numbers[i+1]) and\
            (new_numbers[i+1]==new_numbers[i-1]) and (new_numbers[i-1]==new_numbers[i-2])
        # 4p: steady - error (too big or too small) - steady - steady
        cond_4p_v2 = (new_numbers[i] != new_numbers[i+1]) and\
            (new_numbers[i+1]==new_numbers[i-1]) and (new_numbers[i+1]==new_numbers[i+2])
        if cond_5p or cond_4p_v1 or cond_4p_v2:
            new_numbers[i] = new_numbers[i-1]

    return new_numbers



def curve_correction_statistics(signal, list_changing_points):
    """
    Function to correct the curve (the input signal.)
    Based on the changing points previously identified, for a single region, the signal is flattened to
    the mean of the three statistics: mean, median and mode.

    Parameters
    ----------
    signal : numpy array.
    list_changing_points : list of ints. List of changing points. 
    It has to be sorted and include values within 0 and len(signal)

    Raise
    ------
    ValueError: if signal or list_changing_points are not the expected types.

    Returns
    -------
    new_signal : np array. Flattened signal.

    """
    #breakpoint()
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"'signal' must be a np array but got {type(signal)}.")
    if signal.size==0 or len(signal)<2:
        raise ValueError("'signal must be a non empty np array with at least two points.")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("'signal' cannot contain nans or infinite values.")
    if not isinstance(list_changing_points, list):
        raise ValueError(f"'list_changing_points' must be a list but got type{list_changing_points}.")
    if not all(isinstance(number, int) for number in list_changing_points):
        raise ValueError(f"All elements in 'list_changing_points' must be integers, but got {list_changing_points}.")
    if not np.all(list_changing_points==np.sort(list_changing_points)):
        raise ValueError("'list_changing_points' must be ordered")
    if not all(0 <= changing_point <= len(signal) for changing_point in list_changing_points):
        raise ValueError("'list_changing_points' must contain values between 0 and len(signal).")
    new_signal=signal.copy()
    for i in range(len(list_changing_points)):
        if i==0:
            first_point=0
        else:
            first_point=list_changing_points[i-1]
        second_point = list_changing_points[i]
        local_signal = new_signal[first_point:second_point]
        result_mode=stats.mode(local_signal, keepdims=True)
        mode = result_mode.mode[0]
        median=np.median(local_signal)
        mean=np.mean(local_signal)
        new_signal[first_point:second_point] = np.round((mode+mean+median)/3)
        
    return new_signal
    

def correct_based_on_neighbors(signal, list_changing_points):
    """
    Context: monotonic time series. 
    Function to correct to impose that the time series only can stay constant or increase in value.
    Not all cases are for the moment handled.
    Correction done by considering three or four sections at the time.

    Parameters
    ----------
    signal : numpy array. Signal to correct.
    list_changing_points : list. List of changing points obtain by a suited algorithm able to detect changing points.
        For more details, refer to the "ruptures" library documentation.
        
    Raise
    ------
    ValueError: if signal or list_changing_points are not the expected types.


    Returns
    -------
    final_signal: list. Signal corrected.

    """
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"'signal' must be a np array but got {type(signal)}.")
    if signal.size==0 or len(signal)<5:
        raise ValueError("'signal must be a non empty np array with at least five points.")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("'signal' cannot contain nans or infinite values.")
    if not isinstance(list_changing_points, list):
        raise ValueError(f"'list_changing_points' must be a list but got type{list_changing_points}.")
    if not all(isinstance(number, int) for number in list_changing_points):
        raise ValueError(f"All elements in 'list_changing_points' must be integers, but got {list_changing_points}.")
    if not np.all(list_changing_points==np.sort(list_changing_points)):
        raise ValueError("'list_changing_points' must be ordered.")
    if not all(0 <= changing_point <= len(signal) for changing_point in list_changing_points):
        raise ValueError("'list_changing_points' must contain values between 0 and len(signal).")
        
    final_signal = signal.copy()
    for i in range(1, (len(list_changing_points)-1)):
        if i==1:
            point_a=0
        else:
            point_a=list_changing_points[i-2]
        
        point_b=list_changing_points[i-1]
        point_c = list_changing_points[i]
        point_d=list_changing_points[i+1]

        local_signal_before = final_signal[point_a:point_b]
        local_signal = final_signal[point_b:point_c]
        local_signal_after= final_signal[point_c:point_d]
        
        
        # correction when time series decreases
        if (np.median(local_signal) < np.median(local_signal_before)):
            # 3 section correct
            # steady - too small - steady ->
            # the too small value becomes the steady value before
            if np.median(local_signal_before) == np.median(local_signal_after):
                final_signal[point_b:point_c] = np.median(local_signal_before)
            # steady - too small- bigger than steady ->
            # the too small values become the steady value before
            elif np.median(local_signal_before) < np.median(local_signal_after):
                final_signal[point_b:point_c] = np.median(local_signal_before)
            # 4 section correction
            elif i<(len(list_changing_points)-2):
                last_last_point=list_changing_points[i+2]
                local_signal_after_after=final_signal[point_d:last_last_point]
                # steady - too small - too small - steady
                if (np.median(local_signal_before) == np.median(local_signal_after_after)):
                    final_signal[point_b:point_c] = np.median(local_signal_before)
                # steady - too small - too small - bigger legit
                elif (np.median(local_signal_before) < np.median(local_signal_after_after)):
                    final_signal[point_b:point_c] = np.median(local_signal_before)
        
    return final_signal
                    
                    
                    
def correct_last_section(signal, which_algorithm="pelt",\
                         which_model="l2", pen=0):
    """
    Context: time series correction monotonic.
    Function to correct the last segment of the time series if it has a value
    inferior to the value of the segment before.
    The assumption is that this time series has been already flattened out for oscillations.
    Therefore, by running the appropriate algorithm for finding the changing points, the algorithms considers the last two segments.

    Parameters
    ----------
    signal : np array. Signal to correct.
    which_algorithm: which algorithm to use for finding the change points.
        Default value: "pelt". For details, refer to "ruptures" library documentation.
    which_model: which model to use for finding the change points.
        Default value: "l2". For details, refer to "ruptures" library documentation.
    pen: which penalty to use for finding the change points.
        Default value: 0. For details, refer to "ruptures" library documentation.
    
    Raise
    ------
    ValueError: if signal is not the expected time, 
        if empty or not of the required length,
        if it contains nan or infinite values.
    
    Functions
    ------
    algorithm_change_point_detection: function to find the change points detection.
    This function is defined in the function_postecorrection.py file.

    Returns
    -------
    final_signal : np array. Signal corrected.

    """
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"'signal' must be a np array but got {type(signal)}.")
    if signal.size==0 or len(signal)<3:
        raise ValueError("'signal must be a non empty np array with at least three points.")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("'signal' cannot contain nans or infinite values.")
        
    list_changing_points = algorithm_change_point_detection(signal,\
        which_algorithm, which_model, penalty=pen)
    if len(list_changing_points) < 3:
        raise IndexError(f'The algorithm was not able to find at least 3 points but only {len(list_changing_points)}.')
    final_signal=signal.copy()
    thirdtolast_cp=list_changing_points[-3]
    penultimate_cp=list_changing_points[-2]
    last_cp = list_changing_points[-1]
    local_signal_before_last = final_signal[thirdtolast_cp:penultimate_cp]
    local_signal_last= final_signal[penultimate_cp:last_cp]
    
    value_before_last_segment = np.median(local_signal_before_last)
    value_last_segment = np.median(local_signal_last)
    if (not np.all(value_before_last_segment == local_signal_before_last)) or\
       (not np.all(value_last_segment == local_signal_last)):
        raise ValueError("The segments are not flattened as expected as they contain variations.")

    if value_before_last_segment > value_last_segment:
        final_signal[penultimate_cp:last_cp]=value_before_last_segment
        
    return final_signal
            

def find_points_to_flag(signal):
    """
    Function to find the points where the value of the time series decreases.

    Parameters
    ----------
    signal : numpy array. Input signal.
    
    Raise
    -----
    ValueError if 'signal' is not the expected type, size or if it contains nan or infinite values.

    Returns
    -------
    flag_points: list. Points where the signal decreases.

    """
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"'signal' must be a np array but got {type(signal)}.")
    if signal.size==0 or len(signal)<2:
        raise ValueError("'signal must be a non empty np array with at least two elements.")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("'signal' cannot contain nans or infinite values.")
        
    flag_points = [i for i in range(len(signal)-1) if signal[i] > signal[i+1]]

    return flag_points

def correction_CPD(signal, times,  which_algorithm, which_model,\
                   pen=2, flag_first_correction=0, plot=0, row=None):
    """
    Function to correct time series in order to make it as monotonic as possible.
    Parameters
    ----------
    signal : np array.
    times : np array.
    which_algorithm : which algorithm to use for change point detection. 
        For details, refer to the "ruptures" library documentation.
    which_model : which model to use for change point detection. 
        For details, refer to the "ruptures" library documentation..
    pen : penalty to use for change point detection.
        For details, refer to the "ruptures" library documentation.
        The default value is 2.
    flag_first_correction : flag fo applying shallow correction. 
        The default is 0.
    plot : flag for enabling the plotting of the results. The default is 1.
    row : pandas series. Information about network, well and training. The default is None.

    Raises
    ------
    ValueError: if 'signal' or 'times' or 'row' (if not None) are not in the expected format.
    KeyError: if 'row' (if not None) does not contain the expected keys: Network, Well, Training.
    
    Returns
    -------
    new_signal : np array. First signal after first shallow correction. It returns the original 'signal' if no correction is applied.
    final_signal : np array. Final signal after neighbors correction.
    flag_points : list. Points where the time series decreases even after the corrections.

    """
    if not isinstance(signal, np.ndarray):
        raise ValueError(f"'signal' must be a np array but got {type(signal)}.")
    if signal.size==0 or len(signal)<5:
        raise ValueError("'signal must be a non empty np array with at least five points.")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        raise ValueError("'signal' cannot contain nans or infinite values.")
    if not isinstance(times, np.ndarray):
        raise ValueError(f"'times' must be a np array but got {type(times)}.")
    if times.size==0 or len(times)<2:
        raise ValueError("'times' must be a non empty np array with at least two points.")
    if np.any(np.isnan(times)) or np.any(np.isinf(times)):
        raise ValueError("'times' cannot contain nans or infinite values.")
    if signal.shape != times.shape:
        raise ValueError(f"'signal' and 'times' must have the same shape but got signal shape: {signal.shape} and times shape: {times.shape}")
    if row is not None:
        if not isinstance(row, (pd.core.series.Series, dict)):
            raise ValueError(f"'row' must be a pandas series or dictionary but got {type(row)}.")
        if "Network" not in row.keys():
            raise KeyError("'row' doest not have a key 'Network'.")
        if "Well" not in row.keys():
            raise KeyError("'row' doest not have a key 'Well'.")
        if "Training" not in row.keys():
            raise KeyError("'row' doest not have a key 'Training'.")
        
    # shallow correction
    if flag_first_correction:
        first_correction = curve_correction_v2(signal)
    else:
        first_correction=signal.copy()

    # change point detection
    list_changing_points = algorithm_change_point_detection(first_correction,\
                                              which_algorithm, which_model, pen)
    #breakpoint()
    # flatten regions
    new_signal=curve_correction_statistics(signal, list_changing_points)
    #breakpoint()
    
    # this is if the signal can only increase
    # change point detection 
    #result = algorithm_change_point_detection(new_signal,\
    #                                          "pelt", "l2", penalty=0)
    # correct based on neighbors
    #final_signal = correct_based_on_neighbors(new_signal, result)
    
    # correction of the last section if the before-to-last section
    # is bigger than last section
    #final_signal = correct_last_section(final_signal, which_algorithm="pelt",\
    #                     which_model="l2", pen=0)
    # points to flag because signal decreases   
    import matplotlib.pyplot as plt

    final_signal=new_signal.copy()
    flag_points = find_points_to_flag(final_signal)
        
    if plot:
        plt.figure()
        plt.plot(times, signal, linewidth=1, label="Output YOLO")
        plt.plot(times, final_signal, linewidth=3, label="After correction")
        plt.vlines(x = flag_points, ymin=0, ymax=max(final_signal.max(), signal.max()), color = 'black',\
                    linestyle='dashed')
        plt.xlabel("Number of frame")
        plt.ylabel("Number of cells")  
        plt.legend()
        
        if row is not None:
            plt.title(f"{row['Network']}, {row['Well']}, {row['Training']}")
           
    
        plt.show()

    
    return new_signal, final_signal, flag_points


def extract_position(string):
    """
    Extracts the substring starting with 'Pos' from the input string using regex.

    Parameters
    ----------
    string : str
        The input string to search.

    Raises
    ------
    TypeError
        If the input is not a string.

    Returns
    -------
    str
        The substring starting with 'Pos' if found; otherwise, returns the original string.
    """
    if not isinstance(string, str):
        raise TypeError(f"Input must be a string, got {type(string).__name__} instead.")

    match = re.search(r'(Pos.*)', string)
    return match.group(1) if match else string


def extract_gt(name_experiment, name_series, name_well, times):
    """
    Extract groundtruth corresponding to the experiment, position and well.

    Parameters
    ----------
    name_experiment : string
    name_series : string
    name_well : string
    times : np.ndarray

    Returns
    -------
    gt : np.ndarray or None.

    """
    if os.path.exists(f"../Groundtruth/{name_experiment}.csv"):
        file_gt = pd.read_csv(f"../Groundtruth/{name_experiment}.csv", sep=";")
        # extract position from string
        name_position = extract_position(name_series)
        position_suffix = name_position.split("Pos_")[1]
        name_column = f"{position_suffix}_{name_well.upper()}"
        if name_column not in file_gt.columns:
            matching_columns = [col for col in file_gt.columns if position_suffix in col and name_well in col]
            if not matching_columns:
                print(f"Groundtruth: no matching column found for position '{position_suffix}' and well '{name_well}'")
                return None
            if len(matching_columns) > 1:
                print(f"Groundtruth: more than one matching column found for position '{position_suffix}' and well '{name_well}'") 
                return None
            name_column = matching_columns[0] 
        gt=np.array(file_gt[name_column][times]).astype(np.uint16)
    else:
        gt=None
        print("Groundtruth not found.")
    return gt
    
    
def plot_curve(times, signal, final_signal,\
               name_experiment, name_series, name_well, dir_results, gt=None):
    """
    Plot resulting curve.

    Parameters
    ----------
    times : TYPE
        DESCRIPTION.
    signal : TYPE
        DESCRIPTION.
    final_signal : TYPE
        DESCRIPTION.
    name_experiment : TYPE
        DESCRIPTION.
    name_series : TYPE
        DESCRIPTION.
    name_well : TYPE
        DESCRIPTION.
    gt : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt

    if not isinstance(signal, np.ndarray):
        raise TypeError(f"'signal' must be a np array but got {type(signal)}.")
    if final_signal is not None and not isinstance(final_signal, np.ndarray):
        raise TypeError(f"'final_signal' must be a np array but got {type(final_signal)}.")
    if gt is not None and not isinstance(gt, np.ndarray):
        raise TypeError(f"'gt' must be a np array but got {type(gt)}.")
    # extract position from string
    name_position = extract_position(name_series)
    # points to flag because signal decreases   
    if final_signal is None:
        final_signal = signal.copy()
    flag_points = find_points_to_flag(final_signal)
    # plot
    plt.figure()
    plt.plot(times, signal, linewidth=1, label="Output YOLO")
    plt.plot(times, final_signal, linewidth=3, label="After correction")
    plt.vlines(x = flag_points, ymin=0, ymax=max(final_signal.max(), signal.max()), color = 'black',\
                linestyle='dashed')
    if gt is not None:
        plt.plot(times, gt, linewidth=2, label="Groundtruth")
    plt.xlabel("Number of frame")
    plt.ylabel("Number of cells")  
    plt.legend()
    plt.title(f"{name_experiment} {name_position} {name_well}")
    
    plt.savefig(os.path.join(dir_results, f"fig_{name_experiment}_{name_position}_{name_well}.png"), dpi=300, bbox_inches='tight')  
    #plt.show()
    plt.close("all")
    
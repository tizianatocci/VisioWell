# -*- coding: utf-8 -*-
"""
Functions to compute oscillation score (in any case) and performances (when groundtruth is available).
Author: Tiziana Tocci
"""

import numpy as np


def compute_performance(result_num, gt_num):
    """
    Compute performances with respect to the ground truth (manual annotation of the cell count)

    Parameters
    ----------
    result_num : np.ndarray.
    gt_num : np.ndarray.

    Returns
    -------
    perf_mb : float.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(gt_num, np.ndarray):
        raise TypeError("The groundtruth 'gt_num' must be a np.ndarray.")
    if not result_num.shape==gt_num.shape:
        raise ValueError("The arguments 'result_num' and 'gt_num' must have the same shape.")
    
    n_frames = result_num.shape[0]
    perf_mb=0
    for n in range(n_frames):
        if not np.isnan(gt_num[n]):
            perf_mb = perf_mb+abs(gt_num[n]-result_num[n])/gt_num[n]
    perf_mb=round(1-1/n_frames*perf_mb, 3)
    
    return perf_mb

def column_diff(gt_num, result_num):
    """
    Compute the difference between two numpy arrays.

    Parameters
    ----------
    gt_num : np.ndarray
    result_num : np.ndarray

    Raises
    ------
    TypeError
        If result_num or gt_num are not numpy arrays.
    ValueError
        If result_num and gt_num do not have the same shape.
        
    Returns
    -------
    column_errors : np array. The difference between the two inputs.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(gt_num, np.ndarray):
        raise TypeError("The groundtruth 'gt_num' must be a np.ndarray.")
    if not result_num.shape==gt_num.shape:
        raise ValueError("The arguments 'result_num' and 'gt_num' must have the same shape.")
    column_errors = gt_num-result_num
    return column_errors

def compute_index_exact(result_num, gt_num, column_errors=None):
    """
    Index of exact correpondance values across result_num and gt_num.
    Compute performances with respect to the ground truth (manual annotation of the cell count)
    If the difference between gt_num and result_num is 0, then the value for the frame is 1.
    Sum all the values obtained and divide by the number of frames.
    
    Parameters
    ----------
    n_frames : int. number of frames.
    result_num : np.ndarray.
    gt_num : np.ndarray.
    column_errors: np.ndarray. Default value: None.

    Raises
    ------
    TypeError
        If result_num or gt_num or column_errors (if given) are not numpy arrays.
    ValueError
        If result_num and gt_num and column_errors do not have the same shape.
        
    Returns
    -------
    perf_mb : float.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(gt_num, np.ndarray):
        raise TypeError("The groundtruth 'gt_num' must be a np.ndarray.")
    if not result_num.shape==gt_num.shape:
        raise ValueError("The arguments 'result_num' and 'gt_num' must have the same shape.")
    if column_errors is not None and not isinstance(column_errors, np.ndarray):
        raise TypeError("The argument 'column_errors' must be a np.ndarray.")
    if column_errors is not None and not column_errors.shape==gt_num.shape:
        raise ValueError("The arguments 'column_errors' and 'gt_num' must have the same shape.")
            
    if column_errors is None:
        column_errors = column_diff(gt_num, result_num)
        
    n_frames = result_num.shape[0]
    perf_mb=0
    for n in range(n_frames):
        if not np.isnan(gt_num[n]) and column_errors[n] == 0:
            perf_mb += 1
    perf_mb=round(1/n_frames*perf_mb, 4)
    
    return perf_mb

def compute_index_exact_until_nine(result_num, gt_num, column_errors=None):
    """
    Index of exact correpondance values across result_num and gt_num until the second
    array reaches the value of 9 (excluded).
    Compute performances with respect to the ground truth (manual annotation of the cell count)
    If the difference between gt_num and result_num is 0, then the value for the frame is 1.
    Sum all the values obtained and divide by the number of frames.
    
    Parameters
    ----------
    n_frames : int. number of frames.
    result_num : np.ndarray.
    gt_num : np.ndarray.
    column_errors: np.ndarray. Default value: None.

    Raises
    ------
    TypeError
        If result_num or gt_num or column_errors (if given) are not numpy arrays.
    ValueError
        If result_num and gt_num and column_errors do not have the same shape.
        
    Returns
    -------
    perf_mb : float.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(gt_num, np.ndarray):
        raise TypeError("The groundtruth 'gt_num' must be a np.ndarray.")
    if not result_num.shape==gt_num.shape:
        raise ValueError("The arguments 'result_num' and 'gt_num' must have the same shape.")
    if column_errors is not None and not isinstance(column_errors, np.ndarray):
        raise TypeError("The argument 'column_errors' must be a np.ndarray.")
    if column_errors is not None and not column_errors.shape==gt_num.shape:
        raise ValueError("The arguments 'column_errors' and 'gt_num' must have the same shape.")
            
    
    idx_less_nine = gt_num < 9
    result_num_less_nine = result_num[idx_less_nine]
    gt_num_less_nine = gt_num[idx_less_nine]
    
    if column_errors is not None:
        column_errors = column_errors[idx_less_nine]
        
    perf_mb = compute_index_exact(result_num_less_nine, gt_num_less_nine,\
                                  column_errors=column_errors)
    
    return perf_mb

def compute_std(result_num, gt_num):
    """
    Compute standard deviation of difference between two numpy arrays.

    Parameters
    ----------
    result_num : np.ndarray.
    gt_num : np.ndarray.

    Raises
    ------
    TypeError
        If result_num or gt_num are not numpy arrays.
    ValueError
        If result_num and gt_num do not have the same shape.
        

    Returns
    -------
    index_std : float.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(gt_num, np.ndarray):
        raise TypeError("The groundtruth 'gt_num' must be a np.ndarray.")
    if not result_num.shape==gt_num.shape:
        raise ValueError("The arguments 'result_num' and 'gt_num' must have the same shape.")
    
    index_std = round(np.std(result_num-gt_num), 4)
    
    return index_std
    
def compute_std_until_nine(result_num, gt_num):
    """
    Compute standard deviation of difference between two numpy arrays until the second
    array reaches the value of 9 (excluded).
    
    Parameters
    ----------
    result_num : np.ndarray.
    gt_num : np.ndarray.

    Raises
    ------
    TypeError
        If result_num or gt_num are not numpy arrays.
    ValueError
        If result_num and gt_num do not have the same shape.
        

    Returns
    -------
    index_std_nine_cells : float.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(gt_num, np.ndarray):
        raise TypeError("The groundtruth 'gt_num' must be a np.ndarray.")
    if not result_num.shape==gt_num.shape:
        raise ValueError("The arguments 'result_num' and 'gt_num' must have the same shape.")
    
    idx_less_nine = gt_num < 9
    
    result_num_less_nine = result_num[idx_less_nine]
    gt_num_less_nine = gt_num[idx_less_nine]
    
    index_std_nine=compute_std(result_num_less_nine, gt_num_less_nine)
        
    return index_std_nine


def compute_oscillation_score(result_num):
    """
    Function to compute how many oscillations there are before correction.

    Parameters
    ----------
    result_num : np ndarray. Signal before correction.
    
    Raises
    ------
    TypeError if result_num is not a np ndarray.

    Returns
    -------
    oscillation_score : float. Between 0 and 1. The closest to 0, the best the quality is. The closest to 1, the worst the quality is.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
        
    oscillation_score = 0
    for i in range(1,len(result_num)):
        previous_count = result_num[i-1]
        current_count = result_num[i]
        if current_count != previous_count:
            oscillation_score  = oscillation_score + 1
    oscillation_score = round(oscillation_score/len(result_num),4)
    
    return oscillation_score
            

def compute_oscillation_score_until_nine(result_num, corrected_num):
    """
    Function to compute how many oscillations there are before correction.
    The frames considered are those until the corrected signal shows a count of nine (extreme excluded).
    

    Parameters
    ----------
    result_num : np ndarray. Signal before correction.
    corrected_num: np ndarray. Signal after correction.
    
    Raises
    ------
    TypeError
        If result_num or corrected_num are not numpy arrays.
    ValueError
        If result_num and corrected_num do not have the same shape.
        

    Returns
    -------
    oscillation_score : float. Between 0 and 1. The closest to 0, the best the quality is. The closest to 1, the worst the quality is.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(corrected_num, np.ndarray):
        raise TypeError("The argument 'corrected_num' must be a np.ndarray.")
    if not result_num.shape==corrected_num.shape:
        raise ValueError("The arguments 'result_num' and 'corrected_num' must have the same shape.")
    
    idx_less_nine = corrected_num < 9
    
    result_num_less_nine = result_num[idx_less_nine]    

    oscillation_score = compute_oscillation_score(result_num_less_nine)
    
    return oscillation_score
            


def compute_oscillation_score_height(result_num):
    """
    Function to compute how many oscillations, taking the height into account, there are before correction.

    Parameters
    ----------
    result_num : np ndarray. Signal before correction.
    
    Raises
    ------
    TypeError if result_num is not a np ndarray.

    Returns
    -------
    oscillation_score : float. It can be bigger than 1.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
        
    oscillation_score = 0
    for i in range(1,len(result_num)):
        previous_count = result_num[i-1]
        current_count = result_num[i]
        if current_count != previous_count:
            oscillation_score  = oscillation_score + abs(current_count-previous_count)
    oscillation_score = round(oscillation_score/len(result_num),4)
    
    return oscillation_score
            
    

def compute_oscillation_score_height_until_nine(result_num, corrected_num):
    """
    Function to compute how many oscillations there are before correction.
    The frames considered are those until the corrected signal shows a count of nine (extreme excluded).
    

    Parameters
    ----------
    result_num : np ndarray. Signal before correction.
    corrected_num: np ndarray. Signal after correction.
    
    Raises
    ------
    TypeError
        If result_num or corrected_num are not numpy arrays.
    ValueError
        If result_num and corrected_num do not have the same shape.
        

    Returns
    -------
    oscillation_score : float. Between 0 and 1. The closest to 0, the best the quality is. The closest to 1, the worst the quality is.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(corrected_num, np.ndarray):
        raise TypeError("The argument 'corrected_num' must be a np.ndarray.")
    if not result_num.shape==corrected_num.shape:
        raise ValueError("The arguments 'result_num' and 'corrected_num' must have the same shape.")
    
    idx_less_nine = corrected_num < 9
    
    result_num_less_nine = result_num[idx_less_nine]    
    
    oscillation_score = compute_oscillation_score_height(result_num_less_nine)
    
    return oscillation_score
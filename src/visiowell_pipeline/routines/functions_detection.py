# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:34:08 2024

@author: tizia
"""
import numpy as np
import os
from ..routines.functions_performances import compute_index_exact, compute_index_exact_until_nine,\
    compute_std, compute_std_until_nine,\
    compute_oscillation_score, compute_oscillation_score_until_nine,\
    compute_oscillation_score_height, compute_oscillation_score_height_until_nine
import pandas as pd
from ..routines.functions_postcorrection import extract_position
from ..routines.functions_general import load_pickle_file, write_file
from ..routines.functions_postcorrection import correction_CPD, plot_curve, extract_gt
import multiprocessing
import traceback
from datetime import datetime, date
import ast
from ..classes.classes_detection import CellDetector
#tf.config.run_functions_eagerly(True)


"""
### Save files
"""
def obtain_csv(num_cell_test2, final_signal,\
               dir_results, name_experiment, name_series, name_well, gt=None):
    if not isinstance(num_cell_test2, np.ndarray):
        raise TypeError("The argument 'num_cell_test2' must be np.ndarray.")
    if not isinstance(final_signal, np.ndarray):
        raise TypeError("The groundtruth 'final_signal' must be a np.ndarray.")
    if not num_cell_test2.shape==final_signal.shape:
        raise ValueError("The arguments 'num_cell_test2' and 'final_signal' must have the same shape.")
    if gt is not None and not isinstance(gt, np.ndarray):
        raise TypeError("The argument 'gt' must be a np.ndarray.")
    if gt is not None and not gt.shape==final_signal.shape:
        raise ValueError("The arguments 'gt' and 'final_signal' must have the same shape.")
    
    oscillation_score = compute_oscillation_score(num_cell_test2)
    oscillation_score_height = compute_oscillation_score_height(num_cell_test2)
    oscillation_score_less = compute_oscillation_score_until_nine(num_cell_test2, final_signal)
    oscillation_score_height_less = compute_oscillation_score_height_until_nine(num_cell_test2, final_signal)

    if gt is not None:
        indic_exact = compute_index_exact(final_signal, gt)
        indic_exact_less = compute_index_exact_until_nine(final_signal, gt)
        indic_std = compute_std(final_signal, gt)
        indic_std_less = compute_std_until_nine(final_signal, gt)
        # compute: performance, std(errors), std(errors until gt=9)
    output={"Output YOLO": num_cell_test2, "Correction": final_signal,
            "Groundtruth": gt if gt is not None else np.nan,
            "Indic_exact": indic_exact if gt is not None else np.nan,
            "Indic_exact-9": indic_exact_less if gt is not None else np.nan,
            "Indic_std": indic_std if gt is not None else np.nan,
            "Indic_std-9": indic_std_less if gt is not None else np.nan,
            "Osc_index": oscillation_score,
            "Osc_height_index": oscillation_score_height,
            "Osc_index-9": oscillation_score_less,
            "Osc_height_index-9": oscillation_score_height_less}
    df = pd.DataFrame(output)
    
    name_position = extract_position(name_series)
    df.to_csv(os.path.join(dir_results, f"output_{name_experiment}_{name_position}_{name_well}.csv"), sep=";", index=False)
    return df


def save_matrix_centers(matrix_centers,masks, \
                       name_experiment, name_series, name_well, dir_results):
    import pickle
    name_position = extract_position(name_series)
    name_file_to_save = f"cellinfo{name_experiment}_{name_position}_{name_well}.pkl" # name of file to save
    filename = os.path.join(dir_results, name_file_to_save)
    output = {"matrix_centers": matrix_centers, "masks": masks}
    
    with open(filename, 'wb') as file:
        pickle.dump(output, file)
        


"""
### Summary functions
"""    
def retrieve_configs(CD, Nframe, model, video, flag_plot, type_network):
    if type_network == "yolo_tf":
        def detect_fn():
            return CD.detect_cells(Nframe, model, video, start_frame=0, std_normal=0.1, flag_plot=flag_plot)
        def save_args(output):
            return output["matrix_centers"], output["masks_colors"]
        return {"detect_fn": detect_fn, "save_args": save_args}

    elif type_network in ("cellpose3", "cellpose4"):
        def detect_fn():
            return CD.detect_cells(model=model, imgs=video)
        def save_args(output):
            return None, output["masks"]
        return {"detect_fn": detect_fn, "save_args": save_args, "print": "Prediction..."}

    else:
        raise ValueError(f"Unknown network type: {type_network}")
    
        

def analyze_microwell(type_network, name_series, name_well, flag_plot,\
                      name_experiment, dict_param, model, path_exp=None, flag_gt=None, dir_results=None):
    from ..classes.classes_detection import NetworkLoader
    if not isinstance (name_series, str):
        raise TypeError(f"{name_series} must be a list but got {type(name_series)}")
    if not isinstance (name_well, str):
        raise TypeError(f"{name_well} must be a list but got {type(name_well)}")
    
    print(f"Analyzing {name_series}_{name_well}")
    """
    ### Directories
    """
    name_file_to_load = f"{name_series}.pkl" # name of file to load
    if not path_exp:
        path_exp = "../Wells"     # path of the experiment to analyse
    if not dir_results:
        dir_results = "../Results" # path of results folder

    """
    Load series and well 
    """
    # load specific series and specific well
    x = load_pickle_file(path_exp, name_file_to_load)
    well_data=x[name_well]
    del x
    video = well_data["crop"]
    del well_data

    # Start detection
    CD = CellDetector(type_network)
    CD.retrieve_parameters_for_detection(dict_param)
    Nframe, video = CD.prepare_data(video)
    flag_goodwell = CD.evaluate_first_frame(model, video)
    
    if flag_goodwell or flag_gt:
        """
        ### Detection in all frames.
        """
        cfg = retrieve_configs(CD, Nframe, model, video, flag_plot, type_network) 
        if "print" in cfg:
            print(cfg["print"])
        output = cfg["detect_fn"]()
        if "print" in cfg:
            print("Prediction done")
        matrix_centers, masks = cfg["save_args"](output)
        save_matrix_centers(matrix_centers, masks, name_experiment, name_series, name_well, dir_results)
        del matrix_centers, masks
        
        num_cell_test2 = output["num_cell_test2"]
        
        """
        ### Curve correction
        """
        signal = num_cell_test2.copy()
        times = np.arange(Nframe)
        _, final_signal, _ = correction_CPD(signal, times,\
                                            "pelt", "l2", pen=2)
          
        """
        ### Save output
        """ 
        # Extract ground truth if it exists
        gt=extract_gt(name_experiment, name_series, name_well, times)
        # Plot curve
        plot_curve(times, signal, final_signal,\
                   name_experiment, name_series, name_well, dir_results, gt=gt)
        # Save .csv file with cell counting and performances
        df=obtain_csv(num_cell_test2, final_signal,\
        dir_results, name_experiment, name_series, name_well, gt)
        
        del video
        return df
    else:
        del video
        return None
        

def append_log_error_for_microwell(name_series, name_well, trace, error_log_file):
    if not isinstance(name_series, str):
        raise TypeError(f"name_series must be a string but got {type(name_series)}")
    if not isinstance(name_well, str):
        raise TypeError(f"name_well must be a string but got {type(name_well)}")
    if not isinstance(error_log_file, str):
        raise TypeError(f"error_log_file must be a string but got {type(error_log_file)}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = (
        f"[{timestamp}] Error in series {name_series}, well '{name_well}':\n"
        f"{trace}\n"
    )

    with open(error_log_file, "a") as log:
        log.write(error_message)

import multiprocessing as mp
import traceback


def run_on_gpu(which_network, positions,\
               flag_plot,\
               name_experiment, name_series_list=None, name_well_list=None, training_to_load = None,\
               path_exp=None, flag_gt=None, dir_results=None):
    if positions is not None and not isinstance(positions,list):
        raise TypeError(f"'positions' must be a list but got {type(positions)}")
    if name_series_list is not None and not isinstance (name_series_list, list):
        raise TypeError(f"'name_series_list' must be a list but got {type(name_series_list)}")
    if name_well_list is not None and not isinstance (name_well_list, list):
        raise TypeError(f"'name_well_list' must be a list but got {type(name_well_list)}")
    if name_series_list is not None and name_well_list is not None and not len(name_series_list)==len(name_well_list):
        raise ValueError("Error: name_series_list and name_well_list must have the same length.")
    if (name_well_list is None and name_series_list is not None) or (name_well_list is not None and name_series_list is None):
        raise ValueError("Error: 'name_well_list' and 'name_series_list' must be either both None or both defined.")
    if positions is None and (name_series_list is None or name_well_list is None):
        raise ValueError("Error: If 'positions' is None, then both 'name_series_list' and 'name_well_list' must be defined.")
    if positions is not None and name_series_list is not None and name_well_list is not None:
        raise ValueError("Error: 'positions' must be defined only when 'name_series_list' and 'name_well_list' are None.")

    from ..classes.classes_detection import NetworkLoader

    GL = NetworkLoader()
    dict_param = GL.load_parameters(which_network, training_to_load=training_to_load)
    model = GL.load_network(which_network, dict_param)

    if name_series_list is None and name_well_list is None and positions is not None:
        print(f"Analyzing the positions: {positions}")
        name_series_list, name_well_list = retrieve_series_and_wells(path_exp, positions)
        
    for i in range(len(name_series_list)):
        name_series=name_series_list[i]
        name_well=name_well_list[i]
        status, result = safe_analyze_microwell(which_network, name_series, name_well,\
                                            flag_plot,
                                            name_experiment, dict_param, model, path_exp, flag_gt, dir_results)
        suffix = name_series.split("sherlock_", 1)[1].split("_tile", 1)[0]

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(dir_results, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        error_log_file = f"{logs_dir}/microwells_errors_{date.today().strftime('%Y%m%d')}_{suffix}.txt"
        #output_log_file = f"{logs_dir}/microwells_valid_{date.today().strftime('%Y%m%d')}_{suffix}.txt"

        if status == "error":
            append_log_error_for_microwell(
            result["name_series"],
            result["name_well"],
            result["traceback"],
            error_log_file)
        elif status == "success":
            pass
            #write_file(output_log_file, result["name_series"] + result["name_well"], is_path=0)


def log_memory(tag=""):
    import torch, psutil, os, time
    gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    gpu_cached = torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
    ram = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[{time.strftime('%H:%M:%S')}] {tag} | GPU alloc={gpu_mem:.1f}MB reserved={gpu_cached:.1f}MB | RAM={ram:.1f}MB", flush=True)
          
def safe_analyze_microwell(which_network, name_series, name_well,\
                      flag_plot,\
                      name_experiment, dict_param, model, path_exp=None, flag_gt=None, dir_results=None):
    import torch
    import gc
    try:
        #log_memory(f"Before {name_series}_{name_well}")
        _=analyze_microwell(which_network, name_series, name_well,\
                              flag_plot,\
                              name_experiment, dict_param, model, path_exp, flag_gt, dir_results)
        #log_memory(f"After {name_series}_{name_well}")
        return ("success", {"name_series": name_series, "name_well": name_well})
    except Exception as e:
        print(f"Error for {name_series}, {name_well}: {e}")
        return ("error", {
            "name_series": name_series,
            "name_well": name_well,
            "traceback": traceback.format_exc()
        })
    except BaseException as e:
        print(f"Critical error for {name_series}, {name_well}: {e}")
        return ("critical_error", {
            "name_series": name_series,
            "name_well": name_well,
            "traceback": traceback.format_exc()
        })
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    
def run_on_cpu(which_network, positions, model,\
    dict_param, flag_plot,\
    name_experiment, name_series_list=None, name_well_list=None):
    """
    Function for parallelizing the analysis of microwell on CPU.
    

    """
    if positions is not None and not isinstance(positions,list):
        raise TypeError(f"'positions' must be a list but got {type(positions)}")
    if name_series_list is not None and not isinstance (name_series_list, list):
        raise TypeError(f"'name_series_list' must be a list but got {type(name_series_list)}")
    if name_well_list is not None and not isinstance (name_well_list, list):
        raise TypeError(f"'name_well_list' must be a list but got {type(name_well_list)}")
    if name_series_list is not None and name_well_list is not None and not len(name_series_list)==len(name_well_list):
        raise ValueError("Error: name_series_list and name_well_list must have the same length.")
    if (name_well_list is None and name_series_list is not None) or (name_well_list is not None and name_series_list is None):
        raise ValueError("Error: 'name_well_list' and 'name_series_list' must be either both None or both defined.")
    if positions is None and (name_series_list is None or name_well_list is None):
        raise ValueError("Error: If 'positions' is None, then both 'name_series_list' and 'name_well_list' must be defined.")
    if positions is not None and name_series_list is not None and name_well_list is not None:
        raise ValueError("Error: 'positions' must be defined only when 'name_series_list' and 'name_well_list' are None.")

    # get the number of available CPU cores dynamically
    num_cores = multiprocessing.cpu_count()
    num_cores_to_use = max(1, int(num_cores/3))
    print(f"Using {num_cores_to_use} CPU cores for processing.")
    
    # if name_series_list and name_well_list are passed, then analyze each combination of name_series and name_well
    # else, find the wells for each position
    if name_series_list is None and name_well_list is None and positions is not None:
        print(f"Analyzing the positions: {positions}")
        name_series_list, name_well_list = retrieve_series_and_wells(positions)
          
    # Prepare the arguments to pass to pool.starmap
    video_positions_with_args = [(which_network, name_series, name_well,\
                                  model,\
                                  dict_param, flag_plot,\
                                  name_experiment) for name_series, name_well in zip(name_series_list, name_well_list)]   
    # create a pool of processes with the detected number of cores
    with multiprocessing.Pool(num_cores_to_use) as pool:
        results=pool.starmap(safe_analyze_microwell, video_positions_with_args)
    
    error_log_file = f"../Logs/microwells_errors_{date.today().strftime('%Y%m%d')}.txt"
    output_log_file = f"../Logs/microwells_valid_{date.today().strftime('%Y%m%d')}.txt"
    
    for res in results:
        if res[0] == "error":
            err = res[1]
            append_log_error_for_microwell(
                err["name_series"],
                err["name_well"],
                err["traceback"],
                error_log_file)
        elif res[0] == "success":
            write_file(output_log_file, res[1]["name_series"] + res[1]["name_well"], is_path=0)


def retrieve_all_positions(path_exp):
    """
    Function to retrieve all positions to analyse. For each position, all the microwells will be analysed.

    Returns
    -------
    positions : list
        Unique positions retrieved from the folder.
    """
    pkl_files = [f for f in os.listdir(path_exp) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in '{path_exp}'")
    
    positions = [f.replace('.pkl', '').replace('sherlock_', '', 1) for f in pkl_files]
    positions = list(dict.fromkeys(positions))

    return positions


def retrieve_name_wells_for_position(path_exp, name_pos):
    if not isinstance(name_pos, str):
        raise TypeError("name_pos must be a string but got {type(name_pos)}")
   
    file_path = os.path.join(path_exp, f"listwells_{name_pos}.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'{file_path}' does not exist. Please check the path.")
    
    name_well_list = None
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("name_well_list"):
                name_well_list = ast.literal_eval(line.split('=', 1)[1])
                break
    
    if name_well_list is None:  # Handle case where the expected variable wasn't found
            raise ValueError(f"'name_well_list' not found in file: {file_path}")

    return name_well_list


def retrieve_series_and_wells(path_exp, positions):
    if not isinstance(positions,list):
        raise TypeError("'positions' must be a list but got  {type(positions)}")
    if not all(isinstance(pos, str) for pos in positions):
        raise TypeError("All elements in 'positions' must be strings")
    
    name_series_list=[]
    name_well_list=[]
    for full_name_pos in positions:
        if "_" not in full_name_pos:
            raise ValueError(f"Invalid position format: '{full_name_pos}', expected at least one underscore (_)")
        name_pos = full_name_pos.split('_', 1)[1]
        wells = retrieve_name_wells_for_position(path_exp, name_pos)
        if wells:
            name_series_list += [full_name_pos] * len(wells)
            name_well_list += wells
    return name_series_list, name_well_list
                
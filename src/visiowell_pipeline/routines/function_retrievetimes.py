import os
import pandas as pd
from ..classes.classes_times_qc import TimeRetriever
import traceback
import numpy as np
from datetime import date

def retrieve_times(result_file, dir_results,\
                   imaging_frequence, cell_type,\
                   time_interval_discrepancy, colony_size_limit,\
                   TR):
    """
    Process a single CSV result file for a microwell experiment.

    Args:
        result_file (str): Name of the CSV file to process.
        dir_results (str): Directory where the CSV files are stored.
        imaging_frequence (float): Frame-to-time conversion factor.
        cell_type (str): Type of the cells in the experiment.
        time_interval_discrepancy (int): The time interval to check for discrepancies.
        colony_size_limit (int): Threshold for colony size to apply penalty in trust index calculation.
        TR (TimeRetriever): An instance of the TimeRetriever class.

    Returns:
        list: A list containing the processed row data for the CSV file, including:
            ResultID, CellType, ColonySize, division times (T1-T3.4), 
            quality indexes, generation discrepancy flag, and trust index.
    """
    print(f"Processing {result_file}...")
    df = TR.load_csv_file(dir_results, result_file)
    colony_size = TR.find_colony_size(df)
    result = TR.find_division_time(df)
    result=TR.convert_frame_in_time(result, imaging_frequence)
    qindexes = TR.find_quality_indexes(df)
    flag_d = TR.find_discrepancies(result, time_interval_discrepancy)
    trust_index = TR.compute_trust_index(qindexes["Osc_index-9"], flag_d, colony_size, colony_size_limit)
    row_output = [result_file.replace(".csv", ""), cell_type, colony_size, *result.values(), *qindexes.values(), flag_d, trust_index]
    return row_output

def safe_retrieve_times(result_file, dir_results,\
                   imaging_frequence, cell_type, time_interval_discrepancy, colony_size_limit,\
                   TR, path_logs):   
    """
    Safely process a single CSV file, catching exceptions and logging errors.

    Args:
        result_file (str): Name of the CSV file to process.
        dir_results (str): Directory where the CSV files are stored.
        imaging_frequence (float): Frame-to-time conversion factor.
        cell_type (str): Type of the cells in the experiment.
        time_interval_discrepancy (int): The time interval to check for discrepancies.
        colony_size_limit (int): Threshold for colony size to apply penalty in trust index calculation.
        TR (TimeRetriever): An instance of the TimeRetriever class.
        path_logs (str): Path to the log file where exceptions will be recorded.

    Returns:
        list or None: A list with processed row data if successful; None if an error occurred.
        Errors are logged in the specified path_logs file with full traceback.
    """
    try:
        return retrieve_times(result_file, dir_results,\
                   imaging_frequence, cell_type, time_interval_discrepancy, colony_size_limit, TR)
    except Exception as e:
        print(f"Error with {result_file}: {e}")
        #result={'T1': np.nan, 'T2_1': np.nan, 'T2_2': np.nan,\
        #        'T3_1': np.nan, 'T3_2': np.nan,\
        #        'T3_3': np.nan, 'T3_4': np.nan}
        # error log
        with open(path_logs, "a", encoding="utf-8") as f:
                f.write(f"{result_file}:\n")
                f.write(traceback.format_exc())  # <-- scrive tutto il traceback
                f.write("\n\n")
        return None

def retrieve_times_all_mw(dir_results, csv_files,\
        imaging_frequence, cell_type, time_interval_discrepancy, colony_size_limit, logs_folder = None, name_folder=None, name_scene=None):
    """
    Process all CSV result files in a directory for a given cell type.

    This function iterates over all CSV files, computes division times, quality indexes, 
    generation discrepancy flags, and trust indices, skipping files that raise exceptions.
    Errors are logged in a dated log file.

    Args:
        dir_results (str): Directory where the CSV result files are stored.
        csv_files (list of str): List of CSV filenames to process.
        imaging_frequence (float): Frame-to-time conversion factor.
        cell_type (str): Type of the cells in the experiment.
        time_interval_discrepancy (int): The time interval to check for discrepancies.
        colony_size_limit (int): Threshold for colony size to apply penalty in trust index calculation.
        logs_folder (str, optional): Directory where log files will be saved. Defaults to None, which uses "../Logs".
        name_folder (str): Name of the folder containing the results. Optional.
        name_scene (str): Name of the scene being processed. Optional.

    Returns:
        tuple: (output_file, output_df)
            output_file (list of lists): Raw data rows for all successfully processed files.
            output_df (pandas.DataFrame): DataFrame containing all successfully processed files 
                                          with columns:
                                          ResultID, CellType, ColonySize,
                                          T1, T2.1, T2.2, T3.1, T3.2, T3.3, T3.4,
                                          Indic_exact, Indic_exact-9, Indic_std, Indic_std-9,
                                          Osc_index, Osc_height_index, Osc_index-9, Osc_height_index-9,
                                          GenDiscr, Trust_index
    """
    TR = TimeRetriever()
    output_file = []
    
    if not logs_folder:
        logs_folder = "../Logs"
    os.makedirs(logs_folder, exist_ok=True)
    name_file_log = f"times_errors_{date.today().strftime('%Y%m%d')}"
    if name_folder is not None:
        name_file_log += f"_{name_folder}"
    if name_scene is not None:
        name_file_log += f"_{name_scene}"
    name_file_log += ".txt"
    path_logs = os.path.join(logs_folder, name_file_log)

    column_names = ["ResultID", "CellType", "ColonySize",\
            "T1", "T2.1", "T2.2", "T3.1", "T3.2", "T3.3", "T3.4",\
            'Indic_exact', 'Indic_exact-9', 'Indic_std', 'Indic_std-9',\
            'Osc_index','Osc_height_index',\
            'Osc_index-9', 'Osc_height_index-9', "GenDiscr", "Trust_index"]    
    
    for result_file in csv_files:
        output_row = safe_retrieve_times(result_file, dir_results,\
                imaging_frequence, cell_type, time_interval_discrepancy, colony_size_limit,\
                TR, path_logs)  
        if output_row is not None:
            output_file.append(output_row)
    output_df = pd.DataFrame(output_file, columns=column_names)
    return output_file, output_df
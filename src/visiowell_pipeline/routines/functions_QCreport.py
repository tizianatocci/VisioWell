import ast
import numpy as np
import os
def is_weird_well_naming(item):
    """
    Check whether a microwell name follows the expected naming convention.

    A valid microwell name is expected to:
    - start with a letter between 'A' and 'E' (case-insensitive)
    - be followed by a numeric index <= 5 (e.g. A1, B3, E5)

    Any deviation from this format is considered invalid.

    Parameters
    ----------
    item : str
        Microwell identifier to be checked.

    Returns
    -------
    bool
        True if the microwell name is invalid or out of range, False otherwise.
    """

    if len(item) < 2:
        return True
    letter, number = item[0], item[1:]
    return (
        not letter.isalpha()
        or letter.upper() > "E"
        or not number.isdigit()
        or int(number) > 5
    )

def step0_report(folder_wells, folder_results, path_report_file):
    """
    Compute and append STEP0 quality control metrics to a report file.

    This function analyzes microwell metadata extracted from text files
    produced during STEP0 processing. It computes global statistics across
    all frames, including:

    - total number of detected microwells
    - mean number of microwells per frame
    - predicted loading efficiency based on STEP1 results
    - list of frames containing microwell names outside the expected range

    Results are appended to an existing report file.

    Parameters
    ----------
    folder_wells : str
        Path to the directory containing STEP0 microwell text files.

    folder_results : str
        Path to the directory containing STEP1 result files (.pkl or .csv).

    path_report_file : str
        Path to the report file where QC results will be appended.

    Returns
    -------
    None
    """


    # STEP0 QC
    # Total number of µwells 
    # Mean number of µwells per frame 
    # Predicted percentage of  loading efficency = total number of uwells with 1 cell at time T0 / total number of uwells * 100
    # List of the frames having a naming out of range  [A-E?;1-5]
    scene_files = [f for f in os.listdir(folder_wells) if f.endswith('.txt')]
    n_scenes = len(scene_files)

    total_num_wells = 0
    weird_naming_list = []
    for file_name in scene_files:
        with open(os.path.join(folder_wells, file_name), "r") as f:
            content = [line.strip() for line in f.readlines()]

        wells_lines = [line for line in content if line.startswith("name_well_list=")]
        if wells_lines:
            last_wells_line = wells_lines[-1]  # take last key
            # convert string to list
            name_well_list = ast.literal_eval(last_wells_line.split("=", 1)[1])
            # find weird naming
            weird_naming = any(is_weird_well_naming(item) for item in name_well_list)
            if weird_naming:
                weird_naming_list.append(file_name)
            # add count
            total_num_wells += len(name_well_list)
        #else: error?
    mean_num_wells_per_frame = np.round(total_num_wells/n_scenes,3) if n_scenes else 0

    results_file = [f for f in os.listdir(folder_results) if f.endswith('.pkl')]
    if not results_file: # if the .pkl files are not saved, then refer to .csv files
        results_file = [f for f in os.listdir(folder_results) if f.endswith('.csv')]
    num_results= len(results_file)
    perc_loading_eff = np.round(num_results/total_num_wells*100, 3) if total_num_wells else 0

    with open(path_report_file, "a") as f:
        f.write(
            "QC STEP0\n"
            f"Number of scenes retrieved in step0: {n_scenes}\n"
            f"Total number of microwells found in step0: {total_num_wells}\n"
            f"Mean number of microwells found in step0 per frame: {mean_num_wells_per_frame}\n"
            f"Predicted percentage of loading efficiency (mw in step1/mw found in step0): {perc_loading_eff}%\n"
            f"List of the frames having a naming out of range  [A-E;1-5]: {weird_naming_list}\n"
        )

def step2_report(output_df, path_report_file,\
                 ti_threshold=0.5, osc_threshold=0.3, cs_threshold=30):
    """
    Compute and append STEP2 quality control metrics to a report file.

    This function evaluates microwell-level quality metrics from a dataframe
    produced during STEP2 processing. It reports the percentage of microwells
    failing predefined quality thresholds, including:

    - low trust index
    - high oscillation index
    - abnormal colony size (proxy for intruder/detection issues)
    - generation discrepancies

    Thresholds can be customized via function arguments.

    Parameters
    ----------
    output_df : pandas.DataFrame
        DataFrame containing microwell-level STEP2 results and QC metrics.

    path_report_file : str
        Path to the report file where QC results will be appended.

    ti_threshold : float, optional
        Threshold for the Trust_index metric (default is 0.5).

    osc_threshold : float, optional
        Threshold for the oscillation index metric (default is 0.3).

    cs_threshold : int, optional
        Threshold for colony size indicating potential intruder/detection
        problems (default is 30).

    Returns
    -------
    None
    """

    # STEP2 QC
    # total number of uwells
    # % of uwells with Trust_index < 0.5  
    # % of µwells with Osc_index-9 > 0.3
    # % of µwells with Intruder/Detection problem -> Colony size >30
    # % of µwells with GenerationDiscrepencies
    n_mw = output_df.shape[0]
    if n_mw == 0:
        return
    
    perc_bad_ti = np.round((output_df["Trust_index"] < ti_threshold).sum()/n_mw*100,3)
    perc_bad_osc = np.round((output_df["Osc_index-9"] > osc_threshold).sum()/n_mw*100,3)
    perc_bad_cs = np.round((output_df["ColonySize"] > cs_threshold).sum()/n_mw*100,3)
    perc_bad_gd = np.round((output_df["GenDiscr"]).sum()/n_mw*100,3)

    with open(path_report_file, "a") as f:
        f.write(
            "QC STEP2\n"
            f"Number of uwells analyzed in step2: {n_mw}\n"
            f"Percentage of uwells with trust index < {ti_threshold}: {perc_bad_ti}%\n"
            f"Percentage of uwells with osc-index-9 > {osc_threshold}: {perc_bad_osc}%\n"
            f"Percentage of uwells with colony size > {cs_threshold}: {perc_bad_cs}%\n"
            f"Percentage of uwells with generation discrepancies: {perc_bad_gd}%\n"
        )

def write_report(folder_wells, folder_results, path_report_file, output_df):
    """
    Generate a complete quality control report for the current dataset.

    This function acts as a high-level orchestrator, sequentially running
    STEP0 and STEP2 quality control analyses and appending their results
    to a single report file.

    Parameters
    ----------
    folder_wells : str
        Path to the directory containing STEP0 microwell metadata files.

    folder_results : str
        Path to the directory containing STEP1 result files.

    path_report_file : str
        Path to the report file where QC results will be written.

    output_df : pandas.DataFrame
        DataFrame containing STEP2 analysis results.

    Returns
    -------
    None
    """

    step0_report(folder_wells, folder_results, path_report_file)
    step2_report(output_df, path_report_file)
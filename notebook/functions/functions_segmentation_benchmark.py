import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import sys
import math
sys.path.insert(0, os.path.abspath('..'))
from src.visiowell_pipeline.routines.functions_performances import compute_index_exact,\
    compute_index_exact_until_nine, compute_std, compute_std_until_nine, compute_oscillation_score_until_nine
from src.visiowell_pipeline.routines.functions_postcorrection import correction_CPD

def compute_mean_error(result_num, gt_num, flag_norm=False):
    """
    Compute mean of errors between two numpy arrays.

    Parameters
    ----------
    result_num : np.ndarray.
    gt_num : np.ndarray.
    flag_norm: True if the normalized mean error is desired, False for MAE.

    Raises
    ------
    TypeError
        If result_num or gt_num are not numpy arrays.
    ValueError
        If result_num and gt_num do not have the same shape.
        

    Returns
    -------
    index_mean : float.

    """
    if not isinstance(result_num, np.ndarray):
        raise TypeError("The argument 'result_num' must be np.ndarray.")
    if not isinstance(gt_num, np.ndarray):
        raise TypeError("The groundtruth 'gt_num' must be a np.ndarray.")
    if not result_num.shape==gt_num.shape:
        raise ValueError("The arguments 'result_num' and 'gt_num' must have the same shape.")
    
    if flag_norm:
        mask = gt_num != 0
        index_mean = np.mean(np.abs(result_num[mask] - gt_num[mask]) / gt_num[mask])
    else:
        index_mean = np.mean(np.abs(result_num-gt_num))
    
    return index_mean

    
def compute_mean_error_until_nine(result_num, gt_num, flag_norm=False):
    """
    Compute mean of error between two numpy arrays until the second
    array reaches the value of 9 (excluded).
    
    Parameters
    ----------
    result_num : np.ndarray.
    gt_num : np.ndarray.
    flag_norm: True if the normalized mean error is desired, False for MAE.


    Raises
    ------
    TypeError
        If result_num or gt_num are not numpy arrays.
    ValueError
        If result_num and gt_num do not have the same shape.
        

    Returns
    -------
    index_mean_nine_cells : float.

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
    
    index_mean_nine=compute_mean_error(result_num_less_nine, gt_num_less_nine, flag_norm)
        
    return index_mean_nine

def obtain_output_scores(pipeline_results,\
    col_names = ['Osc_index', 'Osc_height_index', 'Osc_index-9', 'Osc_height_index-9']):
    """
    Extracts and aggregates values from specified columns (`col_names`) 
    across all CSV files found in the subdirectories of a main inference results folder.

    Parameters
    ----------
    pipeline_results : str
        Path to the main directory containing subfolders with inference results.
    col_names : str, optional
        Name of the columns to extract from each CSV file 
        (default: 'Osc_index', 'Osc_height_index', 'Osc_index-9', 'Osc_height_index-9').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated column values,
        each row corresponds to a microwell and each column corresponds to a column in col_names.

    Raises
    ------
    FileNotFoundError
        If the given `pipeline_results` path does not exist.
    NotADirectoryError
        If `pipeline_results` is not a directory.
    ValueError
        If no valid CSV files were found or none contained the requested column.
    """

    # Path validity checks
    if not os.path.exists(pipeline_results):
        raise FileNotFoundError(f"The specified path does not exist: {pipeline_results}")
    if not os.path.isdir(pipeline_results):
        raise NotADirectoryError(f"The specified path is not a directory: {pipeline_results}")

    df_output = {}
    found_any = False  # Track whether any valid file was processed

    for subfolder in os.listdir(pipeline_results):
        subfolder_path = os.path.join(pipeline_results, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(subfolder_path, file)

                    try:
                        df = pd.read_csv(file_path, sep=";")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
                    
                    for col_name in col_names:
                        if col_name in df.columns:
                            value = df[col_name].iloc[0] 
                            
                            file_name = 'Pos_' + os.path.splitext(file)[0].split('Pos_')[-1]
                            key = f"{file_name}_{subfolder}"
                    
                            if key not in df_output:
                                df_output[key] = {}
                    
                            df_output[key][col_name] = value
                    
                            found_any = True
        
    if not found_any:
        raise ValueError(f"No CSV files containing at least one column from '{col_names}' were found in {pipeline_results}.")
    else:
        df_output = pd.DataFrame.from_dict(df_output, orient='index')

    
    return df_output

def filter_score(pipeline_results, thresh_score, name_index):
    """
    Filters the scores obtained from the pipeline results based on a specified threshold 
    and plots the score distribution.
    """
    df_scores = obtain_output_scores(pipeline_results)
    scores = df_scores[name_index]
    # Filter columns based on the threshold
    name_columns_good = list(scores[scores < thresh_score].index)
    # plot
    xmin = math.floor(min(scores) * 10) / 10
    xmax = math.ceil(max(scores) * 10) / 10
    bin_edges = np.arange(xmin, xmax + 0.1, 0.1)
    plt.hist(scores, bins=bin_edges)
    plt.xticks(bin_edges)
    plt.xlabel(name_index)   
    plt.ylabel("Frequency")             
    plt.title("Score distribution")
    plt.show()
    return df_scores, scores, name_columns_good   


def obtain_output_inference(pipeline_results, col_name):
    """
    Extracts and aggregates values from a specified column (`col_name`) 
    across all CSV files found in the subdirectories of a main inference results folder.

    Parameters
    ----------
    pipeline_results : str
        Path to the main directory containing subfolders with inference results.
    col_name : str, optional
        Name of the column to extract from each CSV file (example: "Output YOLO").

    Returns
    -------
    df_output: pd.DataFrame
        A DataFrame containing the aggregated column values, 
        with one column per file/subfolder combination.

    Raises
    ------
    FileNotFoundError
        If the given `pipeline_results` path does not exist.
    NotADirectoryError
        If `pipeline_results` is not a directory.
    ValueError
        If no valid CSV files were found or none contained the requested column.
    """

    # Path validity checks
    if not os.path.exists(pipeline_results):
        raise FileNotFoundError(f"The specified path does not exist: {pipeline_results}")
    if not os.path.isdir(pipeline_results):
        raise NotADirectoryError(f"The specified path is not a directory: {pipeline_results}")

    df_output = pd.DataFrame()
    found_any = False  # Track whether any valid file was processed

    for subfolder in os.listdir(pipeline_results):
        subfolder_path = os.path.join(pipeline_results, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(subfolder_path, file)

                    try:
                        df = pd.read_csv(file_path, sep=";")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                    if col_name in df.columns:
                        output_series = df[col_name]
                        file_name = 'Pos_' + os.path.splitext(file)[0].split('Pos_')[-1]
                        key = f"{file_name}_{subfolder}"
                        df_output[key] = output_series
                        found_any = True
                    else:
                        print(f"Column '{col_name}' not found in {file_path}")

    if not found_any:
        raise ValueError(f"No CSV files containing the column '{col_name}' were found in {pipeline_results}.")

    return df_output


def obtain_performances(df, df_gt, computation_type):
    """
    Computes performance metrics comparing the output of the network (df) 
    and the manually annotated groundtruth (df_gt).

    Parameters  
    df: DataFrame containing the output of the network.
    df_gt: DataFrame containing the groundtruth values.
    computation_type: str. "all" to compute metrics on all data, "nine" to compute metrics until there are 9 cells.
    
    Returns
    dict_results: dict. A dictionary containing the computed metrics for each column in df.
    """

    # this deletes the second column with the same name, if there are duplicated columns in the GT dataframe.
    df_gt = df_gt.loc[:,~df_gt.columns.duplicated()] 
    dict_results = {}

    # Cache GT columns for faster access (especially useful if many columns)
    gt_cols = list(df_gt.columns)

    for col in df.columns:
        # Find all GT columns that are substrings of the current column (result name)
        matches = [gt for gt in gt_cols if gt in col]

        if len(matches) == 0:
            print(f"No GT match found for {col}")
            continue
        
        # If there are multiple matches, take the first one for consistency
        gt_col = matches[0]

        #print(f"Processing {col}  →  GT: {gt_col}")

        # Retrieve the values
        df_values = pd.to_numeric(df[col], errors='coerce')
        gt_values = pd.to_numeric(df_gt[gt_col], errors='coerce')
        
        # Filter rows where neither df_values nor gt_values is NaN
        valid_mask = df_values.notna() & gt_values.notna()
        df_values= df_values[valid_mask].to_numpy()
        gt_values= gt_values[valid_mask].to_numpy()
        
        # Compute metrics
        if computation_type == "all":
            index_exact = compute_index_exact(df_values, gt_values)
            std = compute_std(df_values, gt_values)
            mean_error = compute_mean_error(df_values, gt_values)
            mean_error_norm = compute_mean_error(df_values, gt_values, flag_norm=True)
        elif computation_type == "nine":
            index_exact = compute_index_exact_until_nine(df_values, gt_values)
            std = compute_std_until_nine(df_values, gt_values)
            mean_error = compute_mean_error_until_nine(df_values, gt_values)
            mean_error_norm = compute_mean_error_until_nine(df_values, gt_values, flag_norm=True)


        # Save results
        dict_results[col] = {
            "gt": gt_col,
            "index_exact": index_exact,
            "std": std,
            "mean_error": mean_error,
            "mean_error_norm": mean_error_norm
        }

    return dict_results



def plot_scatterplot_for_architecture(dict_results, y_var, title_plot=None):
    """
    Plots a scatterplot of the performances of the different condition for a given architecture
    (examples of architectures: CellposeSAM, Yolo).
    """

    plt.figure(figsize=(10, 6))
    
    # Set of already plotted training types (for legend)
    plotted_labels = set()
    
    training_types = sorted({key.split('_')[-1] for key in dict_results.keys()})
    colors = cm.tab10(np.linspace(0, 1, len(training_types)))
    color_map = dict(zip(training_types, colors))
    
    for key, value in dict_results.items():
        loss = 1-value["index_exact"]
        y_value = value[y_var]
    
        # Extract training type (e.g., training01)
        training_type = key.split('_')[-1]
    
        # Plot with corresponding color and label
        if training_type not in plotted_labels:
            plt.scatter(loss, y_value, label=training_type, color=color_map[training_type])
            plotted_labels.add(training_type)
        else:
            plt.scatter(loss, y_value, color=color_map[training_type])
        plt.annotate(key.split('_training')[0], (loss, y_value), fontsize=8, alpha=0.7)
    # Final plot formatting
    plt.xlabel("Loss")
    plt.ylabel(y_var)
    plt.title(title_plot)
    plt.legend(title="Training ID")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def group_by_training(dict_results):
    """Groups the results by training type. 
    The training type is extracted from the key of the results dictionary, which is after the last '_' character.
    
    Parameters
    ----------
    dict_results: dict. A dictionary containing the computed metrics for each column in df.

    Returns
    -------
    grouped_by_training: dict. A dictionary where the keys are the training types and the values are lists of results corresponding to each training type.
    """
    grouped_by_training = defaultdict(list)
    for key, value in dict_results.items():
        training_type = key.split('_')[-1]
        grouped_by_training[training_type].append(value)
    return grouped_by_training


def obtain_performances_and_group_by_training(\
    col_name, df_gt,\
    pipeline_results=None, df_results=None,\
    flag_filter=False, name_index=None, thresh_score=None):
    """
    This function: 
        - Step1: If the pandas dataframe 'df_results' is not provided, it will be extracted from 'pipeline_results'.
        The output is the results organized in a dataframe.
        - Step2: Obtains the performances comparing the output of the network and 
        the manually annotated groundtruth.
        Then, for the whole video (Step3) and until there are 9 cells (Step4):
            - Plots all microwells for that architecture
            - Groups by training, if multiple trainings are present.
    Parameters
    ----------
    col_name: str. Name of the column to extract.
    df_gt: dataframe containing the groundtruth values.
    cell_pose_results : str. Path to the main directory containing subfolders with inference results.
    df_cellpose : dataframe containing the output of the network. If not provided, it will be extracted from cellpose_results.
    flag_filter: bool. Whether to filter the results based on a score threshold.
    name_index: str. Name of the column containing the scores to filter on (required if flag_filter is True).
    thresh_score: float. Threshold value for filtering (required if flag_filter is True).

    Returns
    -------
    df_cellpose : dataframe containing the output of the network (filtered if flag_filter is True).
    cellpose_grouped: dict. Performances grouped by training.
    cellpose_nine_grouped: dict. Performances until nine grouped by training.

    """
    # Step1: extract output of the network if the dataframe is not provided
    if pipeline_results and not df_results:
        df_results =obtain_output_inference(pipeline_results, col_name=col_name)
    # Step2: filter results if flag_filter is True
    if flag_filter:
        if pipeline_results:
            df_scores, scores, name_columns_good = filter_score(pipeline_results, thresh_score, name_index=name_index)
            df_results = df_results[name_columns_good]
        else:
            df_results = df_results[df_results[name_index] >= thresh_score]
        
    # Step3: compute performances, plot microwells per architecture, and group by training
    dict_results = obtain_performances(df_results, df_gt, computation_type="all") # compute
    plot_scatterplot_for_architecture(dict_results, y_var = "mean_error")
    results_grouped = group_by_training(dict_results)
    # Step4: compute performances until nine, plot microwells per architecture, and group by training
    dict_results_nine = obtain_performances(df_results, df_gt, computation_type='nine') # compute
    plot_scatterplot_for_architecture(dict_results_nine, y_var = "mean_error")
    results_nine_grouped = group_by_training(dict_results_nine)
    return df_results, results_grouped, results_nine_grouped

def filter_gt_cols(columns, t):
    """
    Filters the GT columns based on the type of GT (TR, HPC, HSC, LMPP, MPP).
    Parameters
    ----------
    columns: list. List of column names.
    t: str. Type of GT (TR, HPC, HSC, LMPP, MPP).

    Return
    ----------
    list. List of column names corresponding to the type of GT.

    """
    if t.startswith("TR"):
        # TR-type GT columns start exactly with this (no parentheses)
        return [c for c in columns if c.startswith(f"{t}_")]
    else:
        # HPC/HSC/LMPP/MPP GT columns start with (type)_
        return [c for c in columns if c.startswith(f"({t})_")]


def filter_pred_cols(columns, t):
    """
    Filters the prediction columns based on the type of GT (TR, HPC, HSC, LMPP, MPP).

    Parameters
    ----------
    columns: list. List of column names.
    t: str. Type of GT (TR, HPC, HSC, LMPP, MPP).

    Return
    ----------

    list. List of column names corresponding to the type of GT.
    """
    if t.startswith("TR"):
        # TR-type predictions contain "...-TR1-A01_tile..."
        token = f"{t}_"
        return [c for c in columns if token in c]
    else:
        # HPC/HSC/LMPP/MPP predictions contain "(type)_tile"
        token = f"({t})_"
        return [c for c in columns if token in c]
        
def plot_results_by_type(df_gt, gt_types, df_results, computation_type,
                         y_var, title_plot=None):
    """
    This function plots the results by type of GT (TR, HPC, HSC, LMPP, MPP) and groups by training.
    Parameters
    ----------
    df_gt: dataframe containing the groundtruth values.
    gt_types: list. List of GT types (TR, HPC, HSC, LMPP, MPP).
    df_results: dataframe containing the output of the network.
    computation_type: str. "all" to compute metrics on all data, "nine" to compute metrics until there are 9 cells.
    """

    all_grouped = {}

    for t in gt_types:
        print("\n### Processing:", t)
        gt_cols = filter_gt_cols(df_gt.columns, t)
        pred_cols = filter_pred_cols(df_results.columns, t)

        print(f" GT columns for {t}: {len(gt_cols)}")
        print(f" Pred columns for {t}: {len(pred_cols)}")

        gt_filtered = df_gt[gt_cols]
        pred_filtered = df_results[pred_cols]

        d_perf = obtain_performances(pred_filtered, gt_filtered, computation_type)
        plot_scatterplot_for_architecture(d_perf, y_var=y_var, title_plot=title_plot)

        grouped = group_by_training(d_perf)
        all_grouped[t] = grouped

    return all_grouped


def final_plot_performance_std(grouped_by_training, y_var = "std",\
                               group_mode = "mean", title=None,\
                               x_axis="Loss", y_axis="Std of errors",\
                               training_to_group=None, markers=None):
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 20}
    matplotlib.rc('font', **font)

    training_names = []
    m_losses = []
    m_stds = []
    y_losses = []
    y_stds = []

    for training, results in grouped_by_training.items():
        loss_values = [1 - r['index_exact'] for r in results]
        y_values = [r[y_var] for r in results]
        
        if group_mode == "mean":
            m_loss = np.mean(loss_values)
            m_y = np.mean(y_values)
        elif group_mode == "median":
            m_loss = np.median(loss_values)
            m_y = np.median(y_values)

            
        std_loss = np.std(loss_values)
        std_y = np.std(y_values)

        training_names.append(training)
        m_losses.append(m_loss)
        y_losses.append(std_loss)
        m_stds.append(m_y)
        y_stds.append(std_y)

    # Gruppi e colori
    if training_to_group is not None:
        groups = sorted(set(training_to_group.values()))
        group_colors = plt.colormaps['Set1'].resampled(len(groups))
        group_color_map = {group: group_colors(i) for i, group in enumerate(groups)}
    else:
        groups = training_names
        group_color_map = {
            name: plt.colormaps['Set1'].resampled(len(training_names))(i)
            for i, name in enumerate(training_names)
        }

    # Marker per ogni training
    if markers is None:
        default_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+', 'x']
        marker_map = {name: default_markers[i % len(default_markers)] for i, name in enumerate(training_names)}
    else:
        marker_map = markers

    fig=plt.figure(figsize=(8, 6))  # Slightly wider
    for i, training in enumerate(training_names):
        group = training_to_group[training] if training_to_group else training
        color = group_color_map[group]
        marker = marker_map[training]
        plt.errorbar(
            m_losses[i], m_stds[i],
            xerr=y_losses[i], yerr=y_stds[i],
            fmt=marker, color=color, ecolor=color,
            linestyle='None', capsize=5, elinewidth=1,
            markersize=15, label=training,\
            markeredgecolor='black'  
        )

    # Prima legenda: per training
    legend1 = plt.legend(
        title="Training ID",
        bbox_to_anchor=(1.50, 0.7),
        loc='upper right',
        fontsize=16,
        title_fontproperties={'size': 14}
    )

    # Seconda legenda: per gruppo (solo colore)
    if training_to_group:
        handles_group = [
            Line2D([0], [0], color=group_color_map[group], lw=4, label=group)
            for group in groups
        ]
        plt.gca().add_artist(legend1)
        plt.legend(
            handles=handles_group,
            title="Group",
            bbox_to_anchor=(1.50, 1),
            loc='center right',
            fontsize=16,
            title_fontproperties={'size': 14}
        )

    if title is None:
        title = 'Scatterplot of Training Results with Error Bars'

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    #plt.subplots_adjust(right=0.7)
    plt.show()
    return fig


def plot_by_type_together(all_grouped, y_var, y_axis, title=None, group_mode="mean", name_label=None):
    """
    This function plots the results of all types of GT together in one scatterplot, 
    with different markers for each type of GT and colors indicating the training group.
    
    Parameters
    ----------
    all_grouped: dict. A dictionary containing the results grouped by type and training.
    y_var: str. The variable to plot on the y-axis (e.g., "std", "mean_error").
    y_axis: str. The label for the y-axis.
    title: str. The title of the plot.
    group_mode: str. "mean" to plot the mean of the results for each condition, "median" to plot the median.
    name_label: str. "SAM" or "YOLO".

    """
    # Combine all grouped into one dict
    final_dict = {}
    for t, grouped in all_grouped.items():
        for training, results in grouped.items():
            key = f"{training}_{t}"  # unique key: training + datatype
            final_dict[key] = results
    
    # Define groups and markers
    training_to_group = {k: name_label for k in final_dict.keys()}  # customize if needed
    default_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+', 'x']
    
    # Assign different markers per training + datatype
    markers_dict = {}
    for i, k in enumerate(final_dict.keys()):
        markers_dict[k] = default_markers[i % len(default_markers)]
    
    # Plot combined

    fig_combined = final_plot_performance_std(final_dict,
                                              y_var = y_var,
                                              y_axis = y_axis, 
                                              title=title,
                                              training_to_group=training_to_group,
                                              markers=markers_dict, group_mode=group_mode)
    return fig_combined


def extract_datatype(col):
    m = re.match(r"(.+?)\)?_tile", col)
    if m:
        t = m.group(1)
        return t.lstrip("(")   
    return None


def divide_by_type(df_gt, df_results, training_to_group,\
                   comp_type,
                   tag_correction=None, flag_save=False, path_save=None, suffix_save=None,
                   group_mode="mean", name_label=None):
    """
     This function divides the results by type of GT 
     (TRx , HPC, HSC, LMPP, MPP) and plots the results for each type separately.

     Parameters
     ----------
        df_gt: dataframe containing the groundtruth values.
        df_results: dataframe containing the output of the network.
        training_to_group: dict. A dictionary mapping each training to a group (e.g., "SAM", "Yolo", etc.) for coloring in the plots.
        comp_type: str. "all" to compute metrics on all data, "nine" to compute metrics until there are 9 cells.
        tag_correction: str. A tag to indicate the type of correction applied (e.g., "Before", "After", etc.) for the title of the plots.
        flag_save: bool. Whether to save the plots.
        path_save: str. Path to the directory where the plots will be saved (required if flag_save is True).
        suffix_save: str. Suffix for the saved plot files.

        Returns
        -------
         all_grouped: dict. A dictionary containing the results grouped by type and training.
     """
    # Extract unique GT types from the GT columns
    gt_types = sorted(set(filter(None, (extract_datatype(c) for c in df_gt.columns))))
    gt_types
    # Plot results by type and group by training
    training_to_group = {'GT': training_to_group}
    # markers_dict = {'GT': 'D'}
    # plot separated results and plot them together
    # scatter plot per condition
    all_grouped=plot_results_by_type(df_gt, gt_types, df_results, computation_type=comp_type,\
                             y_var ="mean_error", title_plot = "Mean error vs Loss")
    # scatter plot with all conditions together
    fig_mean=plot_by_type_together(all_grouped, y_var ="mean_error", y_axis = "Mean error",\
                          title = tag_correction, group_mode = group_mode, name_label = name_label)
    fig_mean_norm=plot_by_type_together(all_grouped, y_var ="mean_error_norm", y_axis = "Mean error Normalized",\
                          title = tag_correction, group_mode = group_mode, name_label=name_label)
    
    if flag_save:
        fig_mean.savefig(os.path.join(path_save, f"fig_mean_{suffix_save}"), bbox_inches='tight', dpi=300)
        fig_mean_norm.savefig(os.path.join(path_save, f"fig_mean_norm_{suffix_save}"), bbox_inches='tight', dpi=300)
    return all_grouped

def workflow_type(pipeline_results, df_results,\
                  col_name, comp_type,
                  flag_filter, name_index, thresh_score,\
                  df_gt, training_to_group, tag_correction, flag_save, path_save, suffix_save,\
                  group_mode,
                  name_label):
    # extract performances and group by training 
    df_results,\
        results_grouped,  results_nine_grouped= obtain_performances_and_group_by_training(\
        col_name, df_gt, pipeline_results, df_results, flag_filter, name_index, thresh_score)
    # divide by condition and save
    all_grouped=divide_by_type(df_gt, df_results, training_to_group, comp_type,\
                               tag_correction, flag_save, path_save, suffix_save,\
                                group_mode, name_label=name_label)
    
    return df_results, results_grouped, results_nine_grouped, all_grouped



def plot_comparison_bc_ac_with_color_legend(all_grouped_bc, all_grouped_ac, df_gt,\
    y_var, y_axis_title = None, \
    title="BC vs AC Comparison", group_mode="mean",\
    flag_save = False, path_save = None,
    datatype_label_map = None,
    datatype_groups=None,
    prefix_save=None):
    """
    Plot BC vs AC together.
    - Same marker per datatype (BC and AC share marker)
    - BC points in red, AC points in green
    - Legend shows marker for datatype (gray) + color legend for BC/AC

    Parameters
    ----------
    all_grouped_bc: dict. A dictionary containing the BC results grouped by type and training.
    all_grouped_ac: dict. A dictionary containing the AC results grouped by type and training.
    df_gt: dataframe containing the groundtruth values.
    y_var: str. The variable to plot on the y-axis (e.g., "mean_error_norm", "mean_error").
    y_axis_title: str. The label for the y-axis. If None, it will be set to "{group_mode} of {y_var}".
    title: str. The title of the plot.
    group_mode: str. "mean" to plot the mean of the results for each condition, "median" to plot the median.
    flag_save: bool. Whether to save the plot.
    path_save: str. Path to the directory where the plot will be saved (required if
        flag_save is True).
    datatype_label_map: dict. Optional mapping from datatype identifiers to display labels for the legend.
    datatype_groups: dict. Optional grouping of datatypes for a multi-column legend.
                Keys are group names (used as subtitles), values are lists of original datatype keys.
                Example: {'Cord Blood': ['HPC', 'HSC'], 'Cell Line': ['BMS']}
    prefix_save: str. Prefix for the saved plot file name (used if flag_save is True).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Combine BC and AC into one dict
    final_dict_combined = {}
    for t, grouped in all_grouped_bc.items():
        for training, results in grouped.items():
            key = f"BC_{training}_{t}"
            final_dict_combined[key] = results
    for t, grouped in all_grouped_ac.items():
        for training, results in grouped.items():
            key = f"AC_{training}_{t}"
            final_dict_combined[key] = results

    # Identify all datatypes and assign markers
    data_types = sorted(set(filter(None, (extract_datatype(c) for c in df_gt.columns))))
    
    # Apply label mapping if provided
    display_label_map = {}
    if datatype_label_map:
        for dt in data_types:
            display_label_map[dt] = datatype_label_map.get(dt, dt)  # fallback to original name
    else:
        display_label_map = {dt: dt for dt in data_types}

    default_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+', 'x']
    marker_map = {dt: default_markers[i % len(default_markers)] for i, dt in enumerate(data_types)}

    # Assign markers and colors
    markers_dict = {}
    colors_dict = {}
    for k in final_dict_combined.keys():
        datatype_found = next((dt for dt in data_types if dt in k), None)
        if datatype_found is None:
            raise ValueError(f"Datatype not found in key: {k}")
        markers_dict[k] = marker_map[datatype_found]
        colors_dict[k] = 'red' if k.startswith("BC") else 'green'

    # Plot
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 14}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 7))

    for k, results in final_dict_combined.items():
        loss_values = [1 - r['index_exact'] for r in results]
        y_values = [r[y_var] for r in results]

        if group_mode == "mean":
            avg_loss = np.mean(loss_values)
            avg_y= np.mean(y_values)
        elif group_mode == "median":
            avg_loss = np.median(loss_values)
            avg_y = np.median(y_values)


        std_loss = np.std(loss_values)
        std_y = np.std(y_values)

        ax.errorbar(avg_loss, avg_y,
                    xerr=std_loss, yerr=std_y,
                    fmt=markers_dict[k],
                    color=colors_dict[k],
                    ecolor=colors_dict[k],
                    markersize=12,
                    markeredgecolor='black',
                    linestyle='None')

    ax.set_xlabel("Loss")
    if not y_axis_title:
        y_axis_title = f"{group_mode} of {y_var}"
    ax.set_ylabel(y_axis_title)
    ax.set_title(title)

            
    # Legend1: for datatypes (markers)
    if datatype_groups:
        group_names = list(datatype_groups.keys())  # ['cord blood', 'cell line']

        group_handles = {}
        for group_name, keys in datatype_groups.items():
            handles = []
            for k in keys:
                if k in marker_map:
                    display_name = display_label_map.get(k, k)
                    handles.append(
                        Line2D([0], [0], color='gray', marker=marker_map[k],
                            linestyle='None', markersize=10, label=display_name)
                    )
            group_handles[group_name] = handles

        legend1 = ax.legend(
            handles=group_handles[group_names[0]],
            title=group_names[0],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            borderaxespad=0.0,
            frameon=False
        )
        ax.add_artist(legend1)

        legend1b = ax.legend(
            handles=group_handles[group_names[1]],
            title=group_names[1],
            loc='upper right',
            bbox_to_anchor=(0.85, 1.0),  
            borderaxespad=0.0,
            frameon=False
        )
        ax.add_artist(legend1b)

    else:
        datatype_handles = [
            Line2D([0], [0], color='gray', marker=marker_map[dt],
                linestyle='None', markersize=10, label=display_label_map[dt])
            for dt in data_types
        ]
        legend1 = ax.legend(handles=datatype_handles, title="Condition", loc='upper right')
        ax.add_artist(legend1)

    # Legend2: for BC/AC colors
    color_handles = [Patch(facecolor='red', edgecolor='black', label='Before Correction'),
                     Patch(facecolor='green', edgecolor='black', label='After Correction')]
    ax.add_artist(legend1)
    ax.legend(handles=color_handles, title="Correction Status", loc='lower right')

    plt.show()
    if flag_save:
        fig.savefig(os.path.join(path_save, f"fig_{prefix_save}_{group_mode}_{y_var}_bc_ac"), bbox_inches='tight', dpi=300)

    return fig

def get_label(label_map, st):
        if label_map and st in label_map:
            return label_map[st]
        return st

def plot_segmentation_error(grouped_results, datatype_groups, marker_map,
                             x_var_name=None, y_var_name=None,
                             title=None,
                             x_label=None, 
                             y_label=None, label_map=None, mode_aggregation=None, legend_inside=True):
    """
    Panel A: Scatter with 9 points (conditions).
    - Color: Identified the group(ex. Cord Blood vs Cell Line).
    - Marker: Identifies the subtype (HPC, HSC, ecc.).
    - Error Bars: Standard Deviation on both axes.
    """
    
    # Setup Font 
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 14}
    matplotlib.rc('font', **font)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. Define markers and colors
    #all_subtypes = list(grouped_results.keys())
    #default_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h']
    #marker_map = {st: default_markers[i % len(default_markers)] for i, st in enumerate(all_subtypes)}
    
    # Colors for macro-groups (Cord Blood vs Cell Line)
    group_names = list(datatype_groups.keys())
    colors = ['#377eb8','#e41a1c'] # blue and red
    group_color_map = {group: colors[i % len(colors)] for i, group in enumerate(group_names)}

    #  plot results
    for subtype, results in grouped_results.items():
        # compute x and y values and their std
        if x_var_name == "Loss":
            x_values = [1 - r['index_exact'] for r in results]
        else:
            x_values = [r[x_var_name] for r in results]
            
        y_values = [r[y_var_name] for r in results]
        
        if mode_aggregation == "aggr_mean":
            m_x, std_x = np.mean(x_values), np.std(x_values)
            m_y, std_y = np.mean(y_values), np.std(y_values)
        elif mode_aggregation == "aggr_median":
            m_x, std_x = np.median(x_values), np.std(x_values)
            m_y, std_y = np.median(y_values), np.std(y_values)            
        
        # find group for each subtype
        current_group = next((g for g, subtypes in datatype_groups.items() if subtype in subtypes), None)
        color = group_color_map[current_group] if current_group else 'gray'

        # get specific marker
        mkr = marker_map.get(subtype, 'o')
        
        # Plot with Error Bars
        ax.errorbar(
            m_x, m_y,
            xerr=std_x, yerr=std_y,
            fmt=mkr,
            color=color,
            ecolor=color,
            markersize=30,
            markeredgecolor='black',
            linestyle='None',
            capsize=4,
            elinewidth=1.5,
            alpha=0.8
        )

    # double legend
    
    # 1. Prepariamo gli handles per la colonna "Cell Line"
    # Aggiungiamo un handle invisibile come titolo
    group1_name = list(datatype_groups.keys())[0] # Es: 'Cell Line'
    group1_subtypes = datatype_groups[group1_name]
    color1 = group_color_map[group1_name]
    
    handles_col1 = [Line2D([0], [0], color='white', label=f"$\\mathbf{{{group1_name}}}$")] # Titolo in grassetto
    for st in group1_subtypes:
        if st in marker_map:
            clean_label = get_label(label_map, st)
            handles_col1.append(Line2D([0], [0], color=color1, marker=marker_map[st], 
                                       linestyle='None', markersize=10, label=clean_label, 
                                       markeredgecolor='black'))

    # 2. Prepariamo gli handles per la colonna "Cord Blood"
    group2_name = list(datatype_groups.keys())[1] # Es: 'Cord Blood'
    group2_subtypes = datatype_groups[group2_name]
    color2 = group_color_map[group2_name]
    
    handles_col2 = [Line2D([0], [0], color='white', label=f"$\\mathbf{{{group2_name}}}$")] # Titolo in grassetto
    for st in group2_subtypes:
        if st in marker_map:
            clean_label = get_label(label_map, st)
            handles_col2.append(Line2D([0], [0], color=color2, marker=marker_map[st], 
                                       linestyle='None', markersize=10, label=clean_label, 
                                       markeredgecolor='black'))

    # 3. Bilanciamo le colonne (Matplotlib riempie per colonna, quindi le liste devono avere stessa lunghezza)
    max_len = max(len(handles_col1), len(handles_col2))
    while len(handles_col1) < max_len:
        handles_col1.append(Line2D([0], [0], color='white', label=" "))
    while len(handles_col2) < max_len:
        handles_col2.append(Line2D([0], [0], color='white', label=" "))

    # Uniamo le liste: Matplotlib con ncol=2 mette i primi N/2 elementi a sx e i restanti a dx
    all_handles = handles_col1 + handles_col2

    if legend_inside:
        # Legenda dentro: Matplotlib decide il posto migliore o la mettiamo in un angolo
        ax.legend(handles=all_handles, 
                  loc='best',        
                  ncol=2, 
                  columnspacing=1.0, 
                  handletextpad=0.5,
                  frameon=True,      
                  framealpha=0.8,    
                  fontsize=11)
    else:
        # Legenda fuori (come prima)
        ax.legend(handles=all_handles, 
                  loc='upper left', 
                  bbox_to_anchor=(1.02, 1), 
                  ncol=2, 
                  columnspacing=1.0, 
                  handletextpad=0.5,
                  frameon=True,
                  fontsize=11)
         
 
    # final esthetics
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([-0.1,1])
    
    plt.tight_layout()
    plt.show()
    
    return fig


def obtain_output_inference(tf_results, col_name="Output YOLO"):
    """
    Extracts and aggregates values from a specified column (`col_name`) 
    across all CSV files found in the subdirectories of a main inference results folder.

    Parameters
    ----------
    tf_results : str
        Path to the main directory containing subfolders with inference results.
    col_name : str, optional
        Name of the column to extract from each CSV file (default: "Output YOLO").

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated column values, 
        with one column per file/subfolder combination.

    Raises
    ------
    FileNotFoundError
        If the given `tf_results` path does not exist.
    NotADirectoryError
        If `tf_results` is not a directory.
    ValueError
        If no valid CSV files were found or none contained the requested column.
    """

    # Path validity checks
    if not os.path.exists(tf_results):
        raise FileNotFoundError(f"The specified path does not exist: {tf_results}")
    if not os.path.isdir(tf_results):
        raise NotADirectoryError(f"The specified path is not a directory: {tf_results}")

    df_output = pd.DataFrame()
    found_any = False  # Track whether any valid file was processed

    for subfolder in os.listdir(tf_results):
        subfolder_path = os.path.join(tf_results, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(subfolder_path, file)

                    try:
                        df = pd.read_csv(file_path, sep=";")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

                    if col_name in df.columns:
                        output_series = df[col_name]
                        file_name = 'Pos_' + os.path.splitext(file)[0].split('Pos_')[-1]
                        key = f"{file_name}_{subfolder}"
                        df_output[key] = output_series
                        found_any = True
                    else:
                        print(f"Column '{col_name}' not found in {file_path}")

    if not found_any:
        raise ValueError(f"No CSV files containing the column '{col_name}' were found in {tf_results}.")

    return df_output


def apply_correction(signal_series):
    signal = signal_series.values
    times = np.arange(len(signal))
    
    # Trova gli indici validi (non NaN)
    valid_idx = ~np.isnan(signal)
    signal_clean = signal[valid_idx]
    times_clean = times[valid_idx]
    
    # Applica la correzione solo sui valori validi
    _, final_signal, _ = correction_CPD(signal_clean, times_clean, "pelt", "l2", pen=2, plot=0)
    
    # Se vuoi, puoi ricostruire un array con NaN nei posti originali
    final_full = np.full_like(signal, np.nan, dtype=float)
    final_full[valid_idx] = final_signal
    
    return final_full

# Funzione wrapper per apply colonna per colonna
def compute_score_column(col_name, df_ultra_bc, df_corrected):
    original_signal = df_ultra_bc[col_name].values
    corrected_signal = df_corrected[col_name].values
    score = compute_oscillation_score_until_nine(original_signal, corrected_signal)
    return score


def extract_key(col):
    import re
    m = re.search(r'(tile\d+_[a-z]\d)_(HPC_3000|HSC_2000|MPP_epp|LMPP_epp|Scene\d+)', col)
    return m.group(0) if m else None


def extract_clean_subtype(key, metrics_dict):
    # 1. Prova a estrarre TRx-Xxx (Cell Lines) dal campo 'gt'
    # Cerchiamo il pattern TR seguito da numeri, trattino e lettera+numeri
    match_tr = re.search(r'TR\d+-[A-Z]\d+', metrics_dict['gt'])
    if match_tr:
        return match_tr.group(0)
    
    # 2. Se non è una TR, estraiamo i sottotipi Cord Blood
    # Cerchiamo quello che sta tra parentesi tonde, es: (HPC_3000) -> HPC_3000
    match_paren = re.search(r'\((.*?)\)', key)
    if match_paren:
        return match_paren.group(1)
    
    # 3. Fallback: se tutto fallisce, usa lo split originale
    return key.split('_')[-2]


def workflow_segmentation_error(pipeline_results, df_results, df_gt, col_name,\
                    flag_filter, thresh_score, name_index, comp_type,\
                    x_var_name, y_var_name, title_plot, x_label, y_label,\
                    datatype_groups, label_map, marker_map, mode_aggregation,\
                    flag_save, path_save, name_fig):
    # Step1: extract output of the network if the dataframe is not provided
    if pipeline_results and not df_results:
        df_results =obtain_output_inference(pipeline_results, col_name=col_name)
    # Step2: filter results if flag_filter is True
    if flag_filter:
        if pipeline_results:
            df_scores, scores, name_columns_good = filter_score(pipeline_results, thresh_score, name_index=name_index)
            df_results = df_results[name_columns_good]
        else:
            df_results = df_results[df_results[name_index] >= thresh_score]
    # Step3: compute performances
    raw_data = obtain_performances(df_results, df_gt, computation_type=comp_type) # compute
    # Step4: group by condition
    grouped_results = {}
    for key, metrics in raw_data.items():
        # extract clean name
        subtype = extract_clean_subtype(key, metrics)
        if subtype not in grouped_results:
            grouped_results[subtype] = []
        grouped_results[subtype].append(metrics)
    print("Conditions found:", list(grouped_results.keys()))

    # Plot scatter plot
    fig_panel_a = plot_segmentation_error(
        grouped_results = grouped_results,   
        datatype_groups = datatype_groups,    
        marker_map=marker_map,     
        x_var_name = x_var_name,    
        y_var_name = y_var_name,        
        title=title_plot,
        x_label = x_label,
        y_label = y_label,
        label_map=label_map,
        mode_aggregation=mode_aggregation
    )
    if flag_save:
        fig_panel_a.savefig(os.path.join(path_save, f"{name_fig}.png"), bbox_inches='tight', dpi=300)


    return fig_panel_a, df_results, raw_data, grouped_results

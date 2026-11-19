import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scikit_posthocs as sp
from statannotations.Annotator import Annotator
from scipy.stats import spearmanr
from matplotlib.ticker import MaxNLocator
import math

# Functions for results
def add_condition_column(name_scenes,exp, path_results, equivalence_dict):
    # obtain a flat df with a new column "Condition"
    global_df = []
    for scene in name_scenes:
        df = pd.read_csv(os.path.join(path_results, scene))
        name_key = equivalence_dict[scene]
        df["Condition"] = name_key
        df["Exp_name"] = exp
        global_df.append(df)
    global_flat_df = pd.concat(global_df, ignore_index=True)
    return global_flat_df

def plot_variable(name_var, global_flat_df, plot_title):
    """
    Plot histograms for each condition.
    """
    # plot colony size
    min_val = global_flat_df[name_var].min()
    max_val = global_flat_df[name_var].max()
    #print(f"Minimum value: {min_val}")
    #print(f"Maximum value: {max_val}")

    # define bins of width 1
    bins = np.arange(min_val, max_val+1, 1)  
    #print(bins)

    # define colors per condition
    conditions = global_flat_df["Condition"].unique()
    palette = sns.color_palette("Set2", len(conditions))
    color_dict = dict(zip(conditions, palette))

    # plot per condition
    g = sns.FacetGrid(global_flat_df, col="Condition", col_wrap=3, sharex=True, sharey=True)
    def hist_colored(data, **kwargs):
        condition = data["Condition"].iloc[0]  
        sns.histplot(
            data=data,
            x=name_var,
            bins=bins,
            stat="density",
            color=color_dict[condition],
            edgecolor="black",
            alpha=0.6
        )

    g.map_dataframe(hist_colored)

    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(plot_title)

    plt.show()


def statistical_test(name_var, global_flat_df):
    # Kruskal test
    kruskal_res = stats.kruskal(
        *[group[name_var].values for name, group in global_flat_df.groupby("Condition")]
    )
    print("Kruskal-Wallis H:", kruskal_res.statistic, "p-value:", kruskal_res.pvalue)

    # Dunn post-hoc test with correction Hochberg
    dunn_res = sp.posthoc_dunn(
        global_flat_df,
        val_col=name_var,
        group_col="Condition",
        p_adjust="simes-hochberg" 
    )
    print(dunn_res)
    return kruskal_res, dunn_res


def extract_pairs(dunn_res, order):
    pairs = []
    pvalues = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            group1 = order[i]
            group2 = order[j]
            p_val = dunn_res.loc[group1, group2]
            
            pairs.append((group1, group2))
            pvalues.append(p_val)
    return pairs, pvalues

"""
def box_plot(name_var, global_flat_df, order, pairs, pvalues,
             title_plot, palette=None, dot_color="#333333"):

    if palette is None:
        palette = "Set2"

    plt.figure(figsize=(10,6))

    ax = sns.boxplot(
        data=global_flat_df,
        x="Condition",
        y=name_var,
        palette=palette,
        order=order
    )

    # points
    sns.stripplot(
        data=global_flat_df,
        x="Condition",
        y=name_var,
        dodge=True,
        jitter=0.2,
        size=5,
        alpha=0.7,
        color=dot_color,
        ax=ax,
        order=order
    )

    # annotation
    annotator = Annotator(
        ax,
        pairs,
        data=global_flat_df,
        x="Condition",
        y=name_var,
        order=order
    )
    annotator.set_pvalues(pvalues)
    annotator.configure(
        test=None,
        text_format='star',
        loc='inside',
        verbose=0,
        pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
    )
    annotator.annotate()

    plt.title(title_plot)
    plt.xlabel("Condition")
    plt.ylabel(title_plot)
    plt.show()
"""


def box_plot(name_var, global_flat_df, order, pairs, pvalues,
             title_plot, ylabel=None, palette=None, 
             flag_save=None, path_save=None, name_file=None, dot_color="#333333"):
    scale = 4
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9 * scale,
        'axes.labelsize': 9 * scale,
        'axes.titlesize': 10 * scale,
        'xtick.labelsize': 7 * scale,
        'ytick.labelsize': 8 * scale,
        'legend.fontsize': 8 * scale,
    })


    if palette is None:
        palette = "Set2"
    if ylabel is None:
        ylabel = name_var  

    fig, ax = plt.subplots(figsize=(3.5*scale, 2*scale), layout='constrained')  


    sns.boxplot(
        data=global_flat_df,
        x="Condition",
        y=name_var,
        palette=palette,
        order=order,
        linewidth=0.5*scale,
        ax=ax
    )

    if len(order) == 1:
        ax.set_xlim(-1, 1)

    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.5))  
        patch.set_linewidth(0.5*scale)
    
    if len(pairs)>0:
        annotator = Annotator(
            ax,
            pairs,
            data=global_flat_df,
            x="Condition",
            y=name_var,
            order=order
        )

        annotator.set_pvalues(pvalues)
        annotator.configure(
            test=None,
            text_format='star',
            loc='inside',     
            verbose=0,
            # fontsize=14,
            text_offset=-3*scale,
            pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
        )
        annotator.annotate()


    ax.set_title(title_plot, fontweight='bold', pad=10)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    n_per_condition = global_flat_df.groupby("Condition").size()
    ax.set_xticklabels([
    f"{cond}\n(n={n_per_condition[cond]})" 
    for cond in order
    ])

    #plt.tight_layout()
    plt.show()

    if flag_save:
        fig.savefig(os.path.join(path_save, f"{name_file}.png"), bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(path_save, f"{name_file}.svg"), bbox_inches='tight')



def analyze_all(global_flat_df, name_var, title_plot, orders, palette, flag_save, path_save, name_file):
    """
    Analyze a variable for all experiments in global_flat_df:
    - Filters groups with <2 samples
    - Runs Kruskal-Wallis + Dunn post-hoc test
    - Generates violin plots with significance
    - Prints Dunn post-hoc p-values

    Parameters:
        global_flat_df (pd.DataFrame): DataFrame with all experiments
        name_var (str): Variable to analyze
        title_plot (str): Title for plots
        orders (dict): Dict of experiment -> desired order of conditions
    """
    for exp in global_flat_df["Exp_name"].unique():
        print(f"\nAnalysis for experiment: {exp}")
        df_exp = global_flat_df[global_flat_df["Exp_name"] == exp].copy()
        df_exp = df_exp[df_exp[name_var].notna()]

        # filter groups with at least 2 samples
        counts = df_exp.groupby("Condition")[name_var].count()
        valid_conditions = counts[counts >= 2].index.tolist()
        df_exp_filtered = df_exp[df_exp["Condition"].isin(valid_conditions)]

        # plot histograms
        plot_variable(name_var, df_exp_filtered, f"{exp}: {title_plot}")

        # keep order only for valid groups
        order = [c for c in orders[exp] if c in valid_conditions]

        # name file for saving
        complete_name_file = name_file + "_" + exp


        if len(valid_conditions) < 2:
            print(f"Not enough valid groups with >=2 samples for {exp} and {name_var}. Plotting without p-values.")
            box_plot(name_var, df_exp_filtered, order, [], [],
                         f"{exp}: {title_plot}", palette=palette,
                         flag_save=flag_save, path_save=path_save, name_file=complete_name_file)
            continue


        # run statistical tests
        res_kruskal, res_dunn = statistical_test(name_var, df_exp_filtered)

        # extract pairs and p-values
        res_pairs, res_pvalues = extract_pairs(res_dunn, order)

        # filter p values that are not significant for plot
        filtered = [(p, pv) for p, pv in zip(res_pairs, res_pvalues) if pv < 0.05]
        # keep only control
        control_name = next((c for c in valid_conditions if c.lower() == "control"), None)
        if control_name is not None:
            filtered = [(p, pv) for p, pv in filtered if control_name in p]

        filtered_pairs = [p for p, _ in filtered]
        filtered_pvalues = [pv for _, pv in filtered]
        res_pairs = filtered_pairs
        res_pvalues = filtered_pvalues

        # violin plots with p-values
        box_plot(name_var, df_exp_filtered, order, res_pairs, res_pvalues,\
                 f"{exp}: {title_plot}", palette=palette,\
                flag_save=flag_save, path_save=path_save, name_file=complete_name_file)
        

        # print Dunn post-hoc p-values
        print(f"\nPost-hoc Dunn p-values for {exp} and {name_var}:")
        for (g1, g2), p in zip(res_pairs, res_pvalues):
            print(f"{g1} vs {g2}: p = {p:.3e}")


# Functions gt
def analyze_gt(data, name_sheet, ordered_conditions, var_boxplot, palette):
    """
    Analyze a single variable from a ground truth sheet:
    - Runs Kruskal-Wallis test
    - Runs Dunn post-hoc test
    - Creates violin plots with significance annotations
    - Filters out groups with <2 samples automatically

    Parameters:
        data (pd.DataFrame): DataFrame containing the data
        name_sheet (str): Name of the sheet / experiment
        ordered_conditions (list): List of conditions in desired order
        var_boxplot (str): Column name of the variable to analyze
    """
    # Build a DataFrame suitable for analysis and plotting
    plot_data = pd.concat(
        [
            pd.to_numeric(data[data["Condition"] == condition][var_boxplot], errors="coerce")
            .dropna()
            .to_frame(name=var_boxplot)
            .assign(Condition=condition)
            for condition in ordered_conditions
        ],
        ignore_index=True
    )

    # Add experiment/sheet name column
    plot_data["Exp_name"] = name_sheet

    # Filter conditions with at least 2 samples
    counts = plot_data.groupby("Condition")[var_boxplot].count()
    valid_conditions = counts[counts >= 2].index.tolist()
    plot_data = plot_data[plot_data["Condition"].isin(valid_conditions)]

    if len(valid_conditions) < 2:
        print("Not enough groups with at least 2 samples each to perform Kruskal-Wallis test.")
        return

    # Run Kruskal-Wallis + Dunn post-hoc
    kruskal_res, dunn_res = statistical_test(var_boxplot, plot_data)

    # Keep order only for valid conditions
    order = [c for c in ordered_conditions if c in valid_conditions]

    # Extract pairs and their p-values
    res_pairs, res_pvalues = extract_pairs(dunn_res, order)

    # Violin plot with significance
    box_plot(
        name_var=var_boxplot,
        global_flat_df=plot_data,
        order=order,
        pairs=res_pairs,
        pvalues=res_pvalues,
        title_plot=f"{var_boxplot} Kruskal-Wallis Test (p = {kruskal_res.pvalue:.4f})",
        palette=palette
    )

    # Print Dunn post-hoc results
    print(f"\nPost-hoc Dunn p-values for {var_boxplot}:")
    for (g1, g2), p in zip(res_pairs, res_pvalues):
        print(f"{g1} vs {g2}: p = {p:.3e}")

        
def randomize_rowwise(df, cols_groups, seed=None):
    """
    Randomizes values row-wise for specified groups of columns.
    
    Parameters:
        df (pd.DataFrame): Original DataFrame
        cols_groups (list of list of str): List of column groups to shuffle row by row
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Copy of the DataFrame with values randomized row-wise
    """
    # Initialize random number generator with seed
    rng = np.random.default_rng(seed)
    
    # Make a copy of the original DataFrame to avoid modifying it
    df_random = df.copy()
    
    # Loop over each group of columns
    for cols in cols_groups:
        # Make a temporary copy of the columns to shuffle
        temp = df_random[cols].copy()
        
        # Loop over each row in the group
        for idx, row in temp.iterrows():
            # Shuffle values within the row
            temp.loc[idx, cols] = rng.permutation(row.values)
        
        # Replace the original columns in df_random with the shuffled ones
        df_random[cols] = temp
    
    return df_random

def plot_scatter(data_random, var1, var2, ordered_conditions):
    # Create a DataFrame for the scatter plot
    scatter_data = pd.concat(
        [
            data_random.loc[data_random["Condition"] == condition, [var1, var2]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
                .assign(Group=condition)
            for condition in ordered_conditions
        ],
        ignore_index=True
    )
    # Scatter plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=scatter_data,
        x=var1,
        y=var2,
        hue="Group",
        palette="Set2",
        s=60,
        alpha=0.7
    )
    plt.title(f"Scatter Plot: {var1} vs {var2}")
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend(title="Scene", bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()


def compute_spearman(data, var1, var2, ordered_conditions):
    print(f"Analysis for {var1} vs {var2}")
    # Global Spearman correlation
    df_corr_global = data[[var1, var2]].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"Number of valid pairs for global correlation: {len(df_corr_global)}")
    corr_global, p_global = spearmanr(df_corr_global[var1], df_corr_global[var2])
    print(f"Global Spearman correlation: r = {corr_global:.3f}, p = {p_global:.4e}")


    # Local Spearman correlation per Condition
    df_corr_local = data[["Condition", var1, var2]].copy()
    df_corr_local[[var1, var2]] = df_corr_local[[var1, var2]].apply(pd.to_numeric, errors="coerce")
    df_corr_local = df_corr_local.dropna()
    df_corr_local["Condition"] = pd.Categorical(df_corr_local["Condition"], categories=ordered_conditions, ordered=True)
    
    for condition, group in df_corr_local.groupby("Condition"):
        n_pairs = len(group)
        if n_pairs >= 2:
            print(f"Number of valid pairs for local correlation per {condition}: {n_pairs} valid pairs")

            corr, p_val = spearmanr(group[var1], group[var2])
            print(f"{condition}: Spearman r = {corr:.3f}, p = {p_val:.4e}")
        else:
            print(f"{condition}: not enough data to calculate correlation")


# correlation
"""
def plot_scatter_with_correlation(df, x_col, y_col, grid_size=20):
    # hexagon view instead of simple scatterplot
    ax_global = df.plot.hexbin(
        x=x_col, 
        y=y_col, 
        gridsize=grid_size,       
        cmap="Reds",      
        mincnt=1           
    )
    ax_global.set_title(f"Hexbin plot of {x_col} vs {y_col}")
    #  compute correlation
    corr_coef, p_value = spearmanr(df[x_col], df[y_col], nan_policy='omit')
    print("Spearman coefficient:", corr_coef)
    print("p-value:", p_value)
    return ax_global

"""

def plot_scatter_with_correlation(df, x_col, y_col, grid_size=20,\
                                  plot_title = None,
                                  by_condition=False, cmap="Reds", order=None,
                                  flag_save = None, path_save=None, name_file=None):

    scale = 4
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 6 * scale,
        'axes.labelsize': 7 * scale,
        'axes.titlesize': 7 * scale, # for axes
        'figure.titlesize': 7 * scale, # whole figure
        'xtick.labelsize': 4 * scale,
        'ytick.labelsize': 4 * scale,
        'legend.fontsize': 4 * scale,
    })


    if not by_condition:
        ax_global = df.plot.hexbin(
            x=x_col, 
            y=y_col, 
            gridsize=grid_size,       
            cmap=cmap,      
            mincnt=1           
        )
        ax_global.set_title(plot_title or f"Hexbin plot of {x_col} vs {y_col}")
        corr_coef, p_value = spearmanr(df[x_col], df[y_col], nan_policy='omit')
        print("Spearman coefficient:", corr_coef)
        print("p-value:", p_value)

    else:
        x_min, x_max = df[x_col].min()-1, df[x_col].max()+5
        y_min, y_max = df[y_col].min()-1, df[y_col].max()+5

        if order is not None:
            conditions = [cond for cond in order 
                          if df[df["Condition"] == cond][[x_col, y_col]].dropna().shape[0] > 0]
        else:
            conditions = [cond for cond in df["Condition"].unique() 
                          if df[df["Condition"] == cond][[x_col, y_col]].dropna().shape[0] > 0]
        

        if len(conditions) == 1:
            n_rows, n_cols = 1,1
        elif len(conditions) < 5:
            n_rows, n_cols = 2, 2
        else:
            n_rows = 2
            n_cols = math.ceil(len(conditions) / n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*scale, 2*scale), layout='constrained')
        axes = np.array(axes).flatten()  
        
        for ax, cond in zip(axes, conditions):
            sub_df = df[df["Condition"] == cond]
            n_samples = len(sub_df[[x_col, y_col]].dropna())
            sub_df.plot.hexbin(
                x=x_col,
                y=y_col,
                gridsize=grid_size,
                cmap=cmap,
                mincnt=1,
                ax=ax
            )
            valid_data = sub_df[[x_col, y_col]].dropna()
            if len(valid_data) >= 3:
                corr_coef, p_value = spearmanr(sub_df[x_col], sub_df[y_col], nan_policy='omit')
                stats_text = f"ρ={corr_coef:.2f}" # ,p={p_value:.2f}"
            else:
                stats_text = "NA"
            ax.set_title(f"{cond} (n={n_samples})\n{stats_text}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_box_aspect(1)
            ax.collections[-1].colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
        for ax in axes[len(conditions):]:
            ax.set_visible(False)
        
        plt.suptitle(plot_title, #or f"Hexbin plot of {x_col} vs {y_col}", 
                     fontweight='bold', y=1.02)
        #plt.tight_layout()
        plt.show()

        if flag_save:
            exp = "_".join(df["Exp_name"].unique())
            complete_name_file = name_file + "_" + exp
            fig.savefig(os.path.join(path_save, f"{complete_name_file}.png"), bbox_inches='tight', dpi=300)
            fig.savefig(os.path.join(path_save, f"{complete_name_file}.svg"), bbox_inches='tight')

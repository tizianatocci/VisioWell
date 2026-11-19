import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from scipy import stats

def compute_bins(data, width=1):
    data = np.asarray(data).ravel()
    data = data[np.isfinite(data)]  # rimuove NaN e inf

    if data.size == 0:
        raise ValueError("No valid numeric data to build bins.")

    min_val = np.floor(data.min())
    max_val = np.ceil(data.max())+1

    if width <= 0:
        raise ValueError("bin_width must be > 0")

    # create at least one bin if all values are the same
    if min_val == max_val:
        max_val = min_val + width

    return np.arange(min_val, max_val + width, width)


def plot_error_histograms(
    diff_df,
    cols = ['T1', 'T2.1', 'T2.2', 'T3.1', 'T3.2', 'T3.3', 'T3.4'],
    condition_col="Condition",
    bin_width=1,
    plot_title=None, 
    path_save=None,
    name_fig_global = None,
):
    """
    Plot histogram of errors globally and per condition
    with fixed width of 1.
    """
    scale = 4
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 5 * scale,
        'axes.labelsize': 6 * scale,
        'axes.titlesize': 6 * scale, # for axes
        'figure.titlesize': 7 * scale, # whole figure
        'xtick.labelsize': 4 * scale,
        'ytick.labelsize': 4 * scale,
        'legend.fontsize': 4 * scale,
    })


    # global with adaptive number of bins based on times
    n_cols = len(cols)
    grid_cols = 4
    grid_rows = (n_cols + grid_cols - 1) // grid_cols
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3*scale, grid_rows *scale), layout='constrained')
    axes = axes.flatten()

    petrol_green = '#1b9e77'

    for i, col_name in enumerate(cols):
        values = diff_df[col_name].dropna()
        #values = diff_df[col_name]
        n_samples = len(values)

        if not values.empty:
            bins_adaptive = compute_bins(values, bin_width)    
            # plot
            axes[i].hist(values, bins=bins_adaptive, color=petrol_green, 
                             edgecolor="black", linewidth=0.3*scale, alpha=0.80)    
            axes[i].set_title(f"Division: {col_name}\n(n = {n_samples})", fontweight='bold') #, fontsize=18)
            axes[i].set_xlabel("Error (h)") #("Time Prediction Error (h)")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(axis='y', linestyle='--', alpha=0.4)
            # integers on y axis
            axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

    # delete empty graph
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(plot_title, y=1.02, fontweight='bold')
    #plt.suptitle(f"Time Error Distribution in hours (n = {len(diff_df)})", y=1.02, fontweight='bold')
    #plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(path_save, f"{name_fig_global}.png"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(path_save, f"{name_fig_global}.svg"), bbox_inches='tight')


    # global with the same number of bins: just to check profiles
    """
    all_values = diff_df[cols].values.flatten()
    all_values_clean = all_values[np.isfinite(all_values)]
    bins = compute_bins(all_values_clean, bin_width)
    diff_df[cols].hist(bins=bins)
    plt.suptitle(f"(n = {len(diff_df)})")
    plt.tight_layout()
    plt.show()
    """

    condition_colors ={}

    # per condition
    for cond, group in diff_df.groupby(condition_col):
        
        all_cond_values = group[cols].stack().values
        if all_cond_values.size == 0:
            continue
            
        bins_cond = compute_bins(all_cond_values, bin_width)
        
        n_cols = len(cols)
        grid_cols = 4
        grid_rows = (n_cols + grid_cols - 1) // grid_cols
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, grid_rows * 4))
        axes = axes.flatten()
        
        current_color = condition_colors.get(cond, "#9b59b6") 

        for i, col_name in enumerate(cols):
            values = group[col_name].dropna()
            n_samples = len(values)
            
            if not values.empty:
                axes[i].hist(values, bins=bins_cond, color=current_color, 
                            edgecolor="black", linewidth=1.2, alpha=0.75)
                
                axes[i].set_title(f"{col_name} (n = {n_samples})", fontweight='bold', fontsize=11)
                axes[i].set_xlabel("Time Error (h)")
                axes[i].set_ylabel("Frequency")
                axes[i].grid(axis='y', linestyle='--', alpha=0.3)
                axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))

            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', alpha=0.5)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])


        plt.suptitle(f"Condition: {cond} | Total Samples n = {len(group)}", 
                    fontsize=16, y=1.02, fontweight='bold', color=current_color)
        
        plt.tight_layout()
        plt.show()

    """
    # per condition comparison
    for cond, group in diff_df.groupby(condition_col):
        values = group[cols].values.flatten()
        bins = compute_bins(values, bin_width)

        group[cols].hist(bins=bins)
        plt.suptitle(f"Condition: {cond}  (n = {len(group)})")
        plt.tight_layout()
        plt.show()
    """


def compute_error_metrics(diff_df, cols, print_results=True):
    """
    Compute MAE, RMSE, BIAS per colonna.

    Returns
    -------
    dict con MAE, RMSE, BIAS (pd.Series)
    """

    MAE  = diff_df[cols].abs().mean()
    RMSE = (diff_df[cols]**2).mean()**0.5
    BIAS = diff_df[cols].mean()

    if print_results:
        print("MAE:\n", MAE)
        print("\nRMSE:\n", RMSE)
        print("\nBIAS:\n", BIAS)

    return {"MAE": MAE, "RMSE": RMSE, "BIAS": BIAS}


def plot_true_vs_predicted(df_true, df_pred, cols, plt_title, path_save, name_file_fig):
    """
    Scatter plot true vs predicted for each column.
    """
    scale = 4
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 5 * scale,
        'axes.labelsize': 6 * scale,
        'axes.titlesize': 7 * scale, # for axes
        'figure.titlesize': 7 * scale, # whole figure
        'xtick.labelsize': 4 * scale,
        'ytick.labelsize': 4 * scale,
        'legend.fontsize': 4 * scale,
    })

    grid_cols = 4
    n_cols = len(cols)
    grid_rows = (n_cols + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(3.5*scale, grid_rows * scale), layout='constrained')
    axes = axes.flatten()
    
    point_color = '#1b9e77' 

    for i, col in enumerate(cols):
        # remove NAs
        mask = df_true[col].notna() & df_pred[col].notna()
        true = df_true.loc[mask, col]
        pred = df_pred.loc[mask, col]
        
        if not true.empty:
            # pearson
            r_pearson, _ = stats.pearsonr(true, pred)
            #print(f"{col}: {r_pearson}")
            r2_val = r_pearson**2
            # spearman
            #rho_spearman, _ = stats.spearmanr(true, pred)
            #print(f"{col}: {rho_spearman}")

            # alpha to handle overlap
            axes[i].scatter(true, pred, alpha=0.5, s=25*scale, color=point_color, edgecolors='black')
            
            # straight line
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 
                         color='black', linestyle='--', linewidth=0.5*scale, label='y=x')
            
            # esthetic
            axes[i].set_title(f"Division: {col}\n(n = {len(true)})", fontweight='bold'), #fontsize=18)
            axes[i].set_xlabel("True (h)")
            axes[i].set_ylabel("Pred (h)")
            axes[i].grid(True, linestyle=':', alpha=0.6)
            # ticks
            start_tick = (min_val // 20) * 20
            end_tick = ((max_val // 20) + 1) * 20
            ticks = np.arange(start_tick, end_tick + 20, 20)
            axes[i].set_xticks(ticks)
            axes[i].set_yticks(ticks)
            
            # pearson coefficient
            axes[i].text(0.05, 0.92, f'$R^2 = {r2_val:.3f}$', 
                 transform=axes[i].transAxes, #fontsize=16, 
                 fontweight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # same scale
            axes[i].set_aspect('equal', 'datalim')
            
    # Rimuovi subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(plt_title, #or f"True vs Predicted Time (n={len(df_true)})", #fontsize=20, 
                 y=1.02, fontweight='bold')
    #plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(path_save, f"{name_file_fig}.png"), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(path_save, f"{name_file_fig}.svg"), bbox_inches='tight')


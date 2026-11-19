"""
Code for retrieving automatically times of division from the .csv files
for each well and generating a QC report.

To implement:
- Take into account the wells step0 and results step1 and time retrieval step2 
in "errors" (see Logs)
I think that currently there are no errors for step1 results,
I think the majority is for step0 well naming
I think there might be some errors for step2 time retrieval

Authors: Tiziana Tocci, Morgan Burq
Project: VisioWell
"""
# ----- LIBRARIES ----- #
import os
import argparse
from ..routines.function_retrievetimes import retrieve_times_all_mw
from ..routines.functions_QCreport import write_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step2: Aggregate detection results and compute division times."
    )
    parser.add_argument(
        "--dir_results",
        type=str,
        required=True,
        help="Path to folder containing detection results (Step 1 output)"
    )
    parser.add_argument(
        "--dir_output",
        type=str,
        required=True,
        help="Path to save final aggregated CSV"
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        required=True,
        help="Cell type being analyzed (e.g., 'HSC', 'cancer_cells')"
    )
    parser.add_argument(
        "--imaging_frequency",
        type=float,
        required=True,
        help="Imaging frequency in images per hour. Needed to compute division times."
    )
    parser.add_argument(
        "--time_interval_discrepancy",
        type=int,
        required=True,
        help="Time interval to check for discrepancies."
    )
    parser.add_argument(
        "--colony_size_limit",
        type=int,
        required=True,
        help="Threshold for colony size to apply penalty in trust index calculation."
    )
    parser.add_argument(
        "--name_folder",
        type=str,
        default="experiment",
        help="Name of the experiment folder. It will be included in the name of the .csv file and report."
    )
    parser.add_argument(
        "--name_scene",
        type=str,
        default="scene",
        help="Name of the scene/region. It will be included in the name of the .csv file and report."
    )
    parser.add_argument(
        "--dir_wells",
        type=str,
        default=None,
        help="Path to wells folder (required for QC report)"
    )
    parser.add_argument(
        "--qcreport_path",
        type=str,
        default=None,
        help="Path to save QC report"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.dir_output, exist_ok=True)
    if args.qcreport_path:
        os.makedirs(args.qcreport_path, exist_ok=True)
    
    # Create logs directory
    logs_folder = os.path.join(args.dir_output, "logs")
    os.makedirs(logs_folder, exist_ok=True)
    
    print(f"Aggregating results from {args.dir_results}")
    print(f"Cell type: {args.cell_type}")
    print(f"Imaging frequency: {args.imaging_frequency} images/hour")
    
    # Find CSV files in results folder
    csv_files = [f for f in os.listdir(args.dir_results) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.dir_results}")
    
    print(f"Found {len(csv_files)} CSV files to aggregate")
    
    # Retrieve times and compute division times
    _, output_df = retrieve_times_all_mw(
        args.dir_results,
        csv_files,
        args.imaging_frequency,
        args.cell_type,
        args.time_interval_discrepancy,
        args.colony_size_limit,
        logs_folder,
        args.name_folder,
        args.name_scene
    )
    
    # Save final output CSV
    output_csv_path = os.path.join(
        args.dir_output,
        f"output_{args.name_folder}_{args.name_scene}.csv"
    )
    output_df.to_csv(output_csv_path, index=False, header=True, na_rep="NA")
    print(f"Saved final output: {output_csv_path}")
    
    # Write QC report if wells path is provided
    if args.dir_wells and args.qcreport_path:
        folder_wells = args.dir_wells
        report_file = f"report_{args.name_folder}_{args.name_scene}.txt"
        report_path = os.path.join(args.qcreport_path, report_file)
        
        if os.path.exists(folder_wells):
            write_report(folder_wells, args.dir_results, report_path, output_df)
            print(f"Saved QC report: {report_path}")
        else:
            print(f"WARNING: Wells folder not found at {folder_wells}, skipping QC report")
    
    print("Step 2 complete")



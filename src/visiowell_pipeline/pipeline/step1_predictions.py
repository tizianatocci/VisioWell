
"""
Code for cell detection in microwells.
This code relies on GPU acceleration if available.
Author: Tiziana Tocci.
Project: VisioWell
"""

# ----- ENV VARIABLE----- #
import os
os.environ["USE_CUPY"]="0"

# ----- CODE----- #
"""
### Libraries
"""
import argparse
import matplotlib 
matplotlib.use("Agg")
from ..routines.functions_detection import retrieve_all_positions, run_on_gpu
from ..classes.classes_detection import NetworkLoader
import GPUtil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step1: Cell detection in microwells."
    )
    parser.add_argument(
        "--path_exp",
        type=str,
        required=True,
        help="Path to the experiment folder (output from step0)"
    )
    parser.add_argument(
        "--dir_results",
        type=str,
        required=True,
        help="Path to save detection results"
    )
    parser.add_argument(
        "--which_network",
        type=str,
        default="cellpose4",
        help="Network model to use (default: cellpose4)"
    )
    parser.add_argument(
        "--name_experiment",
        type=str,
        default="experiment",
        help="Name of the experiment"
    )
    parser.add_argument(
        "--training_to_load",
        type=str,
        default="training_MB_251205/cellpose4training00/model_654269b8d2a64e4e87a3e82ca2e2cb40",
        help="Path to training model to load"
    )
    parser.add_argument(
        "--flag_plot",
        action="store_true",
        help="Save plots"
    )
    parser.add_argument(
        "--flag_gt",
        action="store_true",
        help="If you want to do the groundtruth and continue in the detection even if there is not one cell in frame 0"
    )
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Comma-separated list of positions to analyze (e.g., 'sherlock_pos1,sherlock_pos2')"
    )
    parser.add_argument(
        "--name_series_list",
        type=str,
        default=None,
        help="Comma-separated list of series names (e.g., 'sherlock_1-Pos_000_004,sherlock_1-Pos_001_003')"
    )
    parser.add_argument(
        "--name_well_list",
        type=str,
        default=None,
        help="Comma-separated list of well names (e.g., 'b5,d1,b1')"
    )
    
    args = parser.parse_args()
    
    # Parse list arguments
    positions = [pos.strip() for pos in args.positions.split(",")] if args.positions else None
    name_series_list = [name.strip() for name in args.name_series_list.split(",")] if args.name_series_list else None
    name_well_list = [well.strip() for well in args.name_well_list.split(",")] if args.name_well_list else None

    # Find all positions if not specified
    if name_series_list is None and name_well_list is None and positions is None:
        positions = ["sherlock_" + position for position in retrieve_all_positions(args.path_exp)]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.dir_results, exist_ok=True)

    print(f"Analyzing microwells from {args.path_exp}")

    gpus = GPUtil.getGPUs()
    print(f"Available GPUs: {len(gpus)}")
    
    if gpus:
        print("Running detection on GPU.")
        run_on_gpu(
            args.which_network, 
            positions,
            args.flag_plot,
            args.name_experiment, 
            name_series_list=name_series_list,
            name_well_list=name_well_list,
            training_to_load=args.training_to_load, 
            path_exp=args.path_exp,
            flag_gt=args.flag_gt, 
            dir_results=args.dir_results
        )
    else:
        print("GPU is needed.")
    print("Step 1 complete")

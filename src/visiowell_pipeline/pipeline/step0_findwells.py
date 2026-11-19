# -*- coding: utf-8 -*-
"""
Code for automatically finding microwells.
This code relies on GPU acceleration if a GPU is available.

Author: Tiziana Tocci.
Project: VisioWell
"""
# ----- ENV VARIABLE----- #
import os
os.environ["USE_CUPY"]="1"

# ----- CODE ----- #
"""
### Libraries
"""
import argparse
import matplotlib 
matplotlib.use("Agg")
from ..routines.functions_findwells import run_step0 
import GPUtil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step0: Process CZI file and find microwells."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing the videos to analyze. " \
        "The folder needs to be already existing and needs to contain a file."
    )
    parser.add_argument(
        "--rotation_angle",
        type=int,
        required=True,
        help="Rotation angle in degrees"
    )
    parser.add_argument(
        "--dir_results",
        type=str,
        required=True,
        help="Path to the folder where to save the results. The folder and subfolders will be created."
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.dir_results, exist_ok=True)
    
    print(f"The folder {args.folder_path} is being processed.")
    gpus = GPUtil.getAvailable()
    print(f"Available GPUs: {gpus}")
    run_step0(args.folder_path, args.rotation_angle, args.dir_results)
    print("Step 0 complete")

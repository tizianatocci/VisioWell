# -*- coding: utf-8 -*-
"""
Functions used in step0_pretraining_findwells.py
Author: Tiziana Tocci
"""

# Functions for identifying wells automatically
import os
import numpy as np
import matplotlib.pyplot as plt
from ..routines.functions_general import display_video, write_file
from datetime import datetime
from datetime import date

USE_CUPY = os.environ.get("USE_CUPY", "1") != "0"
if USE_CUPY:
    try:
        import cupy as cp
        _ = cp.zeros(1)  # Verifica accesso GPU
        xp = cp 
        print(f"Using CuPy (GPU)")
    except Exception:
        import numpy as xp
        print("Using NumPy (CPU)")
else:
    import numpy as xp
    print("Using NumPy (forced)")

"""
### Functions for preprocessing
"""

def pipeline_set_parameters(which_microscope):
    """
    Set parameters based on microscope and magnification.

    Parameters
    ----------
    which_microscope : "SherlockBatch1" or "SherlockBatch2" 
    or "SherlockBatch3" or "SherlockBatch4" 
    or "CD7".
    The choice to do based on the experiment is:
        20250123 - SherlockBatch3 (2048x2048 px frames)
        202502xx - SherlockBatch4 (1024x1024 px frames)
        202506xx - SherlockBatch5 (2048x2048 px frames)
        
    Raise
    -----
    TypeError if which_microscope is not a string.

    Returns
    -------
    mw_param1 : int. first parameter CHT.
    mw_param2 : int. second parameter CHT.
    mw_minr : int. minimum radius for well detection.
    mw_maxr : int. maximum radius for well detection.
    mw_mindist : int. minimum distance between identified centers of wells.
    sz_rect : int. size of rectangle used to cut crops around the centers of microwells.
    angle_tolerance: int. Angle tolerance for grouping lines together based on angle.
    distance_threshold: int. distance from line.
    """
    
    if not isinstance(which_microscope, str):
        raise TypeError(f"which_microscope must be a string, got {type(which_microscope)}")
        
    if which_microscope=="SherlockBatch1":
        mw_param1 = 95  # first parameter for CHT.
        mw_param2 = 30 # second parameter for CHT
        mw_minr= 100 # minimum radius for well detection
        mw_maxr= 130 # maximum radius for well detection
        mw_mindist=300 # minimum distance between wells
        sz_rect = 250 # # 280/250 for training. size of rectangle used to cut crops around the center of wells
        angle_tolerance=4 # angle tolerance for grouping lines together based on angle
        distance_threshold = 10  # distance from line
    elif which_microscope=="SherlockBatch2":
        mw_param1 = 95 # f65?
        mw_param2 = 30 # second parameter for CHT. for CD7: mw_param2=50
        mw_minr= 100 # minimum radius for well detection. for CD7: mw_minr=70
        mw_maxr= 130 # maximum radius for well detection. for CD7: mw_maxr=80
        mw_mindist=400 # minimum distance between wells. for CD7: mw_mindist 100
        sz_rect = 250 # 280/250 for training. size of rectangle used to cut crops around the center of wells.for CD7: sz_rect=160
        angle_tolerance=10 # angle tolerance for grouping lines together based on angle
        distance_threshold = 20  # distance from line
    elif which_microscope=="SherlockBatch3":  # 20250123 experiment, frames 2048x2048
        mw_param1 = 75# first parameter for CHT.
        mw_param2 = 30 # second parameter for CHT. 
        mw_minr= 50 # minimum radius for well detection. 
        mw_maxr= 100 # maximum radius for well detection.
        mw_mindist=400 # minimum distance between wells. 
        sz_rect = 170 # 280/250 for training. size of rectangle used to cut crops around the center of wells.
        angle_tolerance=5 # angle tolerance for grouping lines together based on angle
        distance_threshold = 40 #20  # distance from line
    elif which_microscope=="SherlockBatch4": # february 2025 experiments, frames 1024x1024
        mw_param1 = 100 # first parameter for CHT. #before:75 11/04
        mw_param2 = 30 # second parameter for CHT. 
        mw_minr= 25 # minimum radius for well detection. 
        mw_maxr= 50 # maximum radius for well detection.
        mw_mindist=200 # minimum distance between wells. 
        sz_rect = 85 # 280/250 for training. size of rectangle used to cut crops around the center of wells.
        angle_tolerance=5 # angle tolerance for grouping lines together based on angle
        distance_threshold = 20  # distance from line
    elif which_microscope=="SherlockBatch5":  # drug experiment june 2025, frames 2048x2048
        mw_param1 = 100 # first parameter for CHT. #before:75 11/04
        mw_param2 = 30 # second parameter for CHT. 
        mw_minr= 50 # minimum radius for well detection. 
        mw_maxr= 100 # maximum radius for well detection.
        mw_mindist=400 # minimum distance between wells. 
        sz_rect = 170 # 280/250 for training. size of rectangle used to cut crops around the center of wells.
        angle_tolerance= 5 # angle tolerance for grouping lines together based on angle
        distance_threshold = 40  # distance from line
    elif which_microscope=="CD7":
        mw_param1=100
        mw_param2=50
        mw_minr=70
        mw_maxr=80
        mw_mindist=100
        sz_rect=160
        angle_tolerance=4 # angle tolerance for grouping lines together based on angle
        distance_threshold = 10  # distance from line

    return mw_param1, mw_param2, mw_minr, mw_maxr, mw_mindist, sz_rect, angle_tolerance, distance_threshold


def pipeline_retrieve_parameters(video):
    """
    Select parameters based on video size.

    Parameters
    ----------
    video : numpy or cupy array.

    Raises
    ------
    ValueError if video.shape is not among the predifined choices.

    Returns
    -------
    The output of pipeline_set_parameters().

    """
    if not isinstance(video, xp.ndarray):
        raise TypeError(f"video must be a numpy or cupy ndarray, got {type(video)}")
    video_shape=video.shape[:2]
    if video_shape==(2048,2048):
        return pipeline_set_parameters("SherlockBatch5")
    elif video_shape==(1024,1024):
        return pipeline_set_parameters("SherlockBatch4")
    else:
        raise ValueError(f"No option of parameters for a video with shape {video_shape}.")




"""
### Functions to return output
"""
def save_files(dir_result, name_video, fig, merged_info, name_microscope,\
               num_tile = None):
    """
    Save: figure, .pkl and .txt file.

    Parameters
    ----------
    dir_result : string.
    name_video : string.
    fig : matplotlib.pyplot fig.
    merged_info : dict.
    name_microscope: name of the microscope.

    Returns
    -------
    None.

    """
    if not isinstance(dir_result, str):
        raise TypeError(f"dir_result must be a string but got {type(dir_result)}")
    if not isinstance(name_video, str):
        raise TypeError(f"name_video must be a string but got {type(name_video)}")
    if not isinstance(fig, plt.Figure):
        raise TypeError(f"fig must be a matplotlib.pyplot Figure but got {type(fig)}")
    if not isinstance(merged_info, dict):
        raise TypeError(f"merged_info must be a dictionary but got {type(merged_info)}")
    if not isinstance(name_microscope, str):
        raise TypeError(f"name_microscope must be a string but got {type(name_microscope)}")
    
    import pickle

    # name file
    if num_tile is not None:
        name_video += f"_tile{num_tile:03}" 
    # save fig
    fig.savefig(os.path.join(dir_result,  "naming_"+name_video+".png"), bbox_inches='tight', dpi=300)
    
    # save .pkl file result
    name_result = f"{name_microscope}_{name_video}" 
    path_pkl = os.path.join(dir_result, name_result)
    with open(path_pkl + ".pkl", 'wb') as handle:
        pickle.dump(merged_info, handle)
    
    # save. txt file   
    name_well_list= [key for key,value in merged_info.items()]
    name_well_list.sort()
    name_series_list = [name_result] * len(name_well_list)
    name_txt = f"listwells_{name_video}.txt"
    path_txt = os.path.join(dir_result, name_txt)
    with open(path_txt, "a") as file:
        file.write(str(name_video) + "\n")
        file.write(f"name_series_list={name_series_list}\n")
        file.write(f"name_well_list={name_well_list}\n")


"""
### Summary functions.
"""    
def convert_to_numpy(corrected_array):
    """
    Function to convert cupy array into numpy array, if the input is a cupy array, otherway it prints that there is no need to convert.
    Parameters
    ----------
    corrected_array: numpy or cupy array.

    Returns
    -------
    corrected_array: numpy array.
    """
    if not isinstance(corrected_array, xp.ndarray):
        raise TypeError("'corrected_array' expected to be a numpy or cupy error but got {type(corrected_array)}")
    try:
        corrected_array = xp.asnumpy(corrected_array)
    except Exception:
        print("No need to convert into numpy")
    return corrected_array

def process_video(video_path, num_tile, rotation_angle, dir_result):
    """
    Process video to find microwells.
    The preprocessing part can benefit from CuPy GPU acceleration.
    The microwell identification and naming is written in numpy, therefore it runs on CPU.

    Parameters
    ----------
    video_path : string.
    num_tile: if .czi file is being analyzed.
    rotation_angle: if video must be rotated.

    Raises
    ------
    TypeError if video_path is not string.

    Returns
    -------
    None.

    """
    from ..classes.classes_mw import MicrowellPreprocesser, MicrowellFinder, MicrowellNamer
    
    if not isinstance(video_path, str):
        raise TypeError(f"video_path must be a string, got {type(video_path)}")
        
    print(f"Processing {video_path}...")
    """
    ### Parameters
    """
    flag_plot= False
    #mw_param1, mw_param2, mw_minr, mw_maxr, mw_mindist, sz_rect,\
    #angle_tolerance, distance_threshold = pipeline_set_parameters("SherlockBatch4" )
    ref_frame=-1
    
    """
    ### Folders and paths
    """
    path_exp_pos=video_path # position to analyse
    name_video= (os.path.basename(video_path)).rsplit('.', 1)[0]
    if dir_result is None:
        dir_result = "../Wells"
    
    """
    ### Load images, correct for drift, scale images to 8 bit
    """
    MWP = MicrowellPreprocesser()
    images_array = MWP.load_images(path_exp_pos, num_tile)
    
    #images_array = pipeline_load_stack(path_exp_pos, num_tile)
    mw_param1, mw_param2, mw_minr, mw_maxr, mw_mindist, sz_rect,\
    angle_tolerance, distance_threshold =pipeline_retrieve_parameters(images_array)
    #display_video(images_array, 600) # verify if the video is correctly loaded
    # correct of drift according to first frame
    corrected_array = MWP.correct_drift(images_array, 'nearest', nptype=xp.uint16, ref_frame=ref_frame)
    del images_array
    #corrected_array=images_array
    #display_video(corrected_array, 600) # verify the result of the drift correction
    # convert to 8 bit with eventual range adjustment
    corrected_array = MWP.range_adjustment(corrected_array, mode = "whole")
    #display_video(corrected_array, 600) # verify result of conversion
    # rotate
    if rotation_angle is not None and rotation_angle != 0:
        corrected_array = MWP.rotate_video(corrected_array, rotation_angle=rotation_angle)
        #display_cideo(corrected_array, 600)

    """
    Find wells in first frame (with Circular Hough Transform approach)
    and cut a rectangle centered in each microwell identified.
    """
    corrected_array = convert_to_numpy(corrected_array)
    MWF = MicrowellFinder(corrected_array[:,:,ref_frame].copy())
    _ = MWF.find_microwells(1, mw_mindist,\
        mw_param1, mw_param2, mw_minr, mw_maxr,\
        flag_plot=flag_plot, title="All the microwells who were identified")  
    dict_crops = MWF.cut_rectangles(sz_rect, corrected_array, ref_frame, flag_plot)
    dict_final_crops = MWF.filter_crops(dict_crops, sz_rect, corrected_array, ref_frame, flag_plot, mode="flexible") # filter crops who are cut
    del dict_crops, MWF    
    
    """
    Assign name to the wells.
    """
    MWN = MicrowellNamer()
    final_centers = MWN.extract_centers(dict_final_crops)
    #show_circles(corrected_array[:,:,ref_frame].copy(), final_centers, radius=int((mw_minr+mw_maxr)/2))
    final_sorted_data = MWN.order_points(corrected_array, final_centers,\
        angle_tolerance, distance_threshold, user_threshold_angle=10,\
        ref_frame=ref_frame, flag_plot=flag_plot)
    merged_info = MWN.assign_names(final_sorted_data, dict_final_crops)
    # visualize result
    fig=MWN.visualize_naming(corrected_array, merged_info)
    # save figure, .pkl file, .txt file
    save_files(dir_result, name_video, fig, merged_info,name_microscope="sherlock", num_tile=num_tile)
    print(f"Finished processing {video_path}")
       
    
def safe_process_video(video_path, num_tile, rotation_angle=None, dir_results=None):
    """
    Function to safely run process_video and return an error if somethines goes unexpected.

    Parameters
    ----------
    video_path: string.

    Returns
    -------
    Tuple: string, dict.
    """
    import traceback
    
    # name file 
    if num_tile is not None:
        if video_path.endswith(".czi"):
            video_path_with_tile = video_path.removesuffix(".czi")
        video_path_with_tile = f"{video_path_with_tile}_tile{num_tile:03}"
    else:
        video_path_with_tile = video_path
        
    try:
        process_video(video_path, num_tile, rotation_angle, dir_results)
        return ("success", {"video_path": video_path_with_tile})
    except Exception as e:
        print(f"Error for {video_path_with_tile}: {e}")
        return ("error", {
            "video_path": video_path_with_tile,
            "traceback": traceback.format_exc()
        })    


def append_log_error_for_video(video_path, trace, error_log_file):
    """
    Function to write a log error.
    Parameters
    ----------
    video_path: str.
    trace: str.
    error_log_file: str.
    """
    if not isinstance(video_path, str):
        raise TypeError(f"video_path must be a string but got {type(video_path)}")
    if not isinstance(error_log_file, str):
        raise TypeError(f"error_log_file must be a string but got {type(error_log_file)}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = (
        f"[{timestamp}] Error in {video_path}:':\n"
        f"{trace}\n"
    )

    with open(error_log_file, "a") as log:
        log.write(error_message)



def run_step0(video_folder, rotation_angle=None, dir_results=None):
    """
    Processes videos in the specified folder. Supports:
    - .tif image stacks
    - Folders containing .tif images
    - .czi files with multiple tiles

    Uses CuPy (GPU) if available, otherwise falls back to NumPy (CPU).

    Parameters
    ----------
    video_folder : str
        Path to the folder containing videos.

    Raises
    ------
    TypeError
        If video_folder is not a string.
    """
    import os
    from datetime import date
    from aicspylibczi import CziFile
    
    if not isinstance(video_folder, str):
        raise TypeError(f"video_folder must be a string, got {type(video_folder)}")

    video_files = []
    video_type = None  # "tif", "folder", or "czi"
    
    # 1. Look for .tif files
    tif_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.tif')]
    if tif_files:
        video_files = tif_files
        video_type = "tif"
        print(f"Found {len(video_files)} .tif files: {video_files}")

    # 2. Look for folders (if no .tif files found)
    if not video_files:
        tif_folders = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
                       if os.path.isdir(os.path.join(video_folder, f))]
        if tif_folders:
            video_files = tif_folders
            video_type = "folder"
            print(f"Found {len(video_files)} .tif folders: {video_files}")

    # 3. Look for .czi files (if still empty)
    if not video_files:
        czi_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.czi')]
        if czi_files:
            czi_path = czi_files[0]  # Assuming one .czi file
            video_type = "czi"
            print(f"Found .czi file: {czi_path}")

            czi_mosaic = CziFile(czi_path)
            dims = czi_mosaic.dims
            sizes = czi_mosaic.size

            if 'M' not in dims:
                raise ValueError("The .czi file does not contain an 'M' (tiles) dimension.")
            idx_m = dims.index('M')
            num_tiles = sizes[idx_m]

            # Process each tile
            suffix = os.path.splitext(os.path.basename(czi_path))[0]

            # Create logs directory
            logs_dir = os.path.join(dir_results, "logs")
            os.makedirs(logs_dir, exist_ok=True)


            error_log_file = f"{logs_dir}/positions_errors_{date.today().strftime('%Y%m%d')}_{suffix}.txt"
            output_log_file = f"{logs_dir}/positions_valid_{date.today().strftime('%Y%m%d')}_{suffix}.txt"

            for tile_index in range(num_tiles):
                status, result = safe_process_video(czi_path, num_tile=tile_index, rotation_angle=rotation_angle, dir_results=dir_results)
                if status == "error":
                    append_log_error_for_video(result["video_path"], result["traceback"], error_log_file)
                elif status == "success":
                    write_file(output_log_file, result["video_path"])
            return  # End processing for .czi files

    # 4. Process .tif files or folders
    if video_type in ["tif", "folder"]:
        for path_video in video_files:
            status, result = safe_process_video(path_video, rotation_angle=rotation_angle, num_tile=None, dir_results=dir_results)
            if status == "error":
                append_log_error_for_video(result["video_path"], result["traceback"], error_log_file)
            elif status == "success":
                write_file(output_log_file, result["video_path"])
    else:
        print("No valid video files or folders found in the provided directory.")

"""
### Commodity functions to visualize data while debugging.
"""
    
def plot_overlapping_groups(overlapping_group, corrected_array):
    img =corrected_array[:,:,0].copy()
    for key, point_list in overlapping_group.items():
        plt.imshow(img, cmap='gray')

        for segment in point_list:
            x_vals = [segment["p1"][0], segment["p2"][0]]
            y_vals = [segment["p1"][1], segment["p2"][1]]

            plt.plot(x_vals, y_vals, 'ro-')  # line between p1 and p2

        plt.title(f"Segments for {key}")
        plt.axis('off')
        plt.show()
        

def plot_final_sorted_data(final_sorted_data, corrected_array):
    data=final_sorted_data
    img =corrected_array[:,:,0].copy()
    for key, points in data.items():
        plt.imshow(img)
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        plt.plot(x_vals, y_vals, 'ro-')  # red line with dots
        plt.title(f"Points for {key}")
        plt.show()
""" 
### OTHER: not used
"""

def keep_notdrifted_crops(selected_crops):
    final_crops = {}
    for i in range(len(selected_crops)):
        well_key = list(selected_crops.keys())[i]
        one_crop = selected_crops[well_key]["crop"]
        flag_keep = 1
        for j in range(one_crop.shape[-1]):
            if np.sum(one_crop[:,:,j]==0) > one_crop.shape[0]*one_crop.shape[1]*5/100:
                flag_keep = 0
                break
        if flag_keep == 1:
            final_crops[well_key]={'circle': selected_crops[well_key]["circle"],\
                                     'rect': selected_crops[well_key]["rect"],\
                                     'crop': selected_crops[well_key]["crop"]}
    return final_crops
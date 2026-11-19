# -*- coding: utf-8 -*-
"""
Code for obtaining the images of the desired microwell once Step0 has been run.
Workflow: run step 0 -> run this code for obtaining the images of the desired microwell for training.
The images are necessary to do the groundtruth with Cellpose GUI.
Author: Tiziana Tocci
"""
import os
import logging
import numpy as np
import cv2
import pickle

def load_pickle_file(dir_file, name_file):
    """
    Load file .pkl.


    Parameters
    ----------
    dir_file : string. Directory where the .pkl file is.
    name_file : string. Name of the .pkl file.

    Returns
    -------
    data_pkl : dictionary. Content of the .pkl file.

    """
    
    with open(os.path.join(dir_file, name_file), 'rb') as f:
        data_pkl = pickle.load(f)
    return data_pkl        

def from_np_array_to_tif(video, sz_video, path_folder):
    if not isinstance(video, np.ndarray):
        raise TypeError("'video' must be a np array")
    if not isinstance(sz_video, int) or not sz_video > 0:
        raise TypeError("'sz_video' must be a positive integer.")
    if not isinstance(path_folder, str):
        raise TypeError("'path_folder' must be a string.")
    name_subfolder = os.path.basename(path_folder.rstrip(os.sep))
    for i in range(video.shape[-1]):
        img = cv2.resize(video[:,:,i], (sz_video,sz_video), interpolation=cv2.INTER_LINEAR)
        filename = f"{name_subfolder}_{i:03d}.tif"
        cv2.imwrite(os.path.join(path_folder, filename), img)
    logging.info(f"The images were obtained for {name_subfolder}.")
    
def write_frames_from_pkl_file(path_pkl, name_position, name_file,\
                               path_mainfolder_images, name_well, name_subfolder,
                               sz_video=800,flag_show=0):
    if not isinstance(path_pkl, str):
        raise TypeError("'path_pkl' must be a string.")
    if not isinstance(name_position, str):
        raise TypeError("'name_position' must be a string.")
    if not isinstance(name_file, str):
        raise TypeError("'name_file' must be a string.")
    if not isinstance(path_mainfolder_images, str):
        raise TypeError("'path_mainfolder_images' must be a string.")
    if not isinstance(name_well, str):
        raise TypeError("'name_well' must be a string.")
    if not isinstance(name_subfolder, str):
        raise TypeError("'name_subfolder' must be a string.")
    if not isinstance(sz_video, int):
        raise TypeError("'sz_video' must be an integer.")
    if not isinstance(flag_show, bool):
        raise TypeError("'flag_show' must be a boolean.")
   
    # load video
    file = load_pickle_file(path_pkl, name_file)
    if isinstance(file, dict) and not name_well in file:
        raise KeyError(f"{name_well} is not a key in the {name_position} dictionary.")
    if isinstance(file, dict) and not "crop" in file[name_well]: 
        raise KeyError (f"'crop' is not a key in the {name_well} dictionary.")
    # if sherlock -> I have a dictionary
    # if CD7 -> I have a np array
    if isinstance(file, dict):
        video=file[name_well]["crop"]
    else:
        video = file
        dims = video.shape
        if dims[0] != dims[1] or dims[0] != dims[2]:
            video = np.moveaxis(video, 0, -1)  # move different dimension to the end

    if flag_show:
        display_video(video)
        
    # if directory does not exist, create it
    directory = os.path.join(path_mainfolder_images, name_subfolder)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"The folder {name_subfolder} was created.")
        
    # write frames
    from_np_array_to_tif(video, sz_video, os.path.join(path_mainfolder_images, name_subfolder))


# ----- MODIFY ----- #
"""
### File to analyse
"""
if __name__ == "__main__":
    name_experiment = "MB_20251003_MPP_epp" #name of the experiment
    name_position = "tile107" #"002_001" # position in the experiment
    name_well = "a2" #"b1" # name of the well in the position
    flag_show = False
    
    #path_pkl = "/Users/tizia/Documents/Curie/StemCells/Python/videoCD7fortraining" # directory where the result of Step 0 is
    path_pkl =  r"H:\Analysis\03102025\MPP_epp"
    name_file= f"sherlock_MB_251003_cord_blood_34-01(MPP_epp)_{name_position}.pkl" #"New-02(2)_{name_position}_{name_well}.pkl" #f"sherlock_1-Pos_{name_position}.pkl" # name of the .pkl file
    #path_mainfolder_images="/Users/tizia/Documents/Curie/StemCells/Python/Wells dictionaries" # path where the subfolders with the images are stored
    path_mainfolder_images =r"C:\Users\tizia\Documents\Curie\StemCells\Python\Wells dictionaries"
    name_subfolder = f"{name_experiment}_{name_position}_{name_well.upper()}" #name of the folders where the files will be saved
    # ----- REST OF THE CODE ----- #
    """
    ### Set working directory
    """
    import os
    #dir_codes = "/Users/tizia/Documents/Curie/StemCells/Python/Code"
    dir_codes = r"C:\Users\tizia\Documents\Curie\StemCells\Python\Code"
    #os.chdir(dir_codes)
    
    """
    ### Libraries
    """
    import cv2
    import sys
    import logging
    import numpy as np
    sys.path.append("C:/Users/tizia/Documents/Curie/Python for Morgane/Code")
    #sys.path.append(r"C:\Users\team_\Documents\Tiziana\VisioWell")
    from functions_general import display_video, load_pickle_file 
    
    # logging configuration
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    write_frames_from_pkl_file(path_pkl, name_position, name_file,\
                                   path_mainfolder_images, name_well, name_subfolder,
                                   flag_show=flag_show)
# Save .tif from .pkl file
import pickle
import os
import tifffile
import numpy as np

# insert the path and name of file and the microwell to analyze
folder_path = r"C:\Users\tizia\OneDrive - INSTITUT CURIE\UnitPC_training\Wells\MB07082025_new\Scene007"  
file_name = "sherlock_New-01-Scene-07-TR7-B02_tile005.pkl"
microwell = 'a3'
folder_result = "../tiff_videos"  # folder where the tif will be saved - it is created automatically
flag_pickle = True #True if you want to save pickle file otherwise False

### CODE
file_path = os.path.join(folder_path, file_name)
with open(file_path, "rb") as f:
    data = pickle.load(f)

if not microwell in data.keys():
    raise KeyError(f"{microwell} is not a key in {file_name}")
if not "crop" in data[microwell].keys():
    raise KeyError(f"'crop' is not a key in {file_name} of microwell {microwell}")

video = data[microwell]["crop"]
video = np.transpose(video, (2, 0, 1)) 
print("The video was exactracted successfully!")
os.makedirs(folder_result, exist_ok=True) # create video of results if it does not exist
name_video = file_name.strip('sherlock_').strip('.pkl') + f"_{microwell}.tif" 
path_video = os.path.join(folder_result, name_video)
tifffile.imwrite(path_video, video) # save .tif
print(f"The .tif file corresponding to {file_name} of microwell {microwell} was successfully saved.\n Open with FIJI")
if flag_pickle:
    import pickle
    name_pickle = file_name.strip('sherlock_').strip('.pkl') + f"_{microwell}.pkl" 
    path_pickle = os.path.join(folder_result, name_pickle)
    with open(path_pickle, 'wb') as file:
        pickle.dump(video, file)
    print(f"The pickle file corresponding to {file_name} of microwell {microwell} was successfully saved.\n ")

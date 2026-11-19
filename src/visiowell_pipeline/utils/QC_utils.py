# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:53:12 2025

@author: tizia
"""
from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
from ..routines.functions_QCreport import is_weird_well_naming

def FindFilesNameInFolderWtExtension(folder: str, extension: str = "*") -> list:
    """
    Find file names in a folder, either all or those with a desired extension.

    Parameters
    ----------
    folder : str.
    extension : str. The default is "*", which corresponds to all files in folder.

    Returns
    -------
    name_files : list of Paths.

    """
    import pathlib
    import warnings
    
    folder = pathlib.Path(folder)
    
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist.")
    elif not folder.is_dir():
        raise NotADirectoryError(f"{folder} exists but is not a directory.")


    if extension == "*":
        pattern = "*"
    else:
        pattern = f"*.{extension.lstrip('.')}"  

    
    name_files = list(folder.glob(pattern))
    
    if not name_files:
        warnings.warn(f"Warning: No files found in the folder {folder}.")
        
    return name_files


def LoadCSV(full_path: Union[str, Path]) -> pd.DataFrame :
    """
    Load CSV file in pd dataframe

    Parameters
    ----------
    folder : str or Path.
    name_file : str.

    Returns
    -------
    df : pd dataframe.

    """
    from pathlib import Path
    import pandas as pd

    full_path = Path(full_path)
    
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} does not exist.")
        
    df = pd.read_csv(full_path)
    
    return df


def RetrieveNameWells(file_path: Union[str, Path]) -> list:
    """
    Retrieve wells name stored in the .txt file of the position.

    Parameters
    ----------
    file_path : str or Path.

    Raises
    ------
    ValueError
        if name_well_list is not found in file.

    Returns
    -------
    name_well_list: list.

    """
    import ast
    
    name_well_list = None
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("name_well_list"):
                name_well_list = ast.literal_eval(line.split('=', 1)[1])
                break
    
    if name_well_list is None:  # Handle case where the expected variable wasn't found
            raise ValueError(f"'name_well_list' not found in file: {file_path}")
            
    return name_well_list


def IsNamingWeird(name_mws: list) -> bool:
    """

    Parameters
    ----------
    name_mws : list.

    Returns
    -------
    matches : boolean.

    """
    weird_naming = any(is_weird_well_naming(item) for item in name_mws)
    
    return weird_naming


def RetrievePositionsWeirdNaming(name_files: list,\
    #limit_letter : str = "g"
    ) -> list:
    """
    Retrieve list of weird naming files.

    Parameters
    ----------
    name_files : list.
    limit_letter : str, optional. The default is "g".

    Returns
    -------
    weird_naming_files: list.

    """
    weird_naming_positions = [
    name_file.name[len('listwells_'):-len('.txt')]
    for name_file in name_files
    if IsNamingWeird(RetrieveNameWells(name_file))
    ]
    
    return weird_naming_positions


def LoadFigNaming(folder: Union[str, Path], name_pos: str) -> np.ndarray:
    """
    Load .png image for microwell.

    Parameters
    ----------
    folder : Union[str, Path]
        .
    name_pos : str
        .

    Returns
    -------
    img : np ndarray.

    """
    from pathlib import Path
    import cv2
    name_file = f"naming_{name_pos}.png"
    full_path = Path(folder) / name_file
    img = cv2.imread(full_path)  # legge immagine in BGR
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    

def FindWeirdTimes(df_original: pd.DataFrame, imaging_frequence: int) -> pd.DataFrame:
    """
    Find microwells for which there is a weird naming convention,
    namely T1=T2.1 and T2.2=T3.1

    Parameters
    ----------
    df_original : pd.DataFrame.
    imaging_frequence : int.
    
    Returns
    -------
    name_positions : pd.DataFrame.

    """
    if df_original is None or df_original.empty:
        raise TypeError("df_original is None or empty.")
        
    df  = df_original.copy()
    
    # find booleans of where the errors are
    error_first_to_second_gen = df["T1"] == df["T2.1"]
    error_second_to_third_gen = df["T2.2"] == df["T3.1"]
 
    # add frame where there is the error
    # add type of error
    df.loc[error_first_to_second_gen, "Frame"] = (df.loc[error_first_to_second_gen, "T1"] * imaging_frequence).astype(int)
    df.loc[error_second_to_third_gen, "Frame"] = (df.loc[error_second_to_third_gen, "T2.2"] * imaging_frequence).astype(int)
    df.loc[error_first_to_second_gen, "Type"] = "T1"
    df.loc[error_second_to_third_gen, "Type"] = "T2.2"
    
    
    # find suspicious rows
    rows_sus = pd.concat([
        df.loc[error_first_to_second_gen, "ResultID"],
        df.loc[error_second_to_third_gen, "ResultID"]],
        ignore_index = True)
    
    
    # define name_positions df with pos, microwell, frame and type
    name_positions = df.loc[df['ResultID'].isin(rows_sus), ["ResultID", "Frame", "Type"]].copy().reset_index(drop=True)
    del df
    name_positions[["POS", "MW"]] = name_positions["ResultID"].str.extract(r'(tile\d+)_([a-zA-Z0-9]+)')
    name_positions = name_positions[["ResultID", "POS", "MW", "Frame", "Type"]]
    
    return name_positions

def FindMatchingFile(folder, pos, ext=".pkl"):
    """
    Find matching while containing the regex pattern "pos".

    Parameters
    ----------
    folder : Path or str.
    pos : str. A regex pattern.
    ext : str. The default is ".pkl".

    Returns
    -------
    matched_file : Path or None.

    """
    from pathlib import Path
    import re

    folder = Path(folder) 
    files_in_dir = set(folder.iterdir())  
    
    pattern = re.compile(pos)

    matched_file = next(
            (
                f
                for f in files_in_dir
                if f.is_file()
                and f.name.endswith(ext)
                and pattern.search(f.name)
            ),
            None,
        )
    
    if matched_file is None:
        raise FileNotFoundError(f"No matching file in {folder} for {pos}.")
        
    return matched_file


def CreateVideoForWeirdResults(name_positions: pd.DataFrame, num_index: int,\
    dir_wells: Union[str, Path], dir_results: Union[str, Path],\
    num_frames: int = 5) -> np.ndarray:
    """
    Display the images with the masks were the results changes.

    Parameters
    ----------
    name_positions : pd.DataFrame
        DESCRIPTION.
    num_index : int
        DESCRIPTION.
    dir_wells : Union[str, Path]
        DESCRIPTION.
    dir_results : Union[str, Path]
        DESCRIPTION.
    num_frames : int, optional
        DESCRIPTION. The default is 5.

    Raises
    ------
    TypeError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    blended : np ndarray.

    """
    from ..routines.functions_detection import load_pickle_file

    
    if name_positions is None or name_positions.empty:
        raise TypeError("name_positions is None or empty.")
    if not isinstance(num_index,int):
        raise TypeError(f"'num_index' should be an int but got {type(num_index)}")
    if not isinstance(num_frames,int):
        raise TypeError(f"'num_frames' should be an int but got {type(num_frames)}")
         
    pos = name_positions.iloc[num_index]["POS"]
    mw = name_positions.iloc[num_index]["MW"]
    frame = int(name_positions.iloc[num_index]["Frame"])
    
    # load masks
    matched_file_masks = FindMatchingFile(dir_results, f"{pos}.*{mw}", ext=".pkl")
    if matched_file_masks:
        y = load_pickle_file(dir_results, matched_file_masks)
        masks_array = np.dstack(y["masks"])
        masks_colored = masks_array[:,:,frame-num_frames:frame+num_frames+1]
        del y
      
    # load video original
    matched_file_video = FindMatchingFile(dir_wells, pos, ext=".pkl")
    if matched_file_video:
        x = load_pickle_file(dir_wells, matched_file_video)
        frame=int(frame)
        video = x[mw]["crop"][:,:,frame-num_frames:frame+num_frames+1]
        del x
        blended = AlphaBlendResize(video,\
            masks_colored, alpha=0.2)

        return blended
    else:
        raise ValueError(f"No file to open for {pos}")    

def GeneratePalette(n_labels: int) -> dict:
    """
    Palette
    """
    import colorsys
    import matplotlib.pyplot as plt
    base_cmap = plt.get_cmap('tab20', n_labels)
    palette = {}
    for i in range(n_labels):
        r, g, b, _ = base_cmap(i)
        # Converti RGB 0-1 in HLS
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        # Aumenta saturazione (max 1)
        s = min(1, s * 1.5)
        # Aumenta luminosità se vuoi
        l = min(1, l * 1.1)
        # Torna a RGB
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        palette[i] = (int(r*255), int(g*255), int(b*255))
    return palette

def LabelToColor(mask: np.ndarray, color_map: dict) ->  np.ndarray:
    """
    Convert a label mask (H,W,N_frames) with integer labels to RGB (H,W,3,N_frames).
    """
    H, W, N = mask.shape
    colored_mask = np.zeros((H, W, 3, N), dtype=np.uint8)
    for f in range(N):
        for label, color in color_map.items():
            colored_mask[:, :, :, f][mask[:, :, f] == label] = color
    return colored_mask


def AlphaBlendResize(background: np.ndarray, overlay:np.ndarray,\
    alpha: float =0.5, resize_to: tuple =None, color_map: dict = None) -> np.ndarray:
    import cv2
    if color_map is None:
        num_labels = int(overlay.max()) + 1
        color_map = GeneratePalette(num_labels)

    # Check background shape
    if background.ndim == 3:  # (H, W, N_frames)
        H_bg, W_bg, N_frames = background.shape
        C_bg = 3  # assume 3 channels needed
        # Add channel dimension with 3 channels, repeating grayscale or single channel 3 times
        background = np.repeat(background[:, :, np.newaxis, :], 3, axis=2)
    else:
        H_bg, W_bg, C_bg, N_frames = background.shape
        
    H_ov, W_ov, N_frames_ov = overlay.shape

    assert C_bg == 3, "Background must have 3 channels"
    assert N_frames == N_frames_ov, "Number of frames must match"

    overlay_rgb = LabelToColor(overlay, color_map)
    
    if resize_to is not None:
        target_size = tuple(map(int, resize_to))  
    
        background_resized = np.zeros((target_size[1], target_size[0], C_bg, N_frames), dtype=background.dtype)
    
        for f in range(N_frames):
            background_resized[:, :, :, f] = cv2.resize(background[:, :, :, f], target_size, interpolation=cv2.INTER_AREA)
        background = background_resized
    else:
        target_size = (W_bg, H_bg)
    
    output = background.copy()

    for f in range(N_frames):
        resized_ov = cv2.resize(overlay_rgb[:, :, :, f], target_size, interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(overlay[:, :, f].astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

        bg_frame = output[:, :, :, f].astype(float)
        ov_frame = resized_ov.astype(float)

        mask = (resized_mask != 0)

        blended = bg_frame.copy()
        blended[mask] = (1 - alpha) * bg_frame[mask] + alpha * ov_frame[mask]

        output[:, :, :, f] = blended.astype(background.dtype)

    return output

def PlotCurve(name_positions: pd.DataFrame, num_index: int,\
    dir_results: Union[str, Path], sep: str = ";"):
    
    import plotly.express as px
    import pandas as pd
    
    if name_positions is None or name_positions.empty:
        raise TypeError("name_positions is None or empty.")
    if not isinstance(num_index,int):
        raise TypeError(f"'num_index' should be an int but got {type(num_index)}")

    
    pos = name_positions.iloc[num_index]["POS"]
    mw = name_positions.iloc[num_index]["MW"]
    frame = int(name_positions.iloc[num_index]["Frame"])

    matched_file_csv = FindMatchingFile(dir_results, f"{pos}.*{mw}", ext=".csv")

    # load file
    df = pd.read_csv(matched_file_csv, sep=sep)
    output = df["Output YOLO"]
    correction = df["Correction"]
    assert(output.shape[0]==correction.shape[0])
    x = np.arange(output.shape[0])
    
    df_output = pd.DataFrame({'x': x, 'y': output, 'line': 'output'})
    df_correction = pd.DataFrame({'x': x, 'y': correction, 'line': 'correction'})
    df = pd.concat([df_output, df_correction])
    
    fig = px.line(df, x='x', y='y', color='line', title=f"{pos} {mw}")
    
    # intial range around frame
    fig.update_layout(xaxis_range=[frame-5, frame+5])
    
    return fig


def VisualizeVideoBBScore(pos: str, mw: str,\
    dir_wells: Union[str, Path], dir_results: Union[str, Path],\
    dim_resize: tuple) -> np.ndarray :

    from ..routines.functions_detection import load_pickle_file
    from collections import defaultdict
    import numpy as np
    import cv2


    # load masks
    matched_file_masks = FindMatchingFile(dir_results, f"{pos}.*{mw}", ext=".pkl")
    if not matched_file_masks:
        raise ValueError(f"No mask file found for {pos}, {mw}")
    
    y = load_pickle_file(dir_results, matched_file_masks)
    
    matrix_centers = y.get("matrix_centers", None)
    
    use_masks = matrix_centers is None or not isinstance(matrix_centers, list) or len(matrix_centers) == 0

    
    # load video original
    matched_file_video = FindMatchingFile(dir_wells, pos, ext=".pkl")
    if not matched_file_video:
        raise ValueError(f"No video file found for {pos}")
    
    x = load_pickle_file(dir_wells, matched_file_video)
    video = x[mw]["crop"]
    del x
    
    H, W, N = video.shape

    if use_masks:
        masks = y["masks"]  # shape: (H, W, N)
        del y

        masks = np.stack(masks, axis=-1)
        # blend masks using provided function
        video_annotated = AlphaBlendResize(
            background=video,
            overlay=masks,
            alpha=0.3,
            resize_to=dim_resize
        )
    else:
        # use bounding boxes
        grouped = defaultdict(list)
        for item in matrix_centers:
            key = item[0]
            grouped[key].append(item)

        for i in range(N):
            if i not in grouped:
                grouped[i] = []
        grouped = dict(sorted(grouped.items()))
        assert (video.shape[-1] == len(grouped))

        video_annotated = np.zeros((*dim_resize, 3, N), dtype=np.uint8)

        for i, (key, frame) in enumerate(grouped.items()):
            frame_video = cv2.resize(video[:, :, i], dim_resize)
            frame_rgb = cv2.cvtColor(frame_video, cv2.COLOR_GRAY2BGR)
            for iteration in frame:
                info_centers = iteration[2]
                for center in info_centers:
                    coord = tuple(map(int, center[0]))  # (x, y)
                    bb = list(map(int, center[1]))      # (x, y, w, h)
                    score = center[2]

                    # center point
                    cv2.circle(frame_rgb, coord, 3, (0, 255, 0), -1)

                    # bounding box
                    x, y, w, h = bb
                    cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 0, 255), 1)

                    # score
                    cv2.putText(frame_rgb, f'{score:.2f}', (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            video_annotated[:, :, :, i] = frame_rgb

    return video_annotated

    


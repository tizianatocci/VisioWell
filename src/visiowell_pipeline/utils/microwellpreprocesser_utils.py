# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:07:58 2025

@author: tizia
"""
import os
USE_CUPY = os.environ.get("USE_CUPY", "1") != "0"
if USE_CUPY:
    try:
        import cupy as cp
        from cupyx.scipy.fft import fft2, ifft2, fftshift
        from cupyx.scipy.ndimage import shift, rotate
        _ = cp.zeros(1)  # Verifica accesso GPU
        xp = cp 
        print("Using CuPy (GPU)")
    except Exception:
        import numpy as xp
        from scipy.fft import fft2, ifft2, fftshift
        from scipy.ndimage import shift, rotate
        print("Using NumPy + SciPy FFT (CPU)")
else:
    import numpy as xp
    from scipy.fft import fft2, ifft2, fftshift
    from scipy.ndimage import shift, rotate
    print("Using NumPy (forced)")


def ReadImg(folder, filename):
    """
    Read image in a numpy array by NOT modifying the pixel values as 
    "cv2.IMREAD_UNCHANGED" is used.

    Parameters
    ----------
    folder : string. Folder where the image is.
    filename : string. Name of file.
   
    Raise
    -----
    TypeError if folder and filename are nos strings.
    FileNotFoundError if the file path does not exist.
    ValueError if the image was not correctly loaded.
    
    Returns
    -------
    img : numpy array. Image loaded.

    """
    import os
    import cv2
    
    if not isinstance(folder, str):
        raise TypeError(f"'folder' must be a string, got {type(folder)}")
    if not isinstance(filename, str):
        raise TypeError(f"'filename' must be a string, got {type(filename)}")
    
    file_path = os.path.join(folder, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("The image was not correctly loaded.")

    return img


def LoadImagesFromList(img_list, folder):
    """
    Read images in list.

    Parameters
    ----------
    img_list : list. List of strings, which are the names of images in the folder.
    folder : string. Fold where the images are
    
    Returns
    -------
    images : numpy array.

    """
    if not isinstance(img_list, list):
        raise TypeError(f"'img_list' must be a list, got {type(img_list)}")
    
    for filename in img_list:
        if not isinstance(filename, str):
            raise TypeError(f"Each element in 'img_list' must be a string, got {type(filename)}")
    
    if not isinstance(folder, str):
        raise TypeError(f"'folder' must be a string, got {type(folder)}")


    images = []
    for filename in img_list:
        img = ReadImg(folder, filename)
        if img is not None:
            images.append(img)
    return images



def TransposeArray(images_array, order_tuple=None):
    """
    Transpose image in order that the last dimension is the number of frames
    if by default the number of frames is the first dimension.

    Parameters
    ----------
    images_array : numpy array. Image to transpose
    order_tuple : tuple. By default, (1,2,3,0) of the first dimension has to become the last.

    Returns
    -------
    images_array : numpy array. Transposed image.

    """
    if not isinstance(images_array, xp.ndarray):
        raise TypeError(f"'images_array' expected to be a xp array but got {type(images_array)}")
    # if the array is made up of grayscale images
    if images_array.ndim == 3:
        if order_tuple==None:
            order_tuple=(1,2,0)
        images_array=xp.transpose(images_array, order_tuple)
    # if the array is made up of RGB images
    elif images_array.ndim == 4:
        if order_tuple==None:
            order_tuple=(1,2,3,0)
        images_array=xp.transpose(images_array, order_tuple)        
    return images_array


def LoadImage(path_exp_pos):
    """
    Function to load all .tif files inside a folder.

    Parameters
    ----------
    path_exp_pos : string.

    Returns
    -------
    images_array : numpy or cupy array.

    """
    import os

    if not isinstance(path_exp_pos, str):
        raise TypeError(f"path_exp_pos must be a string, got {type(path_exp_pos)}")
    if not os.path.isdir(path_exp_pos):
        raise FileNotFoundError(f"'{path_exp_pos}' does not exist or is not a folder.")
        
    name_files=[f for f in os.listdir(path_exp_pos) if f.endswith('.tif')]  # retrieve names of files in folder
    if not name_files:
        raise FileNotFoundError(f"The directory '{path_exp_pos}' does not contain .tif files.")
        
    #sorted_files=order_name_files(name_files, '_I', '_T') # reorder based on times
    images_list=LoadImagesFromList(name_files, path_exp_pos) # load images from list of names
    images_array = xp.array(images_list) # convert list to a numpy array
    images_array = TransposeArray(images_array) # transpose so that the last dimension is the number of frames
    if images_array is None:
        raise ValueError(f"Something went wrong with loading the files from directory '{path_exp_pos}'.")
        
    return images_array


def LoadStack(path_exp_pos, num_tile=None):
    """
    Load a .tif stack or a folder of .tifs or a .czi file.
    
    Parameters
    ----------
    path_exp_pos : string. It contains the directory and name of file.
    
    Raise
    -----
    TypeError if path_exp_pos is not a string.
    FileNotFoundError  if the path does not exist.
    ValueError if the stack is not correctly loaded.
    

    Returns
    -------
    if path_exp_pos points to a .tif or folder
        images_array : numpy or cupy array. Video loaded, the last dimension is a the number of frames.
    if path_exp_pos points to a czi
    """
    
    import os
    import tifffile
    from skimage import io

    if not isinstance(path_exp_pos, str):
        raise TypeError(f"path_exp_pos must be a string, got {type(path_exp_pos)}")
    if not os.path.isfile(path_exp_pos) and not os.path.isdir(path_exp_pos):
        raise FileNotFoundError(f"'{path_exp_pos}' does not exist or is neither a file nor a folder.")
    
   # if file is .czi
    if os.path.isfile(path_exp_pos) and path_exp_pos.endswith(".czi"):
        return LoadCZISingleTile(path_exp_pos, num_tile)
   
    
    # if path_exp_pos is a .tif stack
    elif os.path.isfile(path_exp_pos): 
        try:
            images_array = io.imread(path_exp_pos) # read images
            images_array = TransposeArray(images_array) # transpose so that the last dimension is the number of frames
            
        except xp.core._exceptions._ArrayMemoryError as e:
            print(f"Memory error. Error details: {e}")
            with tifffile.TiffFile(path_exp_pos) as tif_file:
                available_frames = len(tif_file.pages) 
                if available_frames == 0:
                    raise ValueError(f"No frames found in the tif file at '{path_exp_pos}'.")
                
                frame_shape = tif_file.pages[0].shape
                if not isinstance(frame_shape, tuple) or len(frame_shape) != 2:
                    raise ValueError(f"Unexpected frame shape: {frame_shape}. Expected a 2D image per frame.")
    
                images_array = xp.empty((*frame_shape, available_frames), dtype=xp.uint16)
                for i, page in enumerate(tif_file.pages[:available_frames]):
                    images_array[:,:,i] = page.asarray()
        print(f"The .tif file was correctly loaded.\nSize: {images_array.shape}")
        return images_array
    
    # if path_exp_pos is a folder of .tif images
    elif os.path.isdir(path_exp_pos): 
        images_array = LoadImage(path_exp_pos)
        print(f"The folder was correctly loaded.\nSize: {images_array.shape}")
    
        return images_array


def LoadCZI_TileIterator(path_file):
    """
    Generator that yields one tile (M) at a time from a .czi file,
    each with all timepoints (T) included. The output shape is (Y, X, T).

    Parameters
    ----------
    path_file : str
        Path to the .czi file.

    Yields
    ------
    tuple (int, xp.ndarray)
        - The tile index (M)
        - Array with shape (Y, X, T)
    """
    
    from aicspylibczi import CziFile
    import os

    if not os.path.isfile(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    with CziFile(path_file) as czi_mozaic:
        dims = czi_mozaic.dims
        sizes = czi_mozaic.size

        if 'M' not in dims:
            raise ValueError("The .czi file does not contain 'M' dimension (tiles).")
        if 'T' not in dims:
            raise ValueError("The .czi file does not contain 'T' dimension (timepoints).")

        idx_m = dims.index('M')
        n_tiles = sizes[idx_m]

        idx_t = dims.index('T')
        n_timepoints = sizes[idx_t]

        for m in range(n_tiles):
            tile_stack = [
                xp.asarray(
                    xp.squeeze(czi_mozaic.read_image(M=m, T=t)[0])
                )
                for t in range(n_timepoints)
            ]
            tiff_stack = xp.stack(tile_stack, axis=-1)  # shape (Y, X, T)
            yield m, tiff_stack



def LoadCZISingleTile(path_file, num_tile):
    """
    Load a single tile from a .czi file as an array with shape (Y, X, T).

    Parameters
    ----------
    path_file : str
        Full path to the .czi file.
    num_tile : int
        Tile index to load (0-based). Must be within the number of tiles in the file.

    Returns
    -------
    xp.ndarray
        Array with shape (Y, X, T), where T is the number of timepoints.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If num_tile is out of bounds or required dimensions are missing.
    """
    from aicspylibczi import CziFile

    if not os.path.isfile(path_file):
        raise FileNotFoundError(f"File not found: {path_file}")

    czi_mozaic =  CziFile(path_file)
    dims = czi_mozaic.dims
    sizes = czi_mozaic.size

    # Check for 'M' dimension (tiles)
    if 'M' not in dims:
        raise ValueError("The .czi file does not contain 'M' dimension (tiles).")
    idx_m = dims.index('M')
    n_tiles = sizes[idx_m]
    
    if num_tile < 0 or num_tile >= n_tiles:
        raise ValueError(f"num_tile must be between 0 and {n_tiles - 1}, got {num_tile}")

    # Check for 'T' dimension (timepoints)
    if 'T' not in dims:
        raise ValueError("The .czi file does not contain 'T' dimension (timepoints).")
    idx_t = dims.index('T')
    n_timepoints = sizes[idx_t]

    # Load each timepoint for the specified tile
    tile_stack = [
        xp.asarray(  # convert to numpy or cupy array
            xp.squeeze(czi_mozaic.read_image(M=num_tile, T=t)[0])
        )
        for t in range(n_timepoints)
    ]

    # Stack along last axis to get shape (Y, X, T)
    tiff_stack = xp.stack(tile_stack, axis=-1)

    return tiff_stack



def RetrieveParameters():
    pass


def CorrectDriftVideo(images_array, mode_shift, nptype=xp.uint8, ref_frame=0):
    """
    Correct drift for whole video with crosscorrelation with respect to the reference frame.

    Parameters
    ----------
    images_array : numpy or cupy array. Video to correct.
    mode_shift : string. "constant", "wrap", "nearest"...
    nptype : np or cp dtype. The default is xp.uint8.
    ref_frame: int. Index of frame considered as reference for drift correction.
        The default is ref_frame=0, which means we correct with respect to the first frame.   
    
    Raise
    -----
    TypeError if arguments are not in the expected format.
    
    Returns
    -------
    corrected_array : numpy array. Video corrected.
    """
    
    if not isinstance(images_array, xp.ndarray):
        raise TypeError(f"images_array must be a numpy or cupy ndarray, got {type(images_array)}")
    
    if not isinstance(mode_shift, str):
        raise TypeError(f"mode_shift must be a string, got {type(mode_shift)}")
    
    #if not isinstance(nptype, np.dtype):
    #    raise TypeError(f"nptype must be a NumPy dtype object (e.g., np.uint8), got {type(nptype)}")
    
    if not isinstance(ref_frame, int):
        raise TypeError(f"ref_frame must be an int, got {type(ref_frame)}")

    corrected_array = xp.zeros(images_array.shape).astype(nptype)
    for i in range(images_array.shape[-1]):
        if i%100==0:
            print(f"Correcting frame for drift: {i+1}/{images_array.shape[-1]}")
        corrected_array[:,:,i] = CorrectDrift(images_array[:,:,ref_frame], images_array[:,:,i], mode_shift = mode_shift)
    
    return corrected_array


def RangeAdjustment(frames, mode="no_adjust"):
    """
    Pipeline for converting a video to 8 bit with modified pixel range if necessary.
    
    Parameters
    ----------
    frames : numpy or cupy array. Video to convert to 8 bit.
    mode : string. The default is "no_adjust". If mode="no_adjust", the video is converted from 16 bit to 8 bit.
    If mode="per_frame", the video is converted to 8 bit with the following formula:
        frame = (frame-np.min(frame))/(np.max(frame)-np.min(frame))*255
    If mode="whole", the video is converted to 8 bit with the following formula:
        frame = (frame-min_value)/(max_value-min_value)*255 where
        max_value = np.max(frames)
        min_value = np.min(frames)
        
    Raise
    -----
    TypeError if arguments are not in the expected format.
    ValueError if frames is not xp.uint16 or mode is not among the predefined choices.

    Returns
    -------
    frames_new : numpy or cupy array. Video in 8bit.

    """
    
    if not isinstance(frames, xp.ndarray):
        raise TypeError(f"Expected 'frames' to be a numpy or cupy array, but got {type(frames)}.")
    if frames.dtype != xp.uint16:
        raise ValueError(f"Expected 'frames' to be of type xp.uint16, but got {frames.dtype}.")
    if not isinstance(mode, str):
        raise TypeError(f"Expected 'mode' to be a string, but got {type(mode)}.")
    if mode not in ["per_frame", "whole", "no_adjust"]:
        raise ValueError(f"Invalid 'mode' value: '{mode}'. Supported values are: ['no_adjust'].")

    frames_new = xp.zeros(frames.shape).astype(xp.uint8)
    if mode=="per_frame":
        for i in range(frames.shape[-1]):
            frame = frames[:,:,i]
            frame = (frame-xp.min(frame))/(xp.max(frame)-xp.min(frame))*255
            frame=frame.astype(xp.uint8)
            frames_new[:,:,i] = frame
            if i%100==0: print(f"Adjusting pixel values: {i+1}/{frames.shape[-1]}")
    elif mode=="whole":
        max_value = xp.max(frames)
        min_value = xp.min(frames)
        for i in range(frames.shape[-1]):
            frame = frames[:,:,i]
            frame = (frame-min_value)/(max_value-min_value)*255
            frame=frame.astype(xp.uint8)
            frames_new[:,:,i] = frame
            if i%100==0: print(f"Adjusting pixel values: {i+1}/{frames.shape[-1]}")
    elif mode=="no_adjust":
        frames_new = (frames/255).astype(xp.uint8) 
        
    return frames_new
    

def CorrectDrift(im1, im2, mode_shift='wrap'):
    """
    Correct drift correction between two images.

    Parameters
    ----------
    im1 : numpy array. Image taken as reference.
    im2 : numpy array. Image to correlate to im1.
    mode_shift : string. The default is 'wrap'.

    Returns
    -------
    corrected_im2 : numpy array. Im2 after drift correction.

    """
    if not isinstance(im1, xp.ndarray):
        raise TypeError(f"'im1' expected to be a np or cp array but got {type(im1)}")
    if not isinstance(im2, xp.ndarray):
        raise TypeError(f"'im2' expected to be a np or cp array but got {type(im2)}")
    if not isinstance(mode_shift, str):
        raise TypeError(f"'mode_shift' expected to be a string but got {type(mode_shift)}")
    
    #breakpoint()
    # Size of the reference frame
    sy, sx = im1.shape

    # Using fft cross correlations to detect the moving distance
    fftim1 = fft2(im1)
    fftim2 = fft2(im2)
    cc = fftshift(ifft2(fftim1 * xp.conj(fftim2)))
    shiftY, shiftX = xp.unravel_index(xp.argmax(cc), cc.shape)
    
    # Fix rounding to the nearest integer toward zero
    shiftY = int(shiftY - xp.fix(sy / 2)) # in matlab: -1
    shiftX = int(shiftX - xp.fix(sx / 2)) #in matlab: -1


    # Corrected image by circular shifting
    corrected_im2 = shift(im2, [shiftY, shiftX], mode=mode_shift)
    
    
    return corrected_im2

def RotateVideo(frames, rotation_angle):
    """
    Rotate a video (or sequence of frames) by a given angle around the specified axes.

    Parameters
    ----------
    frames : xp.ndarray
        A multidimensional array containing the video frames.
        This can be either a NumPy or CuPy array, depending on the current backend (`xp`).
        Axes (0, 1) typically correspond to the spatial dimensions (height, width).

    rotation_angle : float
        Rotation angle in degrees (positive = counterclockwise).

    Returns
    -------
    rotated_frames : xp.ndarray
        The rotated frames, with the same shape as `frames`.

    Raises
    ------
    TypeError
        If `frames` is not an instance of `xp.ndarray` or if `rotation_angle` is not a number.
    """
    if not isinstance(frames, xp.ndarray):
        raise TypeError(f"'frames' must be an {xp.__name__}.ndarray, got {type(frames).__name__}")
    if not isinstance(rotation_angle, (int, float)):
        raise TypeError(f"'rotation_angle' must be a number, got {type(rotation_angle).__name__}")
    
    frames = frames.astype(xp.float32) / 255.0
    frames = rotate(frames, angle=rotation_angle, axes=(0, 1), reshape=False)
    frames = xp.clip(frames * 255, 0, 255).astype(xp.uint8)

    return frames


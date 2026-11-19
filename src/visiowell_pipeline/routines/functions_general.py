# -*- coding: utf-8 -*-
"""
Functions used in several scripts.
Author: Tiziana Tocci
"""

import os
import cv2
import numpy as np
#from scipy.fftpack import fft2, ifft2, fftshift
#from scipy.ndimage import shift
import re
import pickle

USE_CUPY = os.environ.get("USE_CUPY", "1") != "0"
if USE_CUPY:
    try:
        import cupy as cp
        from cupyx.scipy.fft import fft2, ifft2, fftshift
        from cupyx.scipy.ndimage import shift
        _ = cp.zeros(1)  # Verifica accesso GPU
        xp = cp 
        print(f"Using CuPy (GPU)")
    except Exception:
        import numpy as xp
        from scipy.fft import fft2, ifft2, fftshift
        from scipy.ndimage import shift
        print("Using NumPy + SciPy FFT (CPU)")
else:
    import numpy as xp
    from scipy.fft import fft2, ifft2, fftshift
    from scipy.ndimage import shift
    print("Using NumPy (forced)")


def set_working_directory(name_pc, which_network="yolo_tf"):
    if name_pc not in ("Milagres", "Tiziana", "Morgane"):
        raise ValueError("Invalid computer name. Working directory not set.")
    
    import os
    if name_pc == "Milagres":
        if which_network=="yolo_tf":
            working_dir = "/mnt/c/Users/team_/Documents/Tiziana/VisioWell"
        else:
            working_dir = "/Users/team_/Documents/Tiziana/VisioWell"
    elif name_pc in ("Tiziana", "Morgane"):
        from pathlib import Path
        if name_pc=="Morgane":
            base_path = Path.home() / "Desktop"
        elif name_pc=="Tiziana":
            base_path = Path.home() / "Documents/Curie"
        working_dir = base_path / "Python for Morgane/Code"
    os.chdir(working_dir)    

    return working_dir


def write_file(output_file, name_file_to_write,is_path=1):
    if not isinstance(output_file, str):
        raise TypeError(f"'output_file' must be a string, but got {type(output_file)}")
    if not isinstance(name_file_to_write, str):
        raise TypeError(f"'name_file_to_write must be a string, but got {type(name_file_to_write)}")
    if not is_path in (0,1):
        raise TypeError(f"'is_path must be 0 or 1, but got {is_path}")
    
    with open(output_file, "a") as f:
        if is_path:
            name_position = os.path.basename(os.path.normpath(name_file_to_write))
        else:
            name_position=name_file_to_write
        f.write(name_position+ "\n")


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image by keeping the aspect ratio,

    Parameters
    ----------
    image : numpy array. 
    width : int.
    height : int
    inter : The default is cv2.INTER_AREA.

    Returns
    -------
    img: numpy array. Image resized.

    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        
    img = cv2.resize(image, dim, interpolation=inter)
    
    return img

def display_video(matrix, user_width=630):
    """
    Display video (3d or 4d.)
    Press bar to pause.
    Press p to go to previous frame.
    Press n to go to next frame.
    Press r to restart.
    Press q to close video.

    Parameters
    ----------
    matrix : numpy array. Video.
    user_width : int. The default is 630. Width for displaying video.

    Returns
    -------
    None.

    """
    frame_count = matrix.shape[-1] #matrix.shape[2]
    current_frame = 0
    paused = False

    while True:
        # display frame
        if matrix.ndim == 3:
            frame = matrix[:,:,current_frame]
        elif matrix.ndim == 4:
            frame = matrix[:,:,:,current_frame]
        frame = ResizeWithAspectRatio(frame, width=user_width) 
        #breakpoint()
        cv2.imshow('Video', frame)
        cv2.setWindowTitle('Video', f'Frame {current_frame+1}/{frame_count}')

        # receive key
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Pause/Resume
            paused = not paused
        elif key == ord('r'):  # Restart
            current_frame = 0
        
        if paused:
            #key = cv2.waitKey(30) & 0xFF
            if key == ord('n'):  # Next frame (when paused)
                current_frame  = min(current_frame + 1, frame_count - 1)
                #print('avanti')
            elif key == ord('p'):  # Previous frame (when paused)
                current_frame  = max(current_frame - 1, 0)
                #print('indietro')
        else:
            current_frame = (current_frame + 1) % frame_count
            

    cv2.destroyAllWindows()
    
def read_img(folder, filename):
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


def load_images_from_list(img_list, folder):
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
        img = read_img(folder, filename)
        if img is not None:
            images.append(img)
    return images


def drift_correction(im1, im2, mode_shift='wrap'):
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

    # Visualize cross-correlation (commented out)
    # import matplotlib.pyplot as plt
    # plt.imshow(np.abs(cc), cmap='gray')
    # plt.plot(np.fix(sx / 2), np.fix(sy / 2), 'r+')
    # plt.show()
    # plt.figure()
    # plt.contourf(cc)
    # plt.colorbar()
    # plt.show()

    # Corrected image by circular shifting
    corrected_im2 = shift(im2, [shiftY, shiftX], mode=mode_shift)
    
    # Show result (commented out)
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(im1, cmap='gray')
    # plt.title('Reference Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(corrected_im2, cmap='gray')
    # plt.title('Corrected Image')
    # plt.show()
    
    return corrected_im2


def find_substring(user_string, first_substring, second_substring):
    """
    Find substring between two substrings with (.*).


    Parameters
    ----------
    user_string : string. String from where to extract the substring of interest.
    first_substring : string. First substring who comes before the substring of interest.
    second_substring : string. Second substring who comes after the substring of interest.

    Returns
    -------
    result.group(1): substring of interest.

    """
    if not isinstance(user_string, str):
         raise TypeError(f"'user_string' expected to be a string but got {type(user_string)}")
    if not isinstance(first_substring, str):
         raise TypeError(f"'first_substring' expected to be a string but got {type(first_substring)}")
    if not isinstance(second_substring, str):
         raise TypeError(f"'second_substring' expected to be a string but got {type(second_substring)}")
       
    result = re.search(f'{first_substring}(.*){second_substring}', user_string)
    return result.group(1)



def order_name_files(name_files, first_substring, second_substring):
    """
    Order name of files  based on the numbers contained in their name.
    The problem is that files like file_1, file_2,... file_10, file_11...
    will be ordered as file_1, file_10, file_11,... file_2, file_20...
    so I want to avoid that.

    Parameters
    ----------
    name_files : list of strings. List with name of files.
    first_substring : string. Substring before number of interest.
    second_substring : string. Substring after number of interest.

    Returns
    -------
    sorted_files : list of strings. Sorted list based on numbers contained in the name of filtes.

    """
    
    time_points = []
    for i in range(len(name_files)):
        result = find_substring(name_files[i], first_substring, second_substring)
        time_points.append((int(result), name_files[i]))
    time_points.sort()
    sorted_files = [name for _, name in time_points]
    return sorted_files


def transpose_array_img(images_array, order_tuple=None):
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


def im2double(img_array):
    """
    Normalize the images (a single image or a video) between 0 and 1.
    If the image is uint8, the values are divided by 2^8-1.
    If the image is uint16, the values are divided by 2^16-1.

    Parameters
    ----------
    img_array : numpy array. It corresponds to the image or to the images (=video).

    Returns
    -------
    img_array : numpy array. It corresponds to the normalized image or images.

    """
    if img_array.dtype==xp.uint8:
        img_array = img_array/(2**8-1)
    elif img_array.dtype==xp.uint16:
        img_array= img_array/(2**16-1)
    else:
        raise("The image is neither a xp.uint8 nor a xp.uint16.")
        
    return img_array        


        
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
        

def create_class_mapping(class_ids):
    """
    Map each class name to a unique numerical identifier

    Parameters
    ----------
    class_ids : A list of class names.

    Returns
    -------
    class_mapping : The mapping class-number.

    """
    class_mapping = dict(zip(range(len(class_ids)), class_ids))
    return class_mapping
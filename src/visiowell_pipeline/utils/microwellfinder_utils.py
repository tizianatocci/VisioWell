# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:39:24 2025

@author: tizia
"""

def FindCircles(img, dp_value, min_dist, param_1, param_2, min_rad, max_rad):
    import numpy as np
    import cv2
    
    """
    Apply CHT algorithm to identify circles in the image.

    Parameters
    ----------
    img : numpy ndarray. Image.
    dp_value : int. CHT parameter.
    min_dist : int. Minimum distance between wells detected.
    param_1 : int. CHT parameter.
    param_2 : int. CHT parameter.
    min_rad : int. minimum radius.
    max_rad : int. maximum radius.
    
    Raise
    -----
    TypeError if arguments are not in the expected format.
    ValueError if min_rad bigger or equal to max_rad or if circles is None.


    Returns
    -------
    circles : numpy ndarray.

    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected 'img' to be a numpy array, but got {type(img)}.")
    if img.ndim != 2:
        raise ValueError(f"Expected 'img' to be a 2D array, but got shape {img.shape}.")
    if not isinstance(dp_value, int):
        raise TypeError(f"Expected 'dp_value' to be an integer, but got {type(dp_value)}.")
    if not isinstance(min_dist, int):
        raise TypeError(f"Expected 'min_dist' to be an integer, but got {type(min_dist)}.")
    if not isinstance(param_1, int):
        raise TypeError(f"Expected 'param_1' to be an integer, but got {type(param_1)}.")
    if not isinstance(param_2, int):
        raise TypeError(f"Expected 'param_2' to be an integer, but got {type(param_2)}.")
    if not isinstance(min_rad, int):
        raise TypeError(f"Expected 'min_rad' to be an integer, but got {type(min_rad)}.")
    if not isinstance(max_rad, int):
        raise TypeError(f"Expected 'max_rad' to be an integer, but got {type(max_rad)}.")
    if min_rad >= max_rad:
        raise ValueError(f"Expected 'min_rad' to be less than 'max_rad', but got min_rad={min_rad} and max_rad={max_rad}.")
    
    circles = cv2.HoughCircles(img ,cv2.HOUGH_GRADIENT, dp_value,min_dist,
                               param1=param_1,param2=param_2,minRadius=min_rad,maxRadius=max_rad)
    if circles is None:
        raise ValueError("No microwells were identified.")
    
    return circles


def ShowCircles(img, circles, title=None, radius=None):
    """
    Show circles identified in a new image.

    Parameters
    ----------
    img : numpy array. Image where to show circles.
    circles : numpy array. Centers' coordinates.
    title: string. Title to display in the plot. The default is title=None.
    radius : int. Radii if not found in circles array. The default is radius=None.
    
    Returns
    -------
    None.

    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected 'img' to be a numpy array, but got {type(img)}.")
    if img.ndim != 2:
        raise ValueError(f"Expected 'img' to be a 2D array, but got shape {img.shape}.")
    if not isinstance(circles, np.ndarray):
        raise TypeError(f"Expected 'circles' to be a numpy array, but got {type(circles)}.")
    if title is not None and not isinstance(title, str):
        raise TypeError(f"Expected 'title' to be a string, but got {type(title)}.")
    if radius is not None and not isinstance(radius, int):
        raise TypeError(f"Expected 'radius' to be a numpy array, but got {type(radius)}.")
          
    circles = np.uint16(np.around(circles))
    for i in range(circles.shape[0]):
        # draw the outer circle
        if radius is None:
            radius=circles[i,2]
        cv2.circle(img,(circles[i,0],circles[i,1]),radius,(255,255,255), 10)
        # draw the center of the circle
        cv2.circle(img,(circles[i,0],circles[i,1]),2,(255,255,255), 10)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    if not (title is None): plt.title(title)
    plt.axis("off")
    plt.show()    



def FindMicrowellsCHT(img, dp_value,\
    min_dist, param_1, param_2, min_rad, max_rad,\
    flag_plot=False, title=None):
    import numpy as np
    """
    Find microwells in image and show result.

    Parameters
    ----------
    images_array : numpy array. Video.
    dp_value : int. CHT parameter.
    min_dist : int. Minimum distance between circles.
    param_1 : int. First parameter for CHT.
    param_2 : int. Second parameter for CHT.
    min_rad : int. Minimum radius for finding circles.
    max_rad : int. Maximum radius for finding circles.
    ref_frame: int. 
    flag_plot: 0 or 1.
    title: string. Title to display in the plot. The default is title=None.

    Returns
    -------
    circles : numpy array, float32. x coordinate of center, y coordinate fo center, radius.

    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected 'img' to be a numpy array, but got {type(img)}.")
    if not isinstance(dp_value, int):
        raise TypeError(f"Expected 'dp_value' to be an integer, but got {type(dp_value)}.")
    if not isinstance(min_dist, int):
        raise TypeError(f"Expected 'min_dist' to be an integer, but got {type(min_dist)}.")
    if not isinstance(param_1, int):
        raise TypeError(f"Expected 'param_1' to be an integer, but got {type(param_1)}.")
    if not isinstance(param_2, int):
        raise TypeError(f"Expected 'param_2' to be an integer, but got {type(param_2)}.")
    if not isinstance(min_rad, int):
        raise TypeError(f"Expected 'min_rad' to be an integer, but got {type(min_rad)}.")
    if not isinstance(max_rad, int):
        raise TypeError(f"Expected 'max_rad' to be an integer, but got {type(max_rad)}.")
    if min_rad >= max_rad:
        raise ValueError(f"Expected 'min_rad' to be less than 'max_rad', but got min_rad={min_rad} and max_rad={max_rad}.")
    if not isinstance(flag_plot, bool):
        raise ValueError("flag_plot must be a boolean value (True or False)")
    if title is not None and not isinstance(title, str):
        raise TypeError(f"Expected 'title' to be a string, but got {type(title)}.")
        
    circles = FindCircles(img, dp_value, min_dist, param_1, param_2, min_rad, max_rad) # find wells according to parameters circles = find_circles(img, 1, 150, 50, 30, 120, 130)
    if flag_plot:
        ShowCircles(img, np.squeeze(circles), title)
        
    return circles
   


def GetCoordinateRectanglesFromCircleCenter(circle, sz_rect):
    """
    Pipeline for cutting rectangle for each microwell and filtering them

    Parameters
    ----------
    circle : numpy ndarray.
    sz_rect : int.

    Returns
    -------
    rectangle_xy : numpy ndarray.

    """
    import numpy as np
    if not isinstance(circle, np.ndarray):
        raise TypeError(f"Expected 'circle' to be a numpy array, but got {type(circle)}.")
    if not isinstance(sz_rect, int):
        raise TypeError(f"Expected 'sz_rect' to be a numpy array, but got {type(sz_rect)}.")
            
    # get coordinates of the starting point (top left
    first_x = circle[0]- sz_rect/2
    first_y = circle[1]- sz_rect/2
    rectangle_xy = np.zeros(shape=(1,3)).astype(np.float32)
    rectangle_xy[0,0] = (first_x) if first_x >= 0 else 0
    rectangle_xy[0,1] = (first_y) if first_y >= 0 else 0
    rectangle_xy[0,2] = sz_rect
    if first_x < 0:    
        rectangle_xy[0,2] += first_x
    elif first_y < 0:
        rectangle_xy[0,2] += first_y
        
    return rectangle_xy
    

def CutRectangleAroundMicrowell(\
    circles, sz_rect, corrected_array, ref_frame=0, flag_plot=False):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    """
    Cut a crop centered in the center of each detected microwells.

    Parameters
    ----------
    circles : numpy array.
    sz_rect : int.
    corrected_array : numpy array.
    ref_frame: reference frame.
    flag_plot: 0 or 1.

    Returns
    -------
    crops_info_complete : dictionary.

    """
    if not isinstance(circles, np.ndarray):
        raise TypeError(f"Expected 'circles' to be a numpy array, but got {type(circles)}.")
    if not isinstance(sz_rect, int):
        raise TypeError(f"Expected 'sz_rect' to be a numpy array, but got {type(sz_rect)}.")
    if not isinstance(corrected_array, np.ndarray):
        raise TypeError(f"Expected 'corrected_array' to be a numpy array, but got {type(corrected_array)}.")
    if not isinstance(ref_frame, int):
        raise TypeError(f"ref_frame must be an int, got {type(ref_frame)}")
    if not isinstance(flag_plot, bool):
        raise ValueError("flag_plot must be a boolean value (True or False)")
        
    n_circles = circles.shape[1] # number of microwells
    rectangles_xy=np.zeros((n_circles,2)).astype(np.float32) # info for drawing rectangle, the point saved is the upper left point of the rectangle
    image = corrected_array[:,:,ref_frame].copy() # first frame
    crops_info_complete = {} # dictionary where I save the crops wt information about center and first point of rectangle
    for i in range(n_circles):
        # get coordinates of the starting point (top left)
        rectangle_xy = GetCoordinateRectanglesFromCircleCenter(circles[0][i], sz_rect)
        rectangles_xy[i,0] = rectangle_xy[0,0]
        rectangles_xy[i,1] = rectangle_xy[0,1]
        # cut crop
        single_crop = corrected_array[int(rectangles_xy[i,1]): int(rectangles_xy[i,1])+int(rectangle_xy[0,2]),\
                                         int(rectangles_xy[i,0]): int(rectangles_xy[i,0])+int(rectangle_xy[0,2]), :]
        rect_firstpoint= rectangles_xy[i,:2]
        circle_center = np.array((circles[0][i][0], circles[0][i][1]))
        crops_info_complete[f"well{i}"] = {"circle": circle_center,\
                                           "rect": rect_firstpoint,
                                           "crop": single_crop}
        # for visualization on first frame #slightly different than the values saved in the rectangles_xy array
        start_point =(int(circles[0][i][0]- sz_rect/2), int(circles[0][i][1]- sz_rect/2))
        end_point = (int(circles[0][i][0] + sz_rect/2), int(circles[0][i][1]+ sz_rect/2))
        cv2.rectangle(image, start_point, end_point, (255,0,0), 3) 

    if flag_plot:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.title("Crops for microwells identified")
        plt.show()
    
    return crops_info_complete



def FilterCrops(dict_crops, sz_rect, corrected_array,\
                ref_frame=0, flag_plot=False, mode="flexible"):
    """
    Filter crops.

    Parameters
    ----------
    dict_crops : dictionary.
    sz_rect : int.
    corrected_array : numpy array.
    ref_frame: int.
    flag_plot: bool.

    Returns
    -------
    selected_crops : dictionary.
    """
    import numpy as np
    
    if not isinstance(dict_crops, dict):
        raise TypeError(f"Expected 'dict_crops' to be a numpy array, but got {type(dict_crops)}.")
    if not isinstance(sz_rect, int):
        raise TypeError(f"Expected 'sz_rect' to be a numpy array, but got {type(sz_rect)}.")
    if not isinstance(corrected_array, np.ndarray):
        raise TypeError(f"Expected 'corrected_array' to be a numpy array, but got {type(corrected_array)}.")
    if not isinstance(ref_frame, int):
        raise TypeError(f"ref_frame must be an int, got {type(ref_frame)}")
    if not isinstance(flag_plot, bool):
        raise ValueError("flag_plot must be a boolean value (True or False)")
        
    # mantain only crops who are not cropped
    selected_crops = KeepCompleteCrops(dict_crops, sz_rect, corrected_array, mode)
    # if there are many black pixels, delete the crop (find something stronger:
    # maybe obtain shift with direction and check which circles will be outside based on their center)
    selected_crops = FilterCropsBlackPixels(selected_crops, ref_frame, thresh=0.03)
    # visualize
    if flag_plot:
        VisualizeRectangles(corrected_array, selected_crops, sz_rect, ref_frame)
    return selected_crops

def KeepCompleteCrops(dict_crops, sz_rect, corrected_array, mode="flexible"):
    """
    Keep crops with the desired size.

    Parameters
    ----------
    dict_crops : dict.
    sz_rect : int.
    corrected_array : np array.
    mode: "string". Either "flexible" or "precise".

    Returns
    -------
    selected_crops : dict.
    """
    import numpy as np
    
    if not isinstance(dict_crops, dict):
        raise TypeError(f"Expected 'dict_crops' to be a numpy array, but got {type(dict_crops)}.")
    if not isinstance(sz_rect, int):
        raise TypeError(f"Expected 'sz_rect' to be a numpy array, but got {type(sz_rect)}.")
    if not isinstance(corrected_array, np.ndarray):
        raise TypeError(f"Expected 'corrected_array' to be a numpy array, but got {type(corrected_array)}.")
    if not isinstance(mode, str):
        raise TypeError(f"Expected 'mode' to be a numpy array, but got {type(mode)}.")
    if mode not in ('precise', 'flexible'):
        raise ValueError("mode must be 'flxible' or 'precise'")
        
    if mode=="precise":
        selected_crops = {key:value
                          for key, value in dict_crops.items()
                          if value["crop"].shape == (sz_rect, sz_rect, corrected_array.shape[-1])}
    elif mode=="flexible":
        selected_crops = {
        key: value
        for key, value in dict_crops.items()
        if (
            value["crop"].shape[0] >= sz_rect * 0.95 and # before:0.8
            value["crop"].shape[1] >= sz_rect * 0.95
        )
        }
    return selected_crops

def FilterCropsBlackPixels(dict_crops, ref_frame, thresh=0.03):
    """
    Filter crops based on the proportion of black pixels in a given reference frame.

    Parameters
    ----------
    dict_crops : dict
        A dictionary where each key maps to a dictionary containing at least:
            - "crop" : numpy.ndarray
              A 3D array (H, W, C) representing the crop image with multiple channels/frames.
    ref_frame : int
        Index of the frame/channel in the crop array to analyze for black pixels.

    Returns
    -------
    dict
        A dictionary containing only the crops where the fraction of black pixels 
        (pixels equal to 0) in the specified reference frame is less than 0.03 (3%).

    Raises
    ------
    TypeError
        If `dict_crops` is not a dictionary.
    KeyError
        If any crop entry does not contain the key "crop".
    ValueError
        If the crop is not a 3D numpy array or `ref_frame` is out of bounds.
    """
    import numpy as np

    if not isinstance(dict_crops, dict):
        raise TypeError("dict_crops must be a dictionary.")

    selected_crops = {}
    for key, value in dict_crops.items():
        if "crop" not in value:
            raise KeyError(f"Missing 'crop' key in entry: {key}")

        crop = value["crop"]
        if not isinstance(crop, np.ndarray):
            raise ValueError(f"'crop' for key {key} must be a numpy.ndarray, got {type(crop)}.")

        if crop.ndim != 3:
            raise ValueError(f"'crop' for key {key} must be a 3D numpy array (H, W, C), got shape {crop.shape}.")

        black_fraction = np.sum(crop[:, :, ref_frame] == 0) / (crop.shape[0] * crop.shape[1])
        if black_fraction < thresh:
            selected_crops[key] = value

    return selected_crops


def VisualizeRectangles(corrected_array, dict_final_crops, sz_rect, ref_frame=0):
    """
    Visualize rectangles used to cut crop around microwells.

    Parameters
    ----------
    corrected_array : np array.
    dict_final_crops : dict.
    sz_rect : int.
    ref_frame : int. The default is 0.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    
    if not isinstance(corrected_array, np.ndarray):
        raise TypeError(f"Expected 'corrected_array' to be a numpy array, but got {type(corrected_array)}.")
    if not isinstance(dict_final_crops, dict):
        raise TypeError(f"Expected 'dict_final_crops' to be a numpy array, but got {type(dict_final_crops)}.")
    if not isinstance(sz_rect, int):
        raise TypeError(f"Expected 'sz_rect' to be a numpy array, but got {type(sz_rect)}.")
    if not isinstance(ref_frame, int):
        raise TypeError(f"ref_frame must be an int, got {type(ref_frame)}")
        
    img_new = corrected_array[:,:,ref_frame].copy()
    for i in range(len(dict_final_crops)):
        well_key = list(dict_final_crops.keys())[i]
        rects=dict_final_crops[well_key]["rect"]
        cv2.rectangle(img_new, tuple(rects.astype(np.uint16)), tuple((rects+ sz_rect).astype(np.uint16)), (255,0,0), 3) 
    plt.imshow(img_new, cmap="gray", vmin=0, vmax=255)
    plt.title("Filtered microwells")
    plt.show()

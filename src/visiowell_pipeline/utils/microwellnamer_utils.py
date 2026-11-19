# -*- coding: utf-8 -*-
"""
MicrowellNamer class utils.

@author: Tiziana Tocci
"""

"""
### Function to find microwells
"""
def ExtractCenters(dict_final_crops):
    """
    Extracts the 'circle' values from each entry in the input dictionary.

    The function assumes that each value in the dictionary is itself a dictionary
    containing a key named 'circle'. It returns a NumPy array containing all
    the 'circle' values.

    Parameters:
        dict_final_crops (dict): A dictionary where each value is a dict containing
                                 a 'circle' key associated with a sequence (e.g., list or tuple).

    Returns:
        np.ndarray: A NumPy array of shape (N, ...) where each row corresponds to a 'circle' entry.

    Raises:
        TypeError: If input is not a dictionary.
        ValueError: If any value does not contain a 'circle' key.
        
    """
    
    import numpy as np

    if not isinstance(dict_final_crops, dict):
        raise TypeError("Input must be a dictionary.")

    for key, value in dict_final_crops.items():
        if not isinstance(value, dict):
            raise ValueError(f"Value at key '{key}' must be a dictionary.")
        if "circle" not in value:
            raise ValueError(f"Missing 'circle' key in entry at key '{key}'.")

    final_centers = np.array([value["circle"] for key, value in dict_final_crops.items()])

    return final_centers




def OrderPointsForNaming(corrected_array, final_centers,\
    angle_tolerance, distance_threshold,\
    user_threshold_angle=45, ref_frame=0, flag_plot=False):
    
    """
    Pipeline to order circles for proper naming convention.
    
    Parameters
    ----------
    corrected_array : np ndarray.
    final_centers : np ndarray.
    angle_tolerance : int.
    distance_threshold : int.
    user_threshold_angle : int. The default is 45.
    ref_frame: int. The default is 0
    flag_plot: True or False. The default is False.

    Returns
    -------
    final_sorted_data : dict.

    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    if not isinstance(corrected_array, np.ndarray):
       raise TypeError(f"corrected_array must be a numpy ndarray, got {type(corrected_array)}")
    if not isinstance(final_centers, np.ndarray):
        raise TypeError(f"final_centers must be a numpy ndarray, got {type(final_centers)}")
    if not isinstance(angle_tolerance, int):
        raise TypeError(f"angle_tolerance must be an int, got {type(angle_tolerance)}")
    if not isinstance(distance_threshold, int):
        raise TypeError(f"distance_threshold must be an int, got {type(distance_threshold)}")
    if not isinstance(user_threshold_angle, int):
        raise TypeError(f"user_threshold_angle must be an int, got {type(user_threshold_angle)}")
    if not isinstance(ref_frame, int):
        raise TypeError(f"ref_frame must be an int, got {type(ref_frame)}")
    if not isinstance(flag_plot, bool):
        raise ValueError(f"flag_plot must be True or False, got {type(flag_plot)}.")
        
    if final_centers.shape[0]==2: # if there are only two centers
        point_groups = {0: [final_centers[0,:]], 1: [final_centers[1,:]]}
    else: # if there are more than two centers
        # Step 1: find all straight lines passing through all centers if angle < user_threshold
        valid_lines = FindLinesBetweenCenters(\
        corrected_array, final_centers, threshold_angle=user_threshold_angle,\
        ref_frame=ref_frame, flag_plot=flag_plot)
        # Step 2: group lines based on angle (with tolerance) 
        angle_groups = GroupLinesOnAngle(valid_lines, angle_tolerance) # dictionary of list of dictionary
        # Step 3: sort angle groups by the number of lines (descending) and angle (ascending)
        # except if the final centers are 3 (sort by angle descending)
        if final_centers.shape[0]==3 or final_centers.shape[0]==4:
            sorted_angle_groups = sorted(angle_groups.items(), key=lambda x: (-len(x[1]), -abs(x[0])))
        else: 
            sorted_angle_groups = sorted(angle_groups.items(), key=lambda x: (-len(x[1]), abs(x[0])))
        # Step 4: select the group with the maximum number of lines and smallest angle
        best_angle_group = sorted_angle_groups[0][1]  #first in sorted list
        # Step 5: group based on common points
        # Each group contains the lines that share at least one point
        overlapping_group = GroupLinesOnCommonPoints(best_angle_group)
        # Step 6: filter based on length: keep the line with the maximum length in each group 
        filtered_lines = FilterGroupedLinesByLength(overlapping_group, best_angle_group)
        # Step 7: Grouping points based on distance threshold from the found lines
        merged_lines = MergeCollinearLinesByDistance(filtered_lines) # to check
        point_groups = GroupPointsOnLines(merged_lines, final_centers, distance_threshold)
    if flag_plot:
    # Draw points in lines        
        img=corrected_array[:,:,ref_frame].copy()
        for i in range(len(point_groups)):
            this_group = point_groups[i]
            for j in range(len(this_group)):
                img = cv2.circle(img,\
                                 (int(this_group[j][0]), int(this_group[j][1])), 10, (0,0,255),3)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("Points on the lines who were found")
        plt.show()
    #  Step 8: sort elements belonging to different lines (mean y increasing)
    # and inside of the same line sort on increasing x (line starts at the left side).
    # Sort each element of the dictionary based on increasing x and
    # then sort each line based on increasing y
    sorted_point_groups = SortByXCoordinate(point_groups)
    final_sorted_data = SortByMeanYCoordinate(sorted_point_groups)
    return final_sorted_data




def FindLinesBetweenCenters(corrected_array, points,\
    threshold_angle=45, ref_frame=0, flag_plot=False):
    """
    Find the lines where the centers are
    (find all straight lines passing through all centers if angle < threshold_angle
     and filter based on sign)

    Parameters
    ----------
    corrected_array : numpy ndarray.
    points : numpy ndarray.
    threshold_angle : int. The default is 45.
    ref_frame : int. The default is 0.
    flag_plot : bool. The default is 0.

    Returns
    -------
    valid_lines : list.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    if not isinstance(corrected_array, np.ndarray):
       raise TypeError(f"corrected_array must be a numpy ndarray, got {type(corrected_array)}")
    if not isinstance(points, np.ndarray):
        raise TypeError(f"points must be a numpy ndarray, got {type(points)}")
    if not isinstance(threshold_angle, int):
        raise TypeError(f"threshold_angle must be an int, got {type(threshold_angle)}")
    if not isinstance(ref_frame, int):
        raise TypeError(f"ref_frame must be an int, got {type(ref_frame)}")
    if not isinstance(flag_plot, bool):
        raise TypeError(f"flag_plot must be a boolean, got {type(flag_plot)}")
           
    img = corrected_array[:,:,ref_frame].copy()
    valid_lines = []
    valid_signs = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            angle, length, sign = ComputeAngleLengthLine(points[i,:], points[j,:])
            #plt.plot((points[i,0], points[j,0]), (points[i,1], points[j,1]), color="white", linewidth=3)         
            cond = (-threshold_angle < angle < threshold_angle) #or (-threshold_angle < 180 - angle < threshold_angle)
            
            #if angle < 0:
            #    angle = 180 - angle
            if cond:
                if flag_plot:
                    plt.plot((points[i,0], points[j,0]), (points[i,1], points[j,1]), color="white", linewidth=3) 
                #print(angle)
                valid_lines.append((points[i,:], points[j,:],angle, length))
                valid_signs.append(sign)
    # filter based on angle sign
    most_common_element = max(set(valid_signs), key=valid_signs.count)
    valid_lines_filtered = [x for x, y in zip(valid_lines, valid_signs)\
                            if y == most_common_element]
    if flag_plot:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(angle)
        plt.show()
    return valid_lines_filtered


def ComputeAngleLengthLine(p1, p2):
    """
    Find angle of straight line between two points.
    If the angle > 90, it is brought back between 0 and 90.
    The sign is computed too.

    Parameters
    ----------
    p1 : First point
    p2 : Second point

    Raises
    ------
    TypeError
        if p1 and p2 are not np ndarray.

    Returns
    -------
    angle : angle in degrees.
    length : length of straight line between two points.
    sign: the positive or negative sign of the straight line between the two points.

    """
    import numpy as np
    import math
    
    if not isinstance(p1, np.ndarray):
        raise TypeError(f"points must be a numpy ndarray, got {type(p1)}")
    if not isinstance(p2, np.ndarray):
        raise TypeError(f"points must be a numpy ndarray, got {type(p2)}")
    delta_x = (p1[0].astype(np.float32) - p2[0].astype(np.float32))
    delta_y = (p1[1].astype(np.float32) -p2[1].astype(np.float32))
    angle = math.degrees(math.atan2(delta_y, delta_x))
  
    sign=0
    if (angle > 0 and angle < 90 ) or (angle >-180 and angle < -90):
        sign = 1
    elif angle in (0,90,-180,-90):
        sign=0
    else:
        sign=-1
        
    angle = abs(angle)

    if angle > 90: 
        angle = 180 - angle  

    length =  math.sqrt(delta_x**2 + delta_y**2)

    return angle,length, sign


def GroupLinesOnAngle(valid_lines, angle_tolerance):
    """
    Group lines with a tolerance based on the angle.

    Parameters
    ----------
    valid_lines : list.
    angle_tolerance : int.

    Returns
    -------
    angle_groups : dict.

    """
    if not isinstance(valid_lines, list):
        raise TypeError(f"valid_lines expected to be a list but got {type(valid_lines)}")
    if not isinstance(angle_tolerance, int):
        raise TypeError(f"angle_tolerance expected to be an integer but got {type(angle_tolerance)}")
    
    angle_groups = {}
    for p1, p2, angle, length in valid_lines:
        grouped = False
        for group_angle, group_lines in angle_groups.items():
            if abs(angle - group_angle) <= angle_tolerance:
                group_lines.append({'p1':p1, 'p2':p2, 'angle':angle, 'length':length})
                grouped = True
                break
        if not grouped:
            angle_groups[angle] = [{'p1':p1, 'p2':p2, 'angle':angle, 'length':length}]
    return angle_groups



def GroupLinesOnCommonPoints(best_angle_group):
    """
    Group lines based on common points.

    Parameters
    ----------
    best_angle_group : dict.

    Returns
    -------
    overlapping_group : dict.

    """
    import numpy as np
    overlapping_group = {}
    i=0
    # for each line with the best selected angle
    for element in best_angle_group:
        p1=element["p1"]
        p2=element["p2"]
        length=element["length"]
        angle=element["angle"]
        #breakpoint()
        grouped = False
        # for each element in the dictionary overlapping_group (dictionary of list of dictionary)
        for group_key, group_lines in overlapping_group.items():
            for line in group_lines: # for each dictionary in the list
            # if one of the points forming the best line is equal to one of the point
            # of the line already added to the dictionary
                if any([np.all(p1==line["p1"]),\
                          np.all(p2==line["p1"]),\
                          np.all(p2==line["p2"]),\
                          np.all(p1==line["p2"])]):
                    group_lines.append({'p1':p1, 'p2':p2, 'angle':angle, 'length':length})
                    grouped = True
                    break
        if not grouped:
            overlapping_group[i] = [{'p1':p1, 'p2':p2, 'angle':angle, 'length':length}]
            i = i +1
    return overlapping_group



def FilterGroupedLinesByLength(overlapping_group, best_angle_group):
    """
    # Filter lines in the groups based on length.


    Parameters
    ----------
    overlapping_group : dict.
    best_angle_group : dict.

    Returns
    -------
    filtered_lines : list.

    """
    # all lengths for each element of the dictionary 
    all_lengths = []
    for key,value in overlapping_group.items():
        single_length=[]
        for i in range(len(value)):
            single_length.append(value[i]["length"])
        all_lengths.append(single_length)
        
    # for each group compute maximum length
    max_length = [max(element) for element in all_lengths]
    # delete replicates
    max_length = list(dict.fromkeys(max_length))
    
    # filter the lines by keeping those with maximum length
    filtered_lines = []
    for element in best_angle_group:
        for single_length in max_length:
            if element["length"]==single_length:
                filtered_lines.append(element)
    return filtered_lines



# Function to calculate distance from a point to a line
def DistancePointLine(point, line):
    """
    Calculate the perpendicular distance from a point to a line segment.

    Parameters
    ----------
    point : TYPE
        DESCRIPTION.
    line : TYPE
        DESCRIPTION.

    Returns
    -------
    distance : TYPE
        DESCRIPTION.

    """

    import numpy as np
    #breakpoint()
    p1, p2 = line
    # Vector from p1 to p2
    line_vec = p2.astype(np.float64) - p1.astype(np.float64)
    # Vector from p1 to point
    point_vec = point.astype(np.float64) - p1.astype(np.float64)
    # Projection of point_vec onto line_vec
    t = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
    # Clamp t to be within [0, 1] to ensure the closest point is on the line segment
    t = np.clip(t, 0, 1)
    # Closest point on the line segment to the point
    closest_point = p1 + t * line_vec
    # Distance from the point to the closest point on the line
    distance = np.linalg.norm(point - closest_point)
    return distance


def GroupPointsOnLines(filtered_lines, points, distance_threshold):
    """
    Grouping points based on distance threshold from the found lines


    Parameters
    ----------
    filtered_lines : list.
    points : TYPE
        DESCRIPTION.
    distance_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    point_groups : TYPE
        DESCRIPTION.

    """
    import numpy as np
    
    point_groups = {i: [] for i in range(len(filtered_lines))}
    points_in_lines = []
    for idx, line in enumerate(filtered_lines):
        for point in points:
            distance = DistancePointLine(point.astype(np.float32), [line['p1'], line['p2']])
            if distance < distance_threshold:
                point_groups[idx].append(point)
                points_in_lines.append(point)
    # add points who belong to no line
    points_in_lines = np.array(points_in_lines)
    arr1_structured = np.core.records.fromarrays(points.T)
    arr2_structured = np.core.records.fromarrays(points_in_lines.T)
    mask = ~np.in1d(arr1_structured, arr2_structured)
    result = points[mask]
    # add to point_groups
    for i in range(result.shape[0]):
        #print(len(point_groups))
        point_groups[len(point_groups)] = [result[i,:]]
    return point_groups

def GroupPointsOnLinesDeBug(filtered_lines, points, distance_threshold):
    """
    -> ATTENTION: THIS WAS SUPPOSED TO BE AN OPTTION FOR GroupPointsOnLines, but merging the lines with
    MergeCollinearLinesByDistance solves the issue of points assigned to multiple lines" <-

    Groups points based on their distance to the detected lines.
    Each point is assigned to its closest line (if within the threshold).

    Parameters
    ----------
    filtered_lines : list
        List of line dictionaries containing 'p1' and 'p2'.
    points : numpy.ndarray
        Array of 2D point coordinates.
    distance_threshold : float
        Maximum allowed distance from a line to assign a point.

    Returns
    -------
    point_groups : dict
        Dictionary mapping group indices to lists of points.
    """
    import numpy as np

    point_groups = {i: [] for i in range(len(filtered_lines))}
    points_in_lines = []

    # FIX: For each point, find only the SINGLE closest line within threshold
    for point in points:
        pt_32 = point.astype(np.float32)
        dists = [DistancePointLine(pt_32, [l['p1'], l['p2']]) for l in filtered_lines]
        min_idx = np.argmin(dists)

        # Assign the point ONLY to its nearest line
        if dists[min_idx] < distance_threshold:
            point_groups[min_idx].append(point)
            points_in_lines.append(point)

    # --- Original logic below stays identical ---
    points_in_lines = np.array(points_in_lines)

    # Guard clause in case no points were matched
    if len(points_in_lines) == 0:
        return {i: [p] for i, p in enumerate(points)}

    arr1_structured = np.core.records.fromarrays(points.T)
    arr2_structured = np.core.records.fromarrays(points_in_lines.T)
    mask = ~np.in1d(arr1_structured, arr2_structured)
    result = points[mask]

    # Add points that belong to no line
    for i in range(result.shape[0]):
        point_groups[len(point_groups)] = [result[i, :]]

    return point_groups

import numpy as np


def MergeCollinearLinesByDistance(lines, dist_thresh=40.0):
    """Merges collinear or overlapping line segments based on their perpendicular distance.

    Parameters
    ----------
    lines : list of dict
        List of line dictionaries containing 'p1', 'p2', 'angle', etc.
    dist_thresh : float
        Maximum distance threshold to consider two line segments as belonging to the same line.

    Returns
    -------
    list of dict
        List of merged line dictionaries with updated 'p1', 'p2', 'length', and 'angle'.
    """
    if not lines:
        return []

    merged = []
    used = [False] * len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue

        l1 = lines[i]
        pts = [l1["p1"], l1["p2"]]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            l2 = lines[j]

            # Calculate perpendicular distance of l2's endpoints to the line formed by l1
            d1 = DistancePointLine(
                l2["p1"].astype(np.float32), [l1["p1"], l1["p2"]]
            )
            d2 = DistancePointLine(
                l2["p2"].astype(np.float32), [l1["p1"], l1["p2"]]
            )

            # If both endpoints are within the threshold, merge the line segments
            if d1 < dist_thresh and d2 < dist_thresh:
                pts.extend([l2["p1"], l2["p2"]])
                used[j] = True

        # Find the furthest pair of points to reconstruct the unified bounding line segment
        max_d = -1.0
        best_pair = (pts[0], pts[1])

        for pA in pts:
            for pB in pts:
                d = np.linalg.norm(pA - pB)
                if d > max_d:
                    max_d = d
                    best_pair = (pA, pB)

        merged.append({
            "p1": best_pair[0],
            "p2": best_pair[1],
            "length": max_d,
            "angle": l1["angle"],
        })

    return merged

def SortByXCoordinate(data):  
    """
    Sort microwells inside the line based on their x coordinate.

    Parameters
    ---------
    data: a dictionary of lists of np array.

    Returns
    -------
    sorted_data: a dictionary of lists of np array.
    """
    if not isinstance(data, dict):
        raise TypeError(f"'data' expected to be a dictionary but got {type(data)}")
    sorted_data = {}
    for key, value in data.items():
        sorted_data[key] = sorted(value, key=lambda arr: arr[0])
    return sorted_data


def SortByMeanYCoordinate(data):
    """
    Sort microwells inside the line based on their y coordinate.

    Parameters
    ---------
    data: a dictionary of lists of np array.

    Returns
    -------
    sorted_data: a dictionary of lists of np array.
    """
    import numpy as np
    if not isinstance(data, dict):
        raise TypeError(f"'data' expected to be a dictionary but got {type(data)}")
    #sorted_data = {}
    # Calculate mean of y-coordinates for each list
    mean_y = {key: np.mean([arr[1] for arr in value]) for key, value in data.items()}
    # Sort dictionary items by the mean y-coordinate
    sorted_keys = sorted(mean_y, key=mean_y.get)
    sorted_data = {key: data[key] for key in sorted_keys}
    return sorted_data





"""
### Functions for naming the wells.
"""


def GenerateNames(final_sorted_data):
    """
    Generate alphabetical labels for elements in the sorted data dictionary.

    Parameters
    ----------
    final_sorted_data : dict
        Dictionary with keys representing groups and values as lists of elements.

    Returns
    -------
    centers_and_names : dict
        Dictionary mapping generated names (e.g. 'a1', 'a2', ...) to corresponding elements.
    """
    
    import string
    alphabet = string.ascii_lowercase
    # cut with name
    i=0
    centers_and_names = {}
    for key, value in final_sorted_data.items():
        #print(key)
        letter = alphabet[i]
        for j in range(1,len(value)+1):
            #print(letter + str(j))
            centers_and_names[letter + str(j)] = value[j-1]
        i += 1
    return centers_and_names


def ChangeKeys(dict_final_crops, centers_and_names):
    """
    Replace old keys with new keys.

    Parameters
    ----------
    dict_final_crops : dict.
    centers_and_names : dict.

    Returns
    -------
    merged_info : dict.

    """
    import numpy as np
    merged_info = dict_final_crops.copy()
    #i=0
    for key, value in dict_final_crops.items():
        for key_c,value_c in centers_and_names.items():
            if np.all(value["circle"]==value_c):
                merged_info[key_c] = merged_info[key]
                del merged_info[key]

    return merged_info


def AssignNames(final_sorted_data, dict_final_crops):
    """
    Assign name to microwells based on letters and numbers.

    Parameters
    ----------
    final_sorted_data : dict.
    dict_final_crops : dict.

    Raises
    ------
    TypeError
        if arguments are not in the expected format.
    ValueError
        if there is a problem in number of microwells among generated names and existing microwells
        (=if the two numbers do not coincide).

    Returns
    -------
    merged_info : dict.

    """
    if not isinstance(final_sorted_data, dict):
        raise TypeError(f"Expected 'final_sorted_data' to be a dictionary but got {type(final_sorted_data)}")
    if not isinstance(dict_final_crops, dict):
        raise TypeError(f"Expected 'dict_final_crops' to be a dictionary but got {type(dict_final_crops)}")
    
    # generate the names according to the lines 
    centers_and_names=GenerateNames(final_sorted_data)
    # change the keys of the dictionary
    if len(centers_and_names) != len (dict_final_crops):
        raise ValueError("'centers_and_names' and 'dict_final_crops' do not have the same number of microwells.")
    merged_info = ChangeKeys(dict_final_crops, centers_and_names)
    
    return merged_info


def VisualizeNaming(corrected_array, merged_info, ref_frame=0, flag_plot=False):
    """
    Visualize naming in a figure.

    Parameters
    ----------
    corrected_array : np array.
    merged_info : dict.
    ref_frame : int, optional
        DESCRIPTION. The default is 0.
    flag_plot : bool, optional
        DESCRIPTION. The default is 0.

    Raises
    ------
    TypeError 
        if arguments are not of the expected type.

    Returns
    -------
    fig : matplotlib.pyplot figure.

    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    
    if not isinstance(corrected_array, np.ndarray):
        raise TypeError(f"corrected_array must be a numpy array, but got {type(corrected_array)}")    
    if not isinstance(merged_info, dict):
        raise TypeError(f"merged_info must be a dictionary, but got {type(merged_info)}")
    if not isinstance(ref_frame, int):
        raise TypeError(f"ref_frame must be an integer, but got {type(ref_frame)}")    
    if not isinstance(flag_plot, bool):
        raise TypeError(f"flag_plot must be a boolean, but got {type(flag_plot)}")    
    
    img=corrected_array[:,:,ref_frame].copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for key, value in merged_info.items():
        #breakpoint()
        img = cv2.circle(img,\
                         (int(value["circle"][0]), int(value["circle"][1])), 3, (0,0,255),3)
        ax.text(int(value["circle"][0]), int(value["circle"][1]), key,\
                color="white")
    ax.axis("off")
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    if flag_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig
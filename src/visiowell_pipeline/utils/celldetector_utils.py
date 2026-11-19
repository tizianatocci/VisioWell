# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 10:48:38 2025

@author: tizia
"""
import numpy as np
from ..routines.functions_general import transpose_array_img

### Prepare data
def resize_video(video, new_size):
    import cv2
    n_frames = video.shape[-1]
    resized_video = np.zeros((new_size[0], new_size[1], n_frames), dtype=video.dtype)
    for i in range(n_frames):
        resized_video[:,:,i] = cv2.resize(video[:,:,i], new_size, interpolation=cv2.INTER_LINEAR)
    return resized_video

def from_video_nparray_to_list(images):
    images_list = [images[:, :, i] for i in range(images.shape[-1])]
    return images_list

def PrepareDataTF(video):
    # convert video to 4d as needed
    images_test = np.array([np.stack([video[:,:,i], video[:,:,i], video[:,:,i]], axis=2) 
            for i in range(video.shape[2])])
    images_test=transpose_array_img(images_test)
    #display_video(images_test, user_width=600)
    # find number of frames
    Nframe=images_test.shape[-1]
    return Nframe, images_test

def PrepareDataCP3(video, dim_reshape):
    Nframe=video.shape[-1]
    # resize
    imgs = resize_video(video, dim_reshape)
    imgs= from_video_nparray_to_list(imgs)
    return Nframe, imgs
    
    
### Fist frame evaluation

def EvaluateGoodWell(number_cells):
    """
    Evaluate if the well is good to be analysed (one cell in first frame.)

    Parameters
    ----------
    number_cells : int. Number of cells.

    Returns
    -------
    flag_goodwell : bool. Indicates if the well is good to continue with the analysis or not.

    """
    #if not isinstance(number_cells, int):
    #    raise TypeError(f"'number_cells' must be an int but got {type(number_cells)}")
    flag_goodwell = False
    print(f"{number_cells} cells were identified.")
    if number_cells == 1:
        flag_goodwell = True
        print("The well is good to be analysed.")
    else:
        print("The well is not good to be analysed.")
    return flag_goodwell


def EvaluateGoodWellTF(model, images_test,\
                       sz_input, thresh, flag_adaptive, min_r, max_r, dist_centers,\
                       **kwargs):
    """
    Evaluate if well is good to be analysed - TF.

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    first_frame : TYPE
        DESCRIPTION.
    sz_input : TYPE
        DESCRIPTION.
    thresh : TYPE
        DESCRIPTION.
    flag_adaptive : TYPE
        DESCRIPTION.
    min_r : TYPE
        DESCRIPTION.
    max_r : TYPE
        DESCRIPTION.
    dist_centers : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    flag_goodwell : TYPE
        DESCRIPTION.

    """
    first_frame = images_test[:,:,:,0]
    first_frame = np.expand_dims(first_frame, axis=-1)
    output = DetectTestYolo8TF(1, model, first_frame,\
           sz_input, thresh, flag_adaptive,\
           min_r, max_r, dist_centers, **kwargs)
    n_bb = output["num_cell_test"]
    flag_goodwell = EvaluateGoodWell(n_bb[0])
    return flag_goodwell
    
def EvaluateGoodWellCP3(model, imgs, diam):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    first_frame : TYPE
        DESCRIPTION.
    diam : TYPE
        DESCRIPTION.

    Returns
    -------
    flag_goodwell : TYPE
        DESCRIPTION.

    """
    first_frame = imgs[0]
    output = SegmentCellPose3(model, first_frame, diam)
    numfirstframe  = output["num_cell_test2"]
    flag_goodwell = EvaluateGoodWell(numfirstframe[0])
    return flag_goodwell

def EvaluateGoodWellCP4(model, imgs):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    first_frame : TYPE
        DESCRIPTION.


    Returns
    -------
    flag_goodwell : TYPE
        DESCRIPTION.

    """
    first_frame = imgs[0]
    output = SegmentCellPose4(model, first_frame, batch_size=None)
    numfirstframe  = output["num_cell_test2"]
    flag_goodwell = EvaluateGoodWell(numfirstframe[0])
    return flag_goodwell

### Full evaluation

def InitializeOutputs(Nframe, sz_input):
    """
    Function to initialize outputs.

    Parameters
    ----------
    Nframe : number of frames.
    sz_input : size of input (width, height).

    Returns
    -------
    output_yolo : contains the raw predictions of yolo, before any correction.
    num_cell_test : num cell test is the number of bounding boxes (after final correction).
    num_cell_test2 : num_cell_test2 is the number of cell segmented (after final correction).
    matrix_centers : contains the centers of all the identified cells.
    masks_colors : cellpose-like mask.
    bb_all: boonding boxes after filtering.
    scores_all: scores after filtering.

    """
    # output_yolo contains the raw predictions of yolo, before any correction
    output_yolo = []
    # num cell test is the number of bounding boxes (after final correction)
    # num_cell_test2 is the number of cell segmented (after final correction)
    num_cell_test = np.zeros((Nframe,)).astype(np.float32)
    num_cell_test2 = np.zeros((Nframe,)).astype(np.float32)
    # matrix contains the centers of all the identified cells
    matrix_centers = []
    # mask is a bw mask, mask_colors is a cellpose-like mask
    masks_colors =  np.zeros((sz_input, sz_input, Nframe)).astype(np.uint16)
    # mask is a bw mask, mask_colors is a cellpose-like mask
    masks = np.zeros((sz_input, sz_input, Nframe)).astype(np.uint8)
    masks_colors =  np.zeros((sz_input, sz_input, Nframe)).astype(np.uint16)
    return output_yolo, num_cell_test, num_cell_test2, matrix_centers, masks, masks_colors

    
def Im2Double(img_array):
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
    if img_array.dtype==np.uint8:
        img_array = (img_array).astype(np.float32)/(2**8-1)
    elif img_array.dtype==np.uint16:
        img_array= (img_array).astype(np.float32)/(2**16-1)
    else:
        raise("The image is neither a xp.uint8 nor a xp.uint16.")
        
    return img_array        


def PrepareImgForDetection(img, sz_input):
    """
    Function to prepare each image for detection.
    Steps: 
    - Conversions into the range [0,1];
    - Resizing to the size used for training the network
    - Add a dimension of 1 at the beginning of the img shape
    - Converting to tensor

    Parameters
    ----------
    img : numpy array.
    sz_input : int.

    Returns
    -------
    img : tensor.

    """
    import cv2
    
    if not (isinstance(img, np.ndarray) and len(img.shape) == 3):
        raise ValueError("`img` must be np array with two elements.")
    img = Im2Double(img) # convert to range [0,1]
    img = cv2.resize(img, (sz_input, sz_input)) # resize to the size used to train the network
    img = np.reshape(img, (1,sz_input,sz_input,3)) # add dimension at the beginning in order to make yolo work as intended
    return img


def YoloPredict(yolo, img):
    """
    Use yolo model to predict the bounding boxes in the image.

    Parameters
    ----------
    yolo : model.
    img : tensor.

    Returns
    -------
    y_pred : dictionary.
    boxes : ragged tensor.
    scores : ragged tensor.
    labels : ragged tensor.

    """
    from keras_cv import bounding_box
    if not (isinstance(img, np.ndarray) and len(img.shape) == 4):
        raise ValueError("`img` must be np array with four elements: (1, x,y, 3).")

    #y_pred = yolo.predict(img, verbose=0)
    outputs = yolo(img, training=False)
    decoded = yolo.decode_predictions(outputs, img)
    y_pred = {k: v.numpy() for k, v in decoded.items()}
    y_pred = bounding_box.to_ragged(y_pred)
    boxes = y_pred["boxes"]
    scores = y_pred["confidence"]
    labels = y_pred["classes"]
    return y_pred, boxes, scores, labels



def FromxywhToy1x1y2x2(boxes_tensor):
    """
    Convert from xywh format to y1x1y2x2.

    Parameters
    ----------
    boxes_tensor : tensor of shape (N, 4), where N is the number of boxes.

    Returns
    -------
    boxes_tensor : tensor of shape (N, 4).
    """
    import tensorflow as tf
    if not isinstance(boxes_tensor, tf.Tensor):
        raise TypeError(f"'boxes_tensor' should be a TensorFlow tensor but got {type(boxes_tensor)}")
    if len(boxes_tensor.shape) != 2:
        raise ValueError(f"'boxes_tensor' should be a 2D tensor but got shape {boxes_tensor.shape}")
    if boxes_tensor.shape[1] != 4:
        raise ValueError(f"'boxes_tensor' should have shape (N, 4) but got {boxes_tensor.shape}")

    boxes_tensor = tf.stack([boxes_tensor[:, 1],             # y1 = y
                             boxes_tensor[:, 0],             # x1 = x
                             boxes_tensor[:, 1] + boxes_tensor[:, 3],  # y2 = y + h
                             boxes_tensor[:, 0] + boxes_tensor[:, 2]   # x2 = x + w
                             ], axis=1)
    return boxes_tensor


def Fromy1x1y2x2Toxywh(boxes_tensor):
    """
    Convert from y1x1y2x2 format to xywh.

    Parameters
    ----------
    boxes_tensor : tensor.

    Returns
    -------
    boxes_tensor : tensor.

    """
    import tensorflow as tf
    if not isinstance(boxes_tensor, tf.Tensor):
        raise TypeError(f"'boxes_tensor' should be a TensorFlow tensor but got {type(boxes_tensor)}")
    if len(boxes_tensor.shape) != 2:
        raise ValueError(f"'boxes_tensor' should be a 2D tensor but got shape {boxes_tensor.shape}")
    if boxes_tensor.shape[1] != 4:
        raise ValueError(f"'boxes_tensor' should have shape (N, 4) but got {boxes_tensor.shape}")

    boxes_tensor = tf.stack([
        boxes_tensor[:, 1],               # x = x1
        boxes_tensor[:, 0],               # y = y1
        boxes_tensor[:, 3] - boxes_tensor[:, 1],  # w = x2 - x1
        boxes_tensor[:, 2] - boxes_tensor[:, 0]   # h = y2 - y1
        ], axis=1)
    return boxes_tensor


def FromxminyminxmaxymaxToxywh(boxes_tensor):
    """
    Convert from xminyminxmaxymax format to xywh.

    Parameters
    ----------
    boxes_tensor : tensor.

    Returns
    -------
    boxes_tensor : tensor.

    """
    import tensorflow as tf
    if not isinstance(boxes_tensor, tf.Tensor):
        raise TypeError(f"'boxes_tensor' should be a TensorFlow tensor but got {type(boxes_tensor)}")
    if len(boxes_tensor.shape) != 2:
        raise ValueError(f"'boxes_tensor' should be a 2D tensor but got shape {boxes_tensor.shape}")
    if boxes_tensor.shape[1] != 4:
        raise ValueError(f"'boxes_tensor' should have shape (N, 4) but got {boxes_tensor.shape}")

    boxes_tensor = tf.stack([boxes_tensor[:, 0],             # x = xmin
                             boxes_tensor[:, 1],             # y = ymin
                             boxes_tensor[:, 2] - boxes_tensor[:, 0],  # w = xmax-xmin
                             boxes_tensor[:, 3] - boxes_tensor[:, 1]   # h=ymax-ymin
                             ], axis=1)
    return boxes_tensor


def FilterNMS(boxes, scores, labels, iou_threshold=0.5, max_output_size=100):
    """
    NMS algorithm.

    Parameters
    ----------
    boxes : ragged tensor?.
    scores : ragged tensor?.
    labels : ragged tensor?.
    iou_threshold : float. The default is 0.5.
    max_output_size : int. The default is 100.

    Returns
    -------
    boxes_tensor : tensor.
    scores_tensor : tensor.
    labels_tensor : tensor.

    """
    import tensorflow as tf
    # convert to tensor 
    if not isinstance(boxes, tf.Tensor):
        boxes_tensor= tf.squeeze(boxes.to_tensor(), axis=0)
    else:
        boxes_tensor=boxes
    if not isinstance(scores, tf.Tensor):
        scores_tensor=tf.squeeze(scores.to_tensor(), axis=0)
    else:
        scores_tensor=scores
    if not isinstance(labels, tf.Tensor):
        labels_tensor=tf.squeeze(labels.to_tensor(), axis=0)
    else:
        labels_tensor=labels
        
    boxes_tensor = FromxywhToy1x1y2x2(boxes_tensor) # convert to wanted format
    # NMS
    #breakpoint()
    idx_nms = tf.image.non_max_suppression(
        boxes_tensor, scores_tensor,max_output_size, iou_threshold=iou_threshold)
    # select the bounding boxes who pass the filtering 
    boxes_tensor = tf.gather(boxes_tensor, idx_nms)
    scores_tensor = tf.gather(scores_tensor, idx_nms)
    labels_tensor = tf.gather(labels_tensor, idx_nms)
    boxes_tensor= Fromy1x1y2x2Toxywh(boxes_tensor) # reconvert to wanted format
    #if n_initial-len(idx_nms) > 0:
    #    print(f"Deleted {n_initial-len(idx_nms)} boxes with NMS")
    return boxes_tensor, scores_tensor, labels_tensor


def FilterBasedOnThresh(thresh, boxes, scores, labels):
    """
    Apply threshold over score to filter the bounding boxes.

    Parameters
    ----------
    thresh : int.
    boxes : tensor.
    scores : tensor.
    labels : tensor.

    Returns
    -------
    boxes_filter : np array.
    scores_filter : np array.
    labels_filter : np array.

    """
    import tensorflow as tf
    
    if not isinstance(boxes, tf.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("`boxes` must be a regular tensor with 2 dimensions.")
    if not isinstance(scores, tf.Tensor) or scores.ndim != 1:
        raise ValueError("`scores` must be a regular tensor with 1 dimensions.")
    if not isinstance(labels, tf.Tensor) or labels.ndim != 1:
        raise ValueError("`labels` must be a regular tensor with 1 dimensions.")    
    idx = (scores).numpy() > thresh
    boxes_filter = (boxes.numpy())[idx,:]
    scores_filter = (scores.numpy())[idx]
    labels_filter = (labels.numpy())[idx]
    return boxes_filter, scores_filter, labels_filter



def AssertAllBoxesInsideImg(boxes, img_shape):
    """
    Assert if all boxes are entirely inside of the frame.
    x, y refer to the top left point of the bounding box, w is the width and h the height.
    If x < 0 -> x=0.
    If y < 0 -> y=0.
    If x2= x + w > respective size, a smaller w has to be recomputed.
    Similar for y2=y+h > respective size.

    Parameters
    ----------
    boxes : np array of shape (N, 4), where each row represents a bounding box in the
    (x,y,w,h) format where (x,y) represent the first top left point, w represents the width
    and h represents the height.
    img_shape: tuple of at least shape (2,) representing the size of the image (height, width).
  
    
    Returns
    -------
    boxes : np array of shape (N, 4).
    
    Notes
    If `boxes` is empty, the function returns an empty array.
    Boxes with zero width or height will remain unchanged.

    """
    #breakpoint()
    if not isinstance(boxes, np.ndarray) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("`boxes` must be a numpy array of shape (N, 4).")
    if not (isinstance(img_shape, (list, tuple, np.ndarray)) and len(img_shape) >= 2):
        raise ValueError("`img_shape` must be a tuple, liste or np array with at least two elements.")
 
    # verify top and left sides of the image
    boxes=boxes.copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1]) # Adjust x
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0]) # Adjust y
    # verify right and bottom sides of the image
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1] - boxes[:, 0])  # Adjust width
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0] - boxes[:, 1])  # Adjust height
    
    return boxes

def BigSmallBB(boxes, scores, max_size=80, min_size=10):
    """
    Delete bounding boxes too big or too small.

    Parameters
    ----------
    boxes : np ndarray.
    scores : np ndarray.
    max_size : int. The default is 80.
    min_size : int. The default is 10.

    Raises
    ------
    TypeError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    bboxes : np ndarray.
    scores : np ndarray.

    """
    # delete boxes and scores for the boxes who are too big or too small
    if not isinstance(boxes, np.ndarray):
        raise TypeError("boxes must be a np ndarray.")
    if not isinstance(scores, np.ndarray):
        raise TypeError("scores must be a np ndarray.")
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError("boxes must be of shape (N, =4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must be 1D and match boxes length")

    keep_bbox = (boxes[:, 2] > min_size) & (boxes[:, 2] < max_size) & \
            (boxes[:, 3] > min_size) & (boxes[:, 3] < max_size)

    
    bboxes = boxes[keep_bbox]
    scores = scores[keep_bbox]
    
    return bboxes, scores


def visualize_bb(img, boxes_nms, scores_nms):
    """
    Function to visualise all the bounding boxes.

    Parameters
    ----------
    img : np array.
    boxes_nms : np array.
    scores_nms : np array.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1)
    ax.imshow(np.squeeze(img))
    i=0
    for box in boxes_nms:
        y1, x1, y2, x2 = box
        rect = patches.Rectangle(
            (y1, x1), y2, x2, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        if not scores_nms is None:
            ax.text(y1, x1 - 0.02, round(scores_nms[i], 2) , color='w', fontsize=10, ha='center', va='bottom')
        i=i+1
    plt.show()



def my_CHT_stem(B,rmin,rmax, dmin, sens, BoundingBox, thresh_dist=10, first_param=100):
    """
    Find circles with CHT.

    Parameters
    ----------
    B : numpy array. Image
    rmin : int. Minimum radius for finding circles.
    rmax : int. Maximum radius for finding circles.
    dmin : int. Minimum distance between circles.
    sens : float. Sensitivity of algorithm.
    BoundingBox : numpy array. Single bounding box.
    thresh_dist: int. To evaluate if the center is contained in the box or not. Default value is 10.
    first_param: int. CHT first sensitivity parameter.

    Returns
    -------
    CHT : numpy array.
    centers : TYPE
        DESCRIPTION.
    radii : TYPE
        DESCRIPTION.

    """
    import cv2
    if not (isinstance(B, np.ndarray) and len(B.shape) == 2):
        raise ValueError("`B` must be np array with two elements.")
    if not isinstance(BoundingBox, np.ndarray) or BoundingBox.ndim !=  1 or BoundingBox.shape[0] != 4:
        raise ValueError("`BoundingBox` must be a numpy array of shape (4,).")
     
    img_used = (B*255).astype(np.uint8).copy() # convert image to uint8
    circles=None
    inside = 0
    
    while (circles is None or circles.shape[0]==0 or np.all(inside != 1)) & (sens > 1):
        circles = cv2.HoughCircles(img_used,cv2.HOUGH_GRADIENT,\
                                  1, dmin,\
                                  param1=first_param,param2=sens,\
                                  minRadius=rmin,maxRadius=rmax) # decreases param2 -> find more circles if you did not find any
        # verify if any of the circles identified are inside. 
        # if more than one circle is identified, keep the one closest to the center.
        # if at least one circle was identified
        if (circles is not None and circles.shape[0]>0):
            circles = circles[0]
            #keep only the circles with the center inside the bounding box
            # and with reasonable distance
            inside_dist, inside = check_if_centers_inside(BoundingBox, circles, thresh_dist)
            if np.any(inside):
                #print(f"Sensitivity is {sens}")
                centers = circles[inside,:2]
                radii = circles[inside,-1]
                m_dist = inside_dist[inside]
                # keep only the circle closest to the center of the crop
                if sum(inside)>1:
                    centers=centers[np.argmin(m_dist),:]
                    radii=radii[np.argmin(m_dist)]
                centers_uint=np.squeeze((np.around(centers)).astype(np.uint16))
                radii_uint=np.squeeze(np.around(radii)).astype(np.uint16).item()
                # create a black mask with the same dimensions as the image
                CHT = np.zeros_like(img_used, dtype=np.uint8)
                cv2.circle(CHT, centers_uint, radii_uint, 255, thickness=-1)
            else:
                sens=min(sens,sens-1)
        else:
            sens=min(sens, sens-1)
            
    if (circles is None or circles.shape[0]==0 or np.all(inside != 1)):
        CHT = np.zeros_like(img_used, dtype=np.uint8)
        centers=np.array([])
        radii=np.array([])
    else:
        centers=np.squeeze(centers)
        radii=radii.item()
    
    return CHT, centers, radii

def check_if_centers_inside(BoundingBox, circles, thresh_dist = 10):
    """
    Function to check if the centers are contained inside of the Bounding Box or not.

    Parameters
    ----------
    BoundingBox : numpy array of shape (4, ). The values correspond to the x, y, w, h format.
    circles : numpy array of shape (N, 3). Each row stores the x, y coordinates and radius of the circle identified.


    Returns
    -------
    inside_dist: numpy array. distances of the circles from the center of the bounding box.
    inside : numpy array. wheter the circles are inside or outside.

    """
    #keep only the circles with the center inside the bounding box
    # and with reasonable distance
    # Bounding box center
    if not isinstance(BoundingBox, np.ndarray) or BoundingBox.ndim !=  1 or BoundingBox.shape[0] != 4:
        raise ValueError("`BoundingBox` must be a numpy array of shape (4,).")
    if not isinstance(circles, np.ndarray) or circles.ndim !=  2 or circles.shape[1] != 3:
        raise ValueError("`circles` must be a numpy array of shape (N,3).")
    if not isinstance(thresh_dist, int):
        raise TypeError(f"'thresh_dist' must be an int but got {type(thresh_dist)}")   
        
    center_x = BoundingBox[0] + BoundingBox[2] / 2
    center_y = BoundingBox[1] + BoundingBox[3] / 2
    inside_dist = np.sqrt((circles[:,0]-center_x)**2+(circles[:,1]-center_y)**2)
    inside = (circles[:, 0] > BoundingBox[0]) &\
        (circles[:, 0] < BoundingBox[0] + BoundingBox[2]) &\
        (circles[:, 1] > BoundingBox[1]) &\
        (circles[:, 1] < BoundingBox[1] + BoundingBox[3]) & (inside_dist < thresh_dist)
    return inside_dist, inside


def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    """
    Apply the equivalent of the function "imadjust" in MATLAB.

    Parameters
    ----------
    src : input one-layer image (numpy array).
    tol : tolerance, from 0 to 100.
        DESCRIPTION. The default is 1.
    vin : src image bounds.
        DESCRIPTION. The default is [0,255].
    vout : dst image bounds.
        DESCRIPTION. The default is (0,255).

    Returns
    -------
    dst: output image.

    """
    import bisect

    if not isinstance(src, np.ndarray):
        raise TypeError("'src' should be a numpy array but got {type(src)}")
    if len(src.shape) != 2:
        raise ValueError("'src' should be a 2D array but got shape {}".format(src.shape))
    if not isinstance(tol, int) or not (0 <= tol <= 100):
        raise ValueError(f"'tol' should be a numeric value between 0 and 100, but got {tol}")
    if not isinstance(vin, (list, tuple)) or len(vin) != 2 or not all(isinstance(v, int) for v in vin):
        raise ValueError(f"'vin' should be a list or tuple of two numeric values, but got {vin}")
    if not isinstance(vout, (list, tuple)) or len(vout) != 2 or not all(isinstance(v, int) for v in vout):
        raise ValueError(f"'vout' should be a list or tuple of two numeric values, but got {vout}")

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst


def preprocess_image_before_CHT(img, BoundingBox):
    """
    Process img before applying CHT. 

    Parameters
    ----------
    img : np array with three dimensions (height, width, number of frames).
    BoundingBox : np array of shape (4,).

    Returns
    -------
    cropped_large : np array 2D.

    """
    if not (isinstance(img, np.ndarray) and len(img.shape) == 3):
        raise ValueError("`img` must be np array with three elements.")
 
    if not isinstance(BoundingBox, np.ndarray) or BoundingBox.ndim !=  1 or BoundingBox.shape[0] != 4:
        raise ValueError("`BoundingBox` must be a numpy array of shape (4,).")
    

    # cropped_large is an 2D image where in correspondance of the crop I have
    # the real values of the original img and elsewhere I have only zeros
    cropped_large=np.zeros(shape=img.shape[0:2])
    cropped=img[BoundingBox[1]:BoundingBox[1]+BoundingBox[3], \
                BoundingBox[0]:BoundingBox[0]+BoundingBox[2]]
    # imadjust
    assert np.allclose(cropped[:, :, 0], cropped[:, :, 1], atol=1) and \
       np.allclose(cropped[:, :, 1], cropped[:, :, 2], atol=1)

    cropped=imadjust((cropped[:,:,0]*255).astype(np.uint8))
    cropped = cropped/255
    cropped=np.stack((cropped, cropped, cropped), axis=2)
    cropped_large[BoundingBox[1]:BoundingBox[1]+BoundingBox[3],\
        BoundingBox[0]:BoundingBox[0]+BoundingBox[2]]=cropped[:,:,0]
    return cropped_large


def dilate_mask(radii, n_radius_divider, CHT):
    """
    Function to dilate mask.

    Parameters
    ----------
    radii : radius of cell.
    n_radius_divider : number by which the radii will be divided.
    CHT : mask.

    Returns
    -------
    CHT_dilate : dilated mask.

    """
    import cv2
    if not (isinstance(CHT, np.ndarray) and len(CHT.shape) == 2):
        raise ValueError("`CHT` must be np array with two elements.")
        
    disk_sz=(np.round(radii/n_radius_divider)).astype(np.uint8) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*disk_sz+1, 2*disk_sz+1))
    CHT_dilate = cv2.dilate(CHT, kernel)
    return CHT_dilate


def replace_cell_with_random_pixels(img, CHT_dilate, delimit_pixel=10, std_normal=0.05):
    """
    Function to compute random pixels to replace the cell.

    Parameters
    ----------
    img : numpy array.
    CHT_dilate : numpy array.
    
    Returns
    -------
    new_img : numpy array.

    """
    if not (isinstance(img, np.ndarray) and len(img.shape) == 3):
        raise ValueError("`img` must be np array with three elements.")
    if not (isinstance(CHT_dilate, np.ndarray) and len(CHT_dilate.shape) == 2):
        raise ValueError("`CHT_dilate` must be np array with two elements.")
        
    #mean_b=np.mean(img[:delimit_pixel,:delimit_pixel,0])
    mean_b = np.mean(img[:,:,0])
    pixels_mask=(CHT_dilate==255)
    new_img=(img[:,:,0]).copy()
    np.random.seed(123)
    #breakpoint()
    new_img[pixels_mask]=np.random.normal(mean_b, std_normal, np.sum(pixels_mask))
    new_img=(new_img*255).astype(np.uint8)
    return new_img


def IterativeImpainting(img, bboxes, scores, min_r, max_r, dmin,\
    sens=40, thresh_dist=10, n_radius_divider=3, std_normal=0.05):
    """
    Iterative impainting. Cells are found with CHT and masked.

    Parameters
    ----------
    img : numpy array. Original image.
    bboxes : numpy array of shape (N,4). Each row stores a bounding box with coordinates
    x, y, w, h. (x,y) represent the coordinates of the top left point, w the width and h the height.
    scores: numpy array of shapoe (N,). Each row store the yolo score for each bounding box.
    min_r : int. Minimum radius for masking.
    max_r : int. Maximum radius for masking.
    dmin : int. minimum distance between two different identified circles.
    sens: sensitivity for CHT. Default value is 40.
    thresh_dist: int. Distance from the center to assert if the center of the circle
    identified with CHT is inside the bounding box or not. Default value is 10 pixels.
    n_radius_divider: int. radii gets divided by this number to compute the disk size for dilatating the cell mask.
    std_normal. float. Standard deviation for computing random values to use for substituting the cell.

    Returns
    -------
    CHT_large : numpy array. Masks for the frame for the specific iteration.
    In the third dimension there is a mask for each bounding box 
    (if the mask has been found) for the specific iteration.
    Example: img has 8 cells. In iteration it=1 the algorithm finds 3 bounding boxes.
    CHT_large is composed of 3 masks, each one is the mask of a single cell.
    CHT_centers: the centers of the identified cells.
    img : numpy array. Image where the cells have been masked.
    cell_info_full: list with centers, relative bounding box and relative score.

    """
    if not (isinstance(img, np.ndarray) and len(img.shape) == 3):
        raise ValueError("`img` must be np array with three elements.")
    if not isinstance(bboxes, np.ndarray) or bboxes.ndim !=  2 or bboxes.shape[1] != 4:
        raise ValueError("`bboxes` must be a numpy array of shape (4,).")
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise ValueError("`scores` must be a numpy array of shape (N,).")
           
    bboxes_copy=bboxes.copy()
    scores_copy = scores.copy()
    # one frame for each bounding box
    CHT_large= np.zeros(shape=(img.shape[0], img.shape[1], bboxes_copy.shape[0])).astype(np.uint16)
    CHT_centers = [] # info about centers
    cell_info_full = [] # centers, box and score
  
    # for each identified bounding box
    for b in range(bboxes_copy.shape[0]):
        # bounding box must contains integers and not float numbers for CHT algorithm to work
        BoundingBox=np.round(bboxes_copy[b,:]).astype(np.uint16)
        cropped_large=preprocess_image_before_CHT(img, BoundingBox)
        # apply CHT
        CHT, centers, radii = my_CHT_stem(cropped_large,min_r,max_r, dmin, sens, BoundingBox, thresh_dist=thresh_dist)        
        #breakpoint()
        if centers.size!=0:
            CHT_centers.append(centers)
            cell_info_full.append([centers, bboxes_copy[b,:], scores_copy[b]])
            # dilatation
            CHT_dilate = dilate_mask(radii, n_radius_divider, CHT)
            CHT_large[:,:,b]=CHT_dilate
            
            assert np.allclose(img[:, :, 0], img[:, :, 1], atol=1e-6) and \
                np.allclose(img[:, :, 1], img[:, :, 2], atol=1e-6)

        
            #random area
            new_img=replace_cell_with_random_pixels(img, CHT_dilate,\
                                                    delimit_pixel=10, std_normal=std_normal)
            
            new_img=np.stack((new_img,new_img,new_img), axis=2)
            del img
            img=(new_img).astype(np.float32)/255
        else:
            continue
       
    return CHT_large, CHT_centers, img, cell_info_full
    #return CHT, centers, radii
    


def correction_cells_identified_twice(i, CHT_centers, matrix_centers, cell_info_full,\
                                      CHT_large, number_cells, boxes_nms, scores_nms,\
                                      user_dist_centers):
    """
    Function to correct cells identified twice across different iterations of the algorithm
    by removing duplicated based on distance threshold.
    This function is called only if the iteration for the frame > 0.
    It assumes  that `CHT_centers` and `matrix_centers` are not empty.
    
    Parameters
    ----------
    i: int. Frame index.
    CHT_centers : list of numpy arrays. Detected centers for frame "i".
    matrix_centers : list. History of detected centers across frames and iterations.
    cell_info_full: list. Full cell information for frame "i": detected center (CHT_centers), bounding box, score.
    CHT_large : numpy array. Mask of the cell.
    num_cell_test : numpy array. Current number of cells detected in frame "i".
    boxes_nms: numpy array. Bounding box coordinates xywh (N,4).
    scores_nms: numpy array. Confidence scores for the bounding boxes (N,).
    user_dist_centers : int. Distance threshold for duplicate detection.

    Returns
    -------
    CHT_large : numpy array. Updated segmentation mask.
    good_centers : list. Filtered list of centers. It must be the same format as CHT_centers.
    good_cell_info_full: list. Filtered list of full cell information. It must be the same format as cell_info_full.
    num_cell_test : int. Updated number of cells detected.


    """
    #breakpoint()
    from scipy.spatial.distance import cdist 

    if not isinstance(CHT_centers, list) or not all(isinstance(c, np.ndarray) for c in CHT_centers):
        raise ValueError("`CHT_centers` must be a list that contains np arrays.")
    if not isinstance(matrix_centers, list):
        raise ValueError("`matrix_centers` must be a list. Each element  contains: an int for the number of frame, an int for the number of iteration of the algorithm,\
                         a second list. This second list contains: a numpy array with the circle coordinates x,y; a np array for with the xywh boxes coordinates, a number of the respective scores. ")
    if not (isinstance(CHT_large, np.ndarray) and len(CHT_large.shape) == 3):
        raise ValueError("`CHT_large` must be np array with three elements.")
    if not isinstance(boxes_nms, np.ndarray) or boxes_nms.ndim != 2 or boxes_nms.shape[1] != 4:
        raise ValueError("`boxes_nms` must be a numpy array of shape (N, 4).")
    if not isinstance(scores_nms, np.ndarray) or scores_nms.ndim != 1:
        raise ValueError("`scores_nms` must be a numpy array of shape (N,).")
    
    # find centers of past iteration
    past_centers = [
        sublist[0]
        for frame, _, groups in matrix_centers if frame == i
        for sublist in groups
    ]
    
    # compute distances between the centers of the past iteration and the center in this iteration
    past_centers = np.array(past_centers)
    current_centers = np.array(CHT_centers)
    if current_centers.shape[1] != 2 or past_centers.shape[1] != 2:
        raise ValueError("Centers must have 2D coordinates (x, y).")
    distances = cdist(current_centers, past_centers) 
    
    # find duplicates (where distance < user_dist_centers)
    duplicates = np.any(distances < user_dist_centers, axis=1)

    # initialize output
    good_centers = CHT_centers
    good_cell_info_full = cell_info_full
    
    # filter centers and cell info and mask and modify number of cells
    if np.any(duplicates):
        good_centers = [center for idx, center in enumerate(CHT_centers) if not duplicates[idx]]
        good_cell_info_full = [info for idx, info in enumerate(cell_info_full) if not duplicates[idx]]
        CHT_large[:, :, duplicates] = 0
        number_cells -= np.sum(duplicates)
        #print(f'Frame {i}... Number of cells delete: {sum(duplicates)}\n')
        #print(f'Frame {i}... Number of BB corrected: {number_cells}\n')

    return CHT_large, good_centers, good_cell_info_full, number_cells


def treat_masks(CHT_large, masks_frame, it):
    """
    Function that treats masks.
    The first line finds the maximum across the frames of CHT_large. 
    There is one cell per frame.
    I obtain a BW mask or a mask where each cell is identified by a number.
    In the second case, if the cells overlap, then the first cell will be covered by
    the second, third cells and so on (first identified, first covered.)
    This mask is saved in masks_frame.

    Parameters
    ----------
    CHT_large : np array.
    masks_frame : np array.
    it : int.

    Returns
    -------
    final_CHT : np array.
    final_mask : np array.

    """
    final_CHT = np.max(CHT_large, axis=2)
    #plt.imshow(final_CHT)
    masks_frame[:,:,it]=final_CHT
    #final_mask= np.max(masks_frame, axis=2)
    return final_CHT, masks_frame
   

def create_colored_mask(CHT_large, n_mask):
    """
    In each frame there is a cell. For each cell I use a different uint.
    I obtain cellpose masks like.

    Parameters
    ----------
    CHT_large : np array.
    n_mask : int.

    Returns
    -------
    n_mask : int.
    CHT_large_uints : np array.

    """
    CHT_large_uints = CHT_large.copy()
    for i_singlemask in range(CHT_large_uints.shape[-1]):
        single_mask = CHT_large_uints[:,:,i_singlemask] 
        single_mask[single_mask!=0] = n_mask
        CHT_large_uints[:,:,i_singlemask] = single_mask
        if np.any(single_mask!=0):
            n_mask=n_mask+1
    return n_mask, CHT_large_uints


def DetectTestYolo8TF(Nframe, yolo, images_test,\
    sz_input, thresh,\
    flag_adaptive, min_r, max_r,\
    user_dist_centers,\
    start_frame=0,\
    max_n_it=100, iou_thresh=0.5, max_size=80, min_size=10, dmin=10, sens=40, thresh_dist=10,\
    pos_rectangle=(10, 10), sz_patch=75, pos_text=45,\
    std_normal=0.05, flag_plot=0):
    """
    Function to detect cells.

    Parameters
    ----------
    Nframe : int. Number of frames in the video.
    yolo : keras_csv model. Yolo with trained weights loaded.
    sz_input : int. sz_input for the network.
    thresh : float. If the score of a detected cell is less than thresh, than the cell will no be considered as such.
    images_test : numpy array. Video in 4d.
    flag_adaptive : int. If flag_adaptive=1, the adaptive method is adopted.
    min_r : int. Minimum radius for masking cells.
    max_r : int. Maximum radius for masking cells.
    user_dist_centers: int. minimum distance between centers in consecutive iterations (it is used to correct the mistake of picking a cell twice).
        Default value is 30 px for a 640x640 image.
    start_frame: int. Default value is 0.
    max_n_it: int. Number of iterations possible for a well. Default value is 100.
    iou_thresh: float. Value for NMS.
    max_size: int. Parameter to delete bounding boxes who are too big. Default value is 80 pixels.
    dmin: int. Value for CHT. Default value is 10. Distances between centers.
    sens: int. Value for CHT sensitivity. Default value is 40.
    thresh_dist: int. Value for deciding if the cell identified with CHT is inside the bounding box.
    pos_rectangle: tuple of ints. Position for plotting rectangle.
    sz_patch: int. Size for plotting rectangle.
    pos_text: int. Position  for plotting test. 
    std_normal: std deviation to use for computing random pixels to use to substitute the cells in the image.
    flag_plot = 0 or 1. 
    
    Returns
    -------
    A dictionary with the following elements.
    num_cell_test : numpy array. Number of bounding boxes detected for frame.
    num_cell_test2 : numpy array. Number of cells masked for frame.
    masks: numpy array. Black and white masks.
    masks_colors: numpy array. Video of colored masks obtained.
    matrix_centers: list with number of frame, number of iteration, a sublist with center, bounding box and score.
    output_yolo: list.
    
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not isinstance(images_test, np.ndarray) or images_test.ndim !=  4 or images_test.shape[2] != 3:
        raise ValueError("`images_test` must be a numpy array of shape (x,y, 3 , N).")
        
    output_yolo, num_cell_test, num_cell_test2,\
    matrix_centers, masks, masks_colors = InitializeOutputs(Nframe, sz_input)
    
    for i in range(start_frame, Nframe): # for each frame, run the analysis
        if i%50==0:
            print(f"Frame analysed: {i}/{Nframe}")
        img=images_test[:,:,:,i]

        img=PrepareImgForDetection(img, sz_input) # operations to prepare image for detection
        if flag_plot:
            plt.imshow(np.squeeze(img)) # show image
            plt.show()
        # new_img is the one uploaded now and 
        # past_img is initialized as a black image
        new_img=img
        past_img=np.zeros((img).shape).astype(np.float32)
        del img
        it = 0 # "it" is for keeping track of the iteration for the current frame
        n_mask = 1         # n_mask is for creating cellpose masks
        # masks for this iteration
        masks_frame = np.zeros((sz_input, sz_input, max_n_it)).astype(np.uint8) # assumption: maximum number of different iterations in a frame is 100
        masks_frame_c = np.zeros((sz_input, sz_input, max_n_it)).astype(np.uint16) # assumption: maximum number of different iterations in a frame is 100
    
        #while new_img != past_img, iterative YOLO
        # new_img != past_img means that new cells have been detected
        while not np.all(past_img == new_img):
            img=new_img # img is set as the new img, i.e. the last result
            img = np.reshape(img, (1,sz_input,sz_input,3)) # ensure img the right size

            
            y_pred, boxes, scores, labels = YoloPredict(yolo, img) # use the trained network
            output_yolo.append([i, it, y_pred]) # save yolo's output not modified 
            # apply corrections: NMS, threshold, assert that all boxes are inside the image
            boxes_tensor, scores_tensor, labels_tensor = FilterNMS(
                boxes, scores, labels, iou_threshold=iou_thresh, max_output_size=max_n_it)
            boxes_nms, scores_nms, labels_nms = FilterBasedOnThresh(thresh, boxes_tensor, scores_tensor, labels_tensor)
            boxes_nms=AssertAllBoxesInsideImg(
                boxes_nms, img.shape[1:])
            boxes_nms, scores_nms = BigSmallBB(boxes_nms, scores_nms,\
                max_size=max_size, min_size=min_size)
            # number of cells detected if we consider the bounding boxes
            num_cell_test[i]=num_cell_test[i]+boxes_nms.shape[0]
            
            del y_pred, boxes, scores, labels,\
                boxes_tensor, scores_tensor, labels_tensor
            #print(f'Frame {i}... Number of BB: {num_cell_test[i]}\n')

            # if some bounding boxes are identified
            if boxes_nms.shape[0]>0:
                if flag_plot:
                    visualize_bb(img, boxes_nms, scores_nms)
                # segmentation with CHT
                # CHT_large: mask corresponding to a single cell.
                # The third dimension stores a single mask. As many masks as bounding boxes.
                # CHT_centers: center of the segmented cell
                # whole_img: image where the segmented cell is impainted
                #breakpoint()
                CHT_large, CHT_centers, whole_img, cell_info_full = IterativeImpainting(\
                    np.squeeze(img), boxes_nms, scores_nms, min_r, max_r, dmin,\
                    sens, thresh_dist, std_normal=std_normal)
                # matrix_centers.append((i, it, CHT_centers))
                # deal with cells identified twice
                if it>0: 
                    CHT_large, good_centers, good_cell_info_full, num_cell_test_corrected = correction_cells_identified_twice(
                        i, CHT_centers, matrix_centers, cell_info_full, CHT_large,
                        num_cell_test[i], boxes_nms, scores_nms, user_dist_centers)
                    num_cell_test[i] = num_cell_test_corrected
                    if len(good_centers) > 0:
                        matrix_centers.append([i, it, good_cell_info_full])
                else:          
                    #breakpoint()
                    matrix_centers.append([i, it, cell_info_full])
                    
             
                # I obtain the mask with all the cells in the same frame for
                # this iteration of the while loop
                _, masks_frame = treat_masks(CHT_large, masks_frame, it)
                # cellpose mask
                n_mask, CHT_large_uints = create_colored_mask(CHT_large, n_mask)
                _, masks_frame_c = treat_masks(
                    CHT_large_uints, masks_frame_c, it)
                
                it=it+1
                #del final_CHT, final_CHT_c
                # compute the number of segmented cells
                # for each frame of CHT_large, if the frame is not entirely
                # black, then there is a cell
                for z in range(CHT_large.shape[-1]):
                    oneframe=np.any(CHT_large[:,:,z])
                    #breakpoint()
                    if np.any(oneframe):
                        num_cell_test2[i]=num_cell_test2[i]+1
                    else:
                        num_cell_test2[i]=num_cell_test2[i]+0
                #if i%50==0:
                #    print(f'Frame {i}... Number of cells: {num_cell_test2[i]}\n')
                del oneframe, CHT_large, CHT_large_uints
                # show image
                if flag_plot:
                    plt.imshow(whole_img)
                    rect = patches.Rectangle(pos_rectangle, sz_patch, sz_patch, linewidth=2, edgecolor='r', facecolor='w')
                    plt.gca().add_patch(rect)
                    plt.text(pos_text,pos_text, f"{int(num_cell_test[i])}, {int(num_cell_test2[i])}",\
                             ha='center', va='center', fontsize=14)
                    plt.show()
            else: # if no bounding box is identified
                whole_img=img
                #final_mask=np.zeros((sz_input, sz_input))
                #final_mask_c=np.zeros((sz_input, sz_input))
            # redefine past_img and new_img
            del past_img, new_img
            past_img=np.squeeze(img)
            new_img=whole_img
            #plt.imshow(np.squeeze(new_img))
            #del idx_nms
            del boxes_nms, scores_nms, labels_nms,\
                whole_img
            
            if not flag_adaptive:
                assert((it==0 or it==1))
                break
            
        masks[:,:,i] = np.max(masks_frame, axis=2)
        masks_colors[:,:,i] = np.max(masks_frame_c, axis=2)
        #del final_mask, final_mask_c #masks_frame
    return {
    "num_cell_test": num_cell_test,
    "num_cell_test2": num_cell_test2,
    "masks": masks,
    "masks_colors": masks_colors,
    "matrix_centers": matrix_centers,
    "output_yolo": output_yolo
    }
    

def SegmentCellPose3(model, imgs, diam):
    """
    Segment cells in given images using the provided CellPose 3 model.

    Parameters
    ----------
    model : object
        A segmentation model object that has an `eval` method. The `eval` method
        should accept parameters:
            - imgs: list or array of images to segment
            - diameter: expected cell diameter
            - flow_threshold: threshold for flow output (default 0.3)
            - cellprob_threshold: threshold for cell probability (default -1)
            - channels: list or tuple specifying image channels to use (default [0,0])

        The `eval` method should return a tuple where the first element is a list of 
        segmentation masks (one per image).

    imgs : list or np.ndarray
        Images to segment. Should be a list or numpy array of images. Each image can 
        be a numpy array.

    diam : float or int
        Expected diameter of the cells to segment.

    Returns
    -------
    Dictionary that contains
    masks : list of np.ndarray
        List of segmentation masks produced by the model. Each mask is a 2D array
        with labeled cells (integers > 0).

    num_cell_test2 : np.ndarray of float32
        Array containing the number of unique cells (excluding background 0) detected
        in each mask.

    Raises
    ------
    ValueError
        If `imgs` is not a list or numpy array.
        If `diam` is not a positive number.

    Notes
    -----
    - The function counts unique non-zero labels in each mask to estimate the number of cells.
    - `num_cell_test2` uses float32 dtype to allow compatibility with further processing
      that might expect floats.

    Example
    -------
    >>> masks, counts = segment_cells(model, imgs, diam=30)
    """
    if not isinstance(imgs, (list, np.ndarray)):
        raise ValueError("`imgs` must be a list or numpy array of images.")
    
    if not (isinstance(diam, (int, float)) and diam > 0):
        raise ValueError("`diam` must be a positive integer or float.")


    masks, _, _ = model.eval(imgs,
                            diameter=diam,
                            flow_threshold=0.3,
                            cellprob_threshold=-1, 
                            channels=[0, 0])
    # compute number of cells
    num_cell_test2 = CountCellsForMasks(masks)

    return {
    "masks": masks,
    "num_cell_test2": num_cell_test2
    }




def SegmentCellPose4(model, imgs, batch_size=8):
    """
    Segment cells in given images using the provided CellPose 3 model.

    Parameters
    ----------
    model : object
        A segmentation model object that has an `eval` method.
        
    imgs : list or np.ndarray
        Images to segment. Should be a list or numpy array of images. Each image can 
        be a numpy array.

    Returns
    -------
    masks : list of np.ndarray
        List of segmentation masks produced by the model. Each mask is a 2D array
        with labeled cells (integers > 0).

    num_cell_test2 : np.ndarray of float32
        Array containing the number of unique cells (excluding background 0) detected
        in each mask.

    Raises
    ------
    ValueError
        If `imgs` is not a list or numpy array.

    Notes
    -----
    Dictionary that contains: 
    - The function counts unique non-zero labels in each mask to estimate the number of cells.
    - `num_cell_test2` uses float32 dtype to allow compatibility with further processing
      that might expect floats.
     """
     
    if not isinstance(imgs, (list, np.ndarray)):
        raise ValueError("`imgs` must be a list or numpy array of images.")
    
    if batch_size is None:
        masks_predicted = model.eval(imgs)
        masks = masks_predicted[0]
    else:
        masks = []
        for i in range(0, len(imgs), batch_size):
            masks_batch, _, _ = model.eval(imgs[i:i+batch_size])
            masks.extend(masks_batch)
        
    # compute number of cells
    num_cell_test2 = CountCellsForMasks(masks)
    return {
    "masks": masks,
    "num_cell_test2": num_cell_test2
    }




def CountCellsForMasks(masks):
    if isinstance(masks, list):
        return np.array([len(np.unique(mask[mask != 0])) for mask in masks], dtype=np.float32)
    else:
        return np.array([len(np.unique(masks[masks != 0]))], dtype=np.float32)


    

    
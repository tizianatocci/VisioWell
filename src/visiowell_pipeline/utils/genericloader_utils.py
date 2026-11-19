# -*- coding: utf-8 -*-
"""
Functions for the GenericLoader class.

@author: Tiziana Tocci
"""


def LoadParameters(version_network, **kwargs):
    """
    Load and return configuration parameters for cell identification based on the specified network version.

    Default parameters can be overridden by passing keyword arguments (**kwargs).

    Parameters
    ----------
    version_network : str
        The network version to use. Options are:
        - "yolo_tf": YOLO implementation based on TensorFlow.
        - "yolo_ultralytics": YOLO implementation based on Ultralytics.
        - "cellpose3": Cellpose version 3.
        - "cellpose4": Cellpose version 4.

    **kwargs : dict, optional
        Additional parameters to override defaults depending on the network type.

        For "yolo_tf":
            - thresh (float): confidence threshold.
            - adaptive (int): flag to enable adaptive thresholding.
            - train_date (str): training date identifier.
            - network_to_load (str): name of the network model to load.
            - training_to_load (str): path to the training folder.
            - sz_input (int): input image size.
            - min_r (int): minimum expected cell radius.
            - max_r (int): maximum expected cell radius.
            - dist_centers (int): minimum distance between cell centers.

        For "yolo_ultralytics":
            - thresh (float)
            - adaptive (int)
            - train_date (str)
            - sz_input (int)
            - min_r (int)
            - max_r (int)
            - dist_centers (int)

        For "cellpose3":
            - train_date (str)
            - training_to_load (str)
            - dim_reshape (tuple): target image dimensions for resizing.
            - diam (int): average cell diameter.

        For "cellpose4":
            - train_date (str)
            - training_to_load (str)
            - dim_reshape (tuple)

    Returns
    -------
    dict_param : dict
        A dictionary containing the parameters configured for the specified network.
    """

    train_date = kwargs.get('train_date', "20250403")
    folder_training = "training_MB_250925"
    if version_network=="yolo_tf":
        thresh = kwargs.get('thresh', 0)
        adaptive = kwargs.get('adaptive', 1)
        network_to_load = kwargs.get('network_to_load', "yolo_v8_m_backbone")
        training_to_load = kwargs.get('training_to_load', f"{folder_training}/yolotftraining1/weights_{train_date}sherlock.h5")
        sz_input = kwargs.get('sz_input',  320)
        min_r = kwargs.get('min_r', 15) 
        max_r = kwargs.get('max_r', 18) 
        dist_centers = kwargs.get('dist_centers', 15) 
        dict_param={"thresh": thresh,
            "adaptive": adaptive,
            "train_date": train_date,
            "network_to_load": network_to_load,
            "training_to_load": training_to_load,
            "sz_input": sz_input,
            "min_r": min_r,
            "max_r": max_r,
            "dist_centers": dist_centers
        }
            
    elif version_network=="yolo_ultralytics":
        thresh = kwargs.get('thresh', 0)
        adaptive = kwargs.get('adaptive', 1)
        sz_input = kwargs.get('sz_input',  640)
        min_r = kwargs.get('min_r', 30) 
        max_r = kwargs.get('max_r', 36) 
        dist_centers = kwargs.get('dist_centers', 30) 
        dict_param={
            "thresh": thresh,
            "adaptive": adaptive,
            "train_date": train_date,
            "sz_input": sz_input,
            "min_r": min_r,
            "max_r": max_r,
            "dist_centers": dist_centers
        }
    
    elif version_network=="cellpose3":
        training_to_load = kwargs.get('training_to_load', f"{folder_training}/cellpose3training06/training_training06")
        dim_reshape = kwargs.get('dim_reshape', (640, 640))
        diam = kwargs.get('diam', 90)
        dict_param={
            "dim_reshape" : dim_reshape,
            "diam" : diam,
            "training_to_load": training_to_load,
            "train_data": train_date}


    elif version_network=="cellpose4":
        training_to_load = kwargs.get('training_to_load', f"{folder_training}/cellpose4training00/sublot_model_7_c37de916c8814054ba0e1af420e403bc")
        dim_reshape = kwargs.get('dim_reshape', (256, 256))
        dict_param={
            "dim_reshape" : dim_reshape,
            "training_to_load": training_to_load,
            "train_data": train_date}

    else:
        # Raise error if unknown network version is provided
        raise ValueError(f"Unknown network version: {version_network}")

    return dict_param  




def load_trained_yolo(dir_network, num_class=1, yolov8backbone="yolo_v8_xs_backbone_coco"):
    """
    Load trained yolo network.

    Parameters
    ----------
    dir_network : string. path to the weights of the trained network.
    num_class : int. Number of classes.
    yolov8backbone : yolo v8 backbone.
        DESCRIPTION. The default is "yolo_v8_xs_backbone_coco".

    Returns
    -------
    yolo: network with the loaded weights.

    """
    import keras_cv
    if not isinstance(dir_network, str):
        raise TypeError("'dir_network' should be a string but got {type(dir_network)}")
    if not isinstance(num_class, int):
        raise TypeError("'num_class' should be an int but got {type(num_class)}")
    if not isinstance(yolov8backbone, str):
        raise TypeError("'yolov8backbone' should be a string but got {type(yolov8backbone)}")
    
    # Set backbone, network and weights.
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        yolov8backbone, load_weights=False
        # We will use yolov8 small backbone with coco weights
    )
    yolo = keras_cv.models.YOLOV8Detector(
        num_classes=num_class,
        bounding_box_format="xywh",
        backbone=backbone,
        fpn_depth=1,
    )
    yolo.load_weights(dir_network)
    
    return yolo


def LoadTrainedNetwork(which_network, dict_param):
    import os
    if dict_param is None:
        raise ValueError("Missing 'dict_param'. Please pass a dictionary with required parameters.")
    required_keys = ["training_to_load"]
    for key in required_keys:
        if key not in dict_param:
            raise KeyError(f"Missing required key '{key}' in dict_param.")
        
    training_to_load=dict_param["training_to_load"]
    file_network = f"../Trainings/{training_to_load}" # path of the trained neural network file
    folder_path = os.path.dirname(file_network)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    if not os.path.isfile(file_network):
        raise FileNotFoundError(f"Weight file not found: {file_network}")
        
    if which_network=="yolo_tf":
        if "network_to_load" not in dict_param:
                    raise KeyError("Missing 'network_to_load' in dict_param for 'yolo_tf' model.")
        network_to_load=dict_param["network_to_load"]
        model = load_trained_yolo(file_network, num_class=1, yolov8backbone=network_to_load)
        print(f"[INFO] Loaded YOLO model from {file_network}")

    elif which_network in ("cellpose3", "cellpose4"):
        from cellpose import models       
        model = models.CellposeModel(pretrained_model=file_network, gpu=True) 
        print(f"[INFO] Loaded Cellpose model from {file_network}")
    else:
        raise ValueError("'which_network' must be 'yolo_tf' or 'cellpose'")
    return model      
# Detection
from ..utils.genericloader_utils  import LoadParameters, LoadTrainedNetwork 
from ..utils.celldetector_utils import DetectTestYolo8TF, SegmentCellPose3, SegmentCellPose4,\
    EvaluateGoodWellTF, EvaluateGoodWellCP3, EvaluateGoodWellCP4,\
    PrepareDataTF, PrepareDataCP3

# Detection
class NetworkLoader(): # for loading parameters and network
    def __init__(self):
        pass
    
    def load_parameters(self, version_network, **kwargs):
        return LoadParameters(version_network, **kwargs)
    
    def load_network(self, version_network, dict_param):
        return LoadTrainedNetwork(version_network, dict_param)
    
    
    
class CellDetector():
    def __init__(self, type_network):
        self.type_network = type_network   
        # TF
        self.sz_input = None
        self.thresh = None
        self.flag_adaptive = None
        self.min_r = None            
        self.max_r = None
        self.dist_centers = None
        # CP3 and CP4
        self.dim_reshape = None
        # CP3
        self.diam = None
        

    def retrieve_parameters_for_detection(self, dict_param):
        if self.type_network=="yolo_tf":
            self.sz_input = dict_param["sz_input"]
            self.thresh = dict_param["thresh"]
            self.flag_adaptive = dict_param["adaptive"]
            self.min_r = dict_param["min_r"]            
            self.max_r = dict_param["max_r"]
            self.dist_centers = dict_param["dist_centers"]
        elif self.type_network=="cellpose3":
            self.dim_reshape=dict_param["dim_reshape"]
            self.diam = dict_param["diam"]
        elif self.type_network=="cellpose4":
            self.dim_reshape = dict_param["dim_reshape"]
            
        
    def prepare_data(self, video):
        if self.type_network=="yolo_tf":
            return PrepareDataTF(video)
        elif self.type_network in ("cellpose3", "cellpose4"):
            return PrepareDataCP3(video, self.dim_reshape)
            
            
    def evaluate_first_frame(self, model, images_test, **kwargs):
        if self.type_network=="yolo_tf":
            return EvaluateGoodWellTF(model, images_test,\
                   self.sz_input, self.thresh, self.flag_adaptive,\
                   self.min_r, self.max_r, self.dist_centers, **kwargs)
            
        elif self.type_network=="cellpose3":
            return EvaluateGoodWellCP3(model, images_test, self.diam)
        
        elif self.type_network=="cellpose4":
            return EvaluateGoodWellCP4(model, images_test)
           
        
    def detect_cells(self, Nframe=None, yolo=None, images_test=None,
                     model=None, imgs=None, **kwargs):
        if self.type_network == "yolo_tf":
            return DetectTestYolo8TF(Nframe, yolo, images_test,\
                   self.sz_input, self.thresh, self.flag_adaptive,\
                   self.min_r, self.max_r, self.dist_centers, **kwargs)
        elif self.type_network == "cellpose3":
            if model is None or imgs is None:
                raise ValueError("For cellpose3 detection, 'model', 'imgs', and 'diam' must be provided.")
            return SegmentCellPose3(model, imgs, self.diam)
        elif self.type_network == "cellpose4":
            if model is None or imgs is None:
                raise ValueError("For cellpose4 detection, 'model', 'imgs' must be provided.")
            return SegmentCellPose4(model, imgs)
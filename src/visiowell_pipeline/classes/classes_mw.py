# Wells identifier
from ..utils.microwellpreprocesser_utils import RetrieveParameters, LoadStack, CorrectDriftVideo, RangeAdjustment, RotateVideo
from ..utils.microwellfinder_utils import FindMicrowellsCHT, ShowCircles, CutRectangleAroundMicrowell, FilterCrops, VisualizeRectangles
from ..utils.microwellnamer_utils import ExtractCenters, OrderPointsForNaming, AssignNames, VisualizeNaming

    
# Wells
class MicrowellPreprocesser():
    def __init__(self):
        pass
    def retrieve_parameters():
        pass
    def load_images(self, path_exp_pos, num_tile):
        return LoadStack(path_exp_pos, num_tile)
    
    def correct_drift(self, images_array, mode_shift, nptype, ref_frame):
        return CorrectDriftVideo(images_array, mode_shift, nptype, ref_frame)
        
    def range_adjustment(self, frames, mode):
        return RangeAdjustment(frames, mode)
    
    def rotate_video(self, frames, rotation_angle):
        return RotateVideo(frames, rotation_angle)
    
    
class MicrowellFinder():
    def __init__(self, img):
        self.img = img
        self.circles = None
        
    def find_microwells(self, dp_value,\
        min_dist, param_1, param_2, min_rad, max_rad,\
        flag_plot=False, title=None):
        self.circles = FindMicrowellsCHT(self.img, dp_value,\
            min_dist, param_1, param_2, min_rad, max_rad,\
            flag_plot=False, title=None)
        return self.circles
        
    def show_circles(self):
        import numpy as np
        if self.circles is not None:
            ShowCircles(self.img, np.squeeze(self.circles))
        else:
            print("circles is None, no circles to show.")
            
    def cut_rectangles(self, sz_rect, array, ref_frame=0, flag_plot=False):
        return CutRectangleAroundMicrowell(\
            self.circles, sz_rect, array, ref_frame=ref_frame,\
            flag_plot=flag_plot)
            
    def show_rectangles(self, array, dict_crops, sz_rect, ref_frame=0):
        VisualizeRectangles(array, dict_crops, sz_rect, ref_frame)
            
    def filter_crops(self, dict_crops, sz_rect, array,\
                    ref_frame=0, flag_plot=False, mode="flexible"):
        return FilterCrops(dict_crops, sz_rect, array, ref_frame, flag_plot, mode)
    
    
class MicrowellNamer():
    def __init__(self):
        pass
    
    def extract_centers(self, dict_final_crops):
        return ExtractCenters(dict_final_crops)
    def order_points(self, corrected_array, final_centers,\
                                angle_tolerance, distance_threshold,\
                                user_threshold_angle, ref_frame, flag_plot):
        return OrderPointsForNaming(corrected_array, final_centers,\
                                    angle_tolerance, distance_threshold,\
                                    user_threshold_angle, ref_frame, flag_plot)
    def assign_names(self, final_sorted_data, dict_final_crops):
        return AssignNames(final_sorted_data, dict_final_crops)
        
    def visualize_naming(self, corrected_array, merged_info, ref_frame=0, flag_plot=False):
        return VisualizeNaming(corrected_array, merged_info, ref_frame, flag_plot)
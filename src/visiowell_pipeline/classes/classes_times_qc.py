# TimeRetrieval
from ..utils.timeretriever_utils import LoadCsvFile,\
    FindColonySize, chronological_model_block_lineage, ConvertFrameInTime, FindQualityIndexes, FindDiscrepancies, ComputeTrustIndex

# QC
from ..utils.QC_utils import FindFilesNameInFolderWtExtension, LoadCSV,\
    RetrieveNameWells,IsNamingWeird, RetrievePositionsWeirdNaming,\
    LoadFigNaming,\
    FindWeirdTimes, CreateVideoForWeirdResults, PlotCurve,\
    VisualizeVideoBBScore


# Time Retrieval
class TimeRetriever():
    def __init__(self):
        pass

    def load_csv_file(self, dir_results, result_file, sep=";"):
        return LoadCsvFile(dir_results, result_file, sep)
    def find_colony_size(self, df):
        return FindColonySize(df)
    def find_division_time(self, df):
        return chronological_model_block_lineage(df)
    def convert_frame_in_time(self, result, imaging_frequence):
        return ConvertFrameInTime(result, imaging_frequence)
    def find_quality_indexes(self, df):
        return FindQualityIndexes(df)
    def find_discrepancies(self, result, time_interval_discrepancy):
        return FindDiscrepancies(result, time_interval_discrepancy)
    def compute_trust_index(self, oscillation_indicator, flag_discrepancy, colony_size, colony_size_limit): 
        return ComputeTrustIndex(oscillation_indicator, flag_discrepancy, colony_size, colony_size_limit) 


# QC
class FilePreprocesser():
    def __init__(self, path):
        self.path = path
        
    def find_name_files(self, ext="*"): # if folder
        self.name_files = FindFilesNameInFolderWtExtension(self.path,\
            extension = ext)
        return self.name_files
    
    def load_csv(self): # if file
        return LoadCSV(self.path)
    
        
class QualityController():
    def __init__(self):
        self.name_files = None # for microwell
        self.df_original = None # for output times
        self.df_weird_times = None

    # QC MW
    def set_name_files(self, name_files):
        self.name_files = name_files
    
    def extract_name_of_wells(self, num_file: int):
        return RetrieveNameWells(self.name_files[num_file])
    
    def is_naming_weird(self, name_mws):
        return IsNamingWeird(self, name_mws)
    
    def retrieve_positions_weird_naming(self#, limit_letter="g"
                                        ):
        return RetrievePositionsWeirdNaming(self.name_files,\
            #limit_letter=limit_letter
            )
    
    def load_fig_naming(self, folder, name_pos):
        return LoadFigNaming(folder, name_pos)
    
    # QC detection
    def visualize_bb_score(self, pos, mw, dir_wells, dir_results, dim_resize):
        return VisualizeVideoBBScore(pos, mw,\
               dir_wells, dir_results, dim_resize)
        
        
    # QC TIMES
    def set_df_original(self, df_original):
        self.df_original = df_original
        
    def retrieve_weird_times(self, imaging_frequence):
        self.df_weird_times = FindWeirdTimes(self.df_original, imaging_frequence)
        return self.df_weird_times
    
    def visualize_video_weird_times(self, num_index, dir_wells, dir_results, num_frames):
        video = CreateVideoForWeirdResults(self.df_weird_times, num_index,\
            dir_wells, dir_results, num_frames=num_frames)
        return video
    
    def load_curve(self, num_index, dir_results):
        return PlotCurve(self.df_weird_times, num_index, dir_results)
    
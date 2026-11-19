from src.visiowell_pipeline.utils.celldetector_utils import resize_video, PrepareDataCP3, CountCellsForMasks, SegmentCellPose4
from src.visiowell_pipeline.routines.functions_general import load_pickle_file
from src.visiowell_pipeline.routines.functions_postcorrection import correction_CPD, plot_curve, extract_gt
from src.visiowell_pipeline.routines.functions_detection import obtain_csv


import numpy as np
import matplotlib.pyplot as plt 
#from skimage.color import label2rgb
import time
import os

YOLO_TRAININGS = {
    "yolov8": { # change paths
        "path": r"C:\Users\tizia\OneDrive - INSTITUT CURIE\UnitPC_training\Ultralytics\model_5b8ee120b4154cafa940606c1933c2db",
        "sz_img": 256,
    },
    "yolov11": {
        "path": r"C:\Users\tizia\OneDrive - INSTITUT CURIE\UnitPC_training\Ultralytics\model_9f4e77befae74e8099b3c968e99208e1",
        "sz_img": 256,
    },
}
YOLO_WEIGHTS = "weights/last.pt"

def load_video(path_exp, name_file_to_load, name_well):
    x = load_pickle_file(path_exp, name_file_to_load)
    video = x[name_well]["crop"]
    return video

def init_model(choice_model):
    if choice_model == "microsam":
        from micro_sam.automatic_segmentation import get_predictor_and_segmenter
        predictor, segmenter = get_predictor_and_segmenter(model_type="vit_l_lm")
        return {"name": choice_model, "predictor": predictor, "segmenter": segmenter}
    if choice_model == "cellposesam":
        from cellpose import models
        return {"name": choice_model, "model": models.CellposeModel(gpu=True)}
    if choice_model in YOLO_TRAININGS:
        from ultralytics import YOLO
        cfg = YOLO_TRAININGS[choice_model]
        model = YOLO(os.path.join(cfg["path"], YOLO_WEIGHTS))
        return {"name": choice_model, "model": model, "sz_img": cfg["sz_img"]}

    raise ValueError(f"Unknown model: {choice_model}")

def _natural_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def run_yolo(ctx, path_exp, name_series, name_well):
    from comparison_models.from_nparray_to_tif import write_frames_from_pkl_file
    import tempfile
    name_subfolder = f"{name_series}_{name_well.upper()}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        # writes one 800x800 .tif per frame into tmp_dir/name_subfolder
        write_frames_from_pkl_file(path_exp, name_series, f"{name_series}.pkl",
                                   tmp_dir, name_well, name_subfolder, flag_show=False)

        path_frames = os.path.join(tmp_dir, name_subfolder)
        files = [f for f in os.listdir(path_frames) if f.lower().endswith((".tif", ".tiff"))]
        files = [os.path.join(path_frames, f) for f in sorted(files, key=_natural_key)]

        results = ctx["model"](files, imgsz=ctx["sz_img"], verbose=False)
        return np.array([len(r.boxes) for r in results], dtype=int)

def run_microsam(ctx, path_exp, name_series, name_well):
    from micro_sam.automatic_segmentation import automatic_instance_segmentation
    
    video = load_video(path_exp, f"{name_series}.pkl", name_well)
    Nframe = video.shape[-1]
    video = resize_video(video, (256, 256))
    counts = np.zeros(Nframe, dtype=int)
    for i in range(Nframe):
        mask = automatic_instance_segmentation(predictor=ctx["predictor"],
                   segmenter=ctx["segmenter"], input_path=video[:, :, i],
                   ndim=2, verbose=False)
        counts[i] = CountCellsForMasks(mask)
    return counts

def run_cellposesam(ctx, path_exp, name_series, name_well):
    video = load_video(path_exp, f"{name_series}.pkl", name_well)
    _, video = PrepareDataCP3(video, (256, 256))
    output = SegmentCellPose4(ctx["model"], video, batch_size=8)
    return np.asarray(output["num_cell_test2"])

RUNNERS = {
    "microsam": run_microsam,
    "cellposesam": run_cellposesam,
}
RUNNERS.update({name: run_yolo for name in YOLO_TRAININGS})


def runner(ctx, path_exp, name_series, name_well):
    return RUNNERS[ctx["name"]](ctx, path_exp, name_series, name_well)

def postprocess_and_save(counts, name_experiment, name_series, name_well, dir_results):
    times = np.arange(len(counts))
    _, final_signal, _ = correction_CPD(counts.copy(), times, "pelt", "l2", pen=2)
    gt = extract_gt(name_experiment, name_series, name_well, times)
    plot_curve(times, counts, final_signal, name_experiment, name_series, name_well, dir_results, gt=gt)
    obtain_csv(counts, final_signal, dir_results, name_experiment, name_series, name_well, gt)
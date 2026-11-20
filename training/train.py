# -*- coding: utf-8 -*-
"""
Train Cellpose SAM
@author: Tiziana Tocci
"""

"""
## Libraries
"""
import os

from functions_train import load_masks_and_images, resize_video, split_train_val, train_model
import numpy as np
from hyperopt import Trials, fmin, hp, tpe, space_eval
import mlflow

"""
### Parameters
old: sz_img=256, sublot_training: True, epochs=50
"""
sz_img =  256 #256
split_ratio = 0.20 # divide images in train and val
seed= 42
imgs_dir = "../training_20251205" #r"C:\Users\team_\Documents\Tiziana\UltralyticsTraining\training_20250925"
masks_dir = "../training_20251205/Masks" #r"C:\Users\team_\Documents\Tiziana\UltralyticsTraining\training_20250925\Masks"

user_params = {"sz_img": sz_img, "split_ratio": split_ratio, "seed":seed,\
               "imgs_dir": imgs_dir, "masks_dir": masks_dir,\
                "sublot_training": False, "sublot_size":100}

"""
### Prepare the data
"""
# load masks and images
images, masks = load_masks_and_images(masks_dir, imgs_dir)
# resizing
images = resize_video(images, (sz_img, sz_img))
masks = resize_video(masks, (sz_img, sz_img))

# filtered empty
#valid_idx = [i for i in range(masks.shape[-1]) if masks[:, :, i].sum() > 0]
#images = images[:, :, valid_idx]
#masks  = masks[:, :, valid_idx]

# splits datasets for training and test
images=[images[:, :, i] for i in range(images.shape[-1])]
masks=[masks[:, :, i] for i in range(masks.shape[-1])]
#images = images[:10]
#masks=masks[:10]
train_x, train_y,\
    test_x, test_y = split_train_val(images,\
    masks, split_ratio, seed=seed)

# infer signature
#signature = infer_signature(train_x, train_y)

"""
### CellposeSAM model
"""

space = {
    "lr": hp.loguniform("lr", np.log(1e-5), np.log(5e-3)),
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-3)),
    "epochs": 50,
    "batch_size": 8,
}
suffix = f"nepochs_{space['epochs']}" # to distinguish runs

from hyperopt import STATUS_OK, STATUS_FAIL

def objective(params):
    import traceback
    try:
        print(f"Starting trial with params: {params}")
        result = train_model(
            params,
            train_x = train_x,
            train_y = train_y,
            test_x = test_x,
            test_y = test_y,
            user_params = user_params
        )
        print(f"Trial completed with fscore: {result.get('fscore', 'N/A')}")
        return result
    except Exception as e:
        print(f"Trial failed with error: {e}")
        traceback.print_exc()
        return {
            "loss": 9999,
            "status": STATUS_FAIL,
            "fscore": 0.0,
            "average_precision": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "train_loss": [9999],
            "test_loss": [9999],
            "model_path": "N/A"
        }



mlflow.set_experiment(f"/CellposeSAM_{suffix}")
with mlflow.start_run():
    # Conduct the hypermeter search with HyperOpt
    trials=Trials()
    best = fmin(
        fn = objective,
        space=space,
        algo= tpe.suggest, 
        max_evals = 3,
        trials = trials
    )


best_run = max(trials.results, key=lambda x: x["fscore"])
best_params = space_eval(space, best)

with mlflow.start_run(run_name=f"Best_Run_Summary_{suffix}"):
    # log best params
    mlflow.log_params(best_params)
    mlflow.log_param("model_path", best_run["model_path"])       
    mlflow.log_params({
        "sz_img": user_params["sz_img"],
        "split_ratio": user_params["split_ratio"],
        "seed": user_params["seed"],
        "imgs_dir": user_params["imgs_dir"],
        "masks_dir": user_params["masks_dir"]
    })

    # log metrics
    mlflow.log_metric("average_precision", best_run["average_precision"])
    mlflow.log_metric("precision", best_run["precision"])
    mlflow.log_metric("recall", best_run["recall"])
    mlflow.log_metric("fscore", best_run["fscore"])

    # log last point of losses
    mlflow.log_metric("objective_loss", best_run["loss"])
    mlflow.log_metric("train_loss_final", best_run["train_loss"][-1])
    mlflow.log_metric("test_loss_final", best_run["test_loss"][-1])

    # Print out the best parameters and corresponding loss
    print(f"Best parameters: {best_params}")
    print(f"Best fscore loss: {best_run['fscore']}")

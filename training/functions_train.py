def load_masks_and_images(data_dir, data_dir_frames):
    import numpy as np
    import os
    import cv2

    npy_files = os.listdir(data_dir)
    files = {} # loaded .npy objects
    masks = [] # list of masks
    name_files = [] #list of namefiles for images
    images = [] # list of images
    for i in range(len(npy_files)):
        files[i] = np.load(os.path.join(data_dir, npy_files[i]), allow_pickle=True)
        masks.append(files[i].item()["masks"])
        name_files.append(os.path.join(data_dir_frames, find_substring(npy_files[i], "", "_seg.npy") + ".tif"))
        images.append(cv2.imread(name_files[i], cv2.IMREAD_UNCHANGED))
    masks = np.array(masks).astype(np.uint16)
    masks = transpose_array_img(masks)
    images = np.array(images)
    images = transpose_array_img(images)
    assert(images.shape[-1]==masks.shape[-1])
    return images, masks



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
    import numpy as np
    # if the array is made up of grayscale images
    if images_array.ndim == 3:
        if order_tuple==None:
            order_tuple=(1,2,0)
        images_array=np.transpose(images_array, order_tuple)
    # if the array is made up of RGB images
    elif images_array.ndim == 4:
        if order_tuple==None:
            order_tuple=(1,2,3,0)
        images_array=np.transpose(images_array, order_tuple)        
    return images_array

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
    import re
    result = re.search(f'{first_substring}(.*){second_substring}', user_string)
    return result.group(1)


def from_video_nparray_to_list(images):
    images_list = [images[:, :, i] for i in range(images.shape[-1])]
    return images_list


def resize_video(video, new_size):
    import cv2
    import numpy as np
    n_frames = video.shape[-1]
    resized_video = np.zeros((new_size[0], new_size[1], n_frames), dtype=video.dtype)
    for i in range(n_frames):
        resized_video[:,:,i] = cv2.resize(video[:,:,i], new_size, interpolation=cv2.INTER_LINEAR)
    return resized_video

def split_train_val(images_list, bbox_list, split_ratio, seed=42):
    import random
    zipped = list(zip(images_list, bbox_list))
    random.seed(seed) 
    random.shuffle(zipped)
    train_size = int((1-split_ratio) * len(zipped))  # 80% for training
    train_data = zipped[:train_size]
    val_data = zipped[train_size:]
    train_images, train_boxes = zip(*train_data)
    val_images, val_boxes = zip(*val_data)
    train_images=list(train_images)
    train_boxes=list(train_boxes)
    val_images=list(val_images)
    val_boxes=list(val_boxes)
    return train_images, train_boxes, val_images, val_boxes

def batch_eval(model, images, batch_size=8):
    all_masks = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        masks_batch = model.eval(batch)[0]  
        all_masks.extend(masks_batch)
    return all_masks


def train_model(params, train_x, train_y, test_x, test_y, user_params):
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import mlflow
    import torch, gc
    from cellpose import models, metrics, train
    from hyperopt import STATUS_OK


    # define the model
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    
    # train the model
    with mlflow.start_run(nested=True):
        if user_params["sublot_training"]:
            model_path, train_losses, test_losses = train_with_sublots(
            model,
            train_x,
            test_x,
            train_y,
            test_y,
            batch_size=params["batch_size"],
            sublot_size=user_params["sublot_size"],
            total_epochs=params["epochs"]
        )
        else:
            model_path, train_losses, test_losses = train.train_seg(model.net,
            train_data = train_x,
            train_labels = train_y,
            test_data = test_x,
            test_labels = test_y,
            batch_size=params["batch_size"],
            n_epochs=params["epochs"],
            weight_decay=params["weight_decay"],
            learning_rate=params["lr"],
            min_train_masks=0,
            model_name=f"model_{mlflow.active_run().info.run_id}")
        print("Train finished.")

        # run model on test images
        #masks_predicted = model.eval(test_x)[0]
        trained_model = models.CellposeModel(gpu=True, pretrained_model=model_path)
        masks_predicted = batch_eval(trained_model, test_x, batch_size=8)

        # compute average precision
        ap = metrics.average_precision(test_y, masks_predicted)[0]
        ap_mean = ap[:,0].mean()
        # compute precision, recall, f-score
        precision, recall, fscore = metrics.boundary_scores(test_y, masks_predicted, scales=[1])
        precision = float(np.nanmean(precision))
        recall = float(np.nanmean(recall))
        fscore = float(np.nanmean(fscore))

        # plot losses
        path_file = "./PlotsTraining"
        os.makedirs(path_file, exist_ok=True)
        fig_name = f"loss_curve_{mlflow.active_run().info.run_id}.png"
        path_fig = os.path.join(path_file, fig_name)
        fig=plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(path_fig)
        plt.close(fig) 

        # Log the parameters and results
        mlflow.log_params(params)
        mlflow.log_param("model_path", model_path)  
        mlflow.log_param("imgs_dir", user_params["imgs_dir"])
        mlflow.log_param("masks_dir", user_params["masks_dir"])

        # for graphs of losses
        obj_loss = 1-fscore
        mlflow.log_metric("objective_loss", obj_loss)
        for i, loss in enumerate(train_losses):
            mlflow.log_metric("train_loss", loss, step=i)
        for i, loss in enumerate(test_losses):
            mlflow.log_metric("test_loss", loss, step=i)
        mlflow.log_metric("average_precision", ap_mean)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("fscore", fscore)
        mlflow.log_param("sz_img", user_params["sz_img"])
        mlflow.log_param("split_ratio", user_params["split_ratio"])
        mlflow.log_param("seed", user_params["seed"])
        mlflow.log_param("sublot_training", user_params["sublot_training"])
        mlflow.log_param("sublot_size", user_params["sublot_size"])

        # log the artifact
        mlflow.log_artifact(model_path, artifact_path="cellpose_model")
        mlflow.log_artifact(path_fig, artifact_path="plots")
        
        del model, trained_model
        torch.cuda.empty_cache()
        gc.collect()

        return {"loss": obj_loss,\
                "train_loss": train_losses, "test_loss": test_losses,\
                "average_precision": ap_mean,
                "precision": precision, "recall": recall, "fscore": fscore,
                "status": STATUS_OK, "model_path": model_path}




def safe_boundary_scores(y_true_list, y_pred_list):
    import numpy as np
    from cellpose import metrics

    precisions, recalls, fscores = [], [], []
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        has_gt = np.any(y_true > 0)
        has_pred = np.any(y_pred > 0)

        if not has_gt and not has_pred:
            # both empty
            precisions.append(1.0)
            recalls.append(1.0)
            fscores.append(1.0)
        elif not has_gt and has_pred:
            # false positive
            precisions.append(0.0)
            recalls.append(0.0)
            fscores.append(0.0)
        elif has_gt and not has_pred:
            # false negative
            precisions.append(0.0)
            recalls.append(0.0)
            fscores.append(0.0)
        else:
            # both have masks
            p, r, f = metrics.boundary_scores([y_true], [y_pred], scales=[1])
            precisions.append(np.mean(p))
            recalls.append(np.mean(r))
            fscores.append(np.mean(f))

    # means
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(fscores))


import math

def train_with_sublots(model, images, test_images, masks, test_labels, batch_size=8, sublot_size=100, total_epochs=50, min_train_masks=0):
    """
    Train Cellpose model using sublots (mini-datasets) to handle large datasets.
    """
    import math
    #import random
    from cellpose import train
    import mlflow

    #combined = list(zip(images, masks))
    #random.seed(42)
    #random.shuffle(combined)
    #images, masks = zip(*combined)
    #images, masks = list(images), list(masks)

    n_sublots = math.ceil(len(images) / sublot_size)
    epochs_per_sublot = max(1, total_epochs // n_sublots)

    all_train_losses, all_test_losses = [], []

    for i in range(n_sublots):
        start = i * sublot_size
        end = min((i+1) * sublot_size, len(images))
        sublot_images = images[start:end]
        sublot_masks = masks[start:end]

        print(f">>> Training sublot {i+1}/{n_sublots}, {len(sublot_images)} images, {epochs_per_sublot} epochs")

        model_name = f"sublot_model_{i+1}_{mlflow.active_run().info.run_id}"

        model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=sublot_images,
            train_labels=sublot_masks,
            test_data=test_images,
            test_labels=test_labels,
            batch_size=batch_size,
            n_epochs=epochs_per_sublot,
            min_train_masks=min_train_masks,
            model_name=model_name
        )

        all_train_losses.extend(train_losses)
        all_test_losses.extend(test_losses)

    return model_path, all_train_losses, all_test_losses

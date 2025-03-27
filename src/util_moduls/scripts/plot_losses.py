import os, sys
import shutil
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

import torch
import torch.functional as F
import torch.nn as nn
import cv2
from data.dataloader_vae import make_dataloaders
from util_moduls.args import args
import numpy as np

from src.util_moduls.utils_functions import load_checkpoint_with_shape_match, SphericalGrid
from matplotlib import pyplot as plt
from util_moduls.loss_funcs import MaskedMSELoss, CLIPloss
from model.AE import AE

from src.util_moduls.utils_functions import procrustes_align

import numpy as np
from scipy.spatial.distance import cdist
from roma import rotmat_geodesic_distance


if __name__ == "__main__":
    """
    Visualize the GT depth maps of the nuscenes dataset.
    
    """
   
    ##################### Load the model #####################
    
    def get_latest_file_in_folder(father_folder):
        # List all files in the folder
        
        
        # Pick the last created folder out of a list of all the folers within father_folder
        curr_folder = max([os.path.join(father_folder, file) for file in os.listdir(father_folder)], key=os.path.getctime)
        
        # Filter out directories and get the modification timestamps
        file_list = os.listdir(curr_folder)
        file_info_list = [(file, os.path.getmtime(os.path.join(curr_folder, file))) for file in file_list if os.path.isfile(os.path.join(curr_folder, file))]

        file_info_list = [file for file in file_info_list if file[0].endswith(".pth")]
        
        # Find the file with the latest modification timestamp
        latest_file = max(file_info_list, key=lambda x: x[1])[0] if file_info_list else None
        
        # An error message if the folder is empty:
        if latest_file is None:
            raise FileNotFoundError(f"No files found in {curr_folder}")
        
        # Return the absolute path of the latest file
        latest_file = os.path.join(curr_folder, latest_file)
        
        
        return latest_file
    
    
    dir_path = str(Path("/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE") / args.model)
    args.checkpoint = get_latest_file_in_folder(dir_path)
    
    args.checkpoint = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/stage_3/3/mlt_epoch_35_best_RMSE_2.5440254_aux_4.675e+00.pth"
    # args.model = "Depth"
    # args.input_channels = 5
    

    # If your GPU is currently occupied, you could simply use the CPU to visuazlie your recent predictions.
    use_gpu = True
    if use_gpu:
        cuda_id = 0
        with torch.cuda.device(cuda_id):
            # model  = CamRaDepth(input_channels=args.input_channels)
            # model = CamRaBin(input_channels=args.input_channels)
            model = AE(input_channels=args.input_channels,  mode=args.model)

            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            state = torch.load(args.checkpoint, map_location=device)
            load_checkpoint_with_shape_match(model, state['state_dict'])
            print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
            model.eval()
            model.to(device)
    else:
        # model  = CamRaDepth(input_channels=args.input_channels)
        # model = CamRaBin(input_channels=args.input_channels)
        model = AE(input_channels=args.input_channels, mode=args.model)
        device = torch.device('cpu')
        state = torch.load(args.checkpoint, map_location=device)
        load_checkpoint_with_shape_match(model, state['state_dict'])
        print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
        model.eval()
        model.to(device)

    # exit()
        
    ##################### Data #####################
    dataloaders = make_dataloaders(batchsize=1, split="test")
    # dataloaders = make_dataloaders(batchsize=1, split="test")
    test_dl = dataloaders["test"]
    dataloaders = make_dataloaders(batchsize=1, split="train", shuffle_training=False)
    trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
    splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
    keys, values = list(splits.keys()), list(splits.values())
    
    ##################### Create an output folder #####################
    path = Path(args.output_dir) /  "plots" # Where should it be created?
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    ##################### Plot the losses #####################
    train_losses = state["train_losses"]
    val_losses = state["val_losses"]
    
    print(train_losses.shape, val_losses.shape)
    
    fig, axes = plt.subplots(1, train_losses.shape[1], figsize=(10,10))
    for i, ax in enumerate(axes):
        if i == 0:
            ax.plot(train_losses[:,i], label="train")
            ax.plot(val_losses[:,i], label="val")
            ax.legend()
        else:
            ax.plot(train_losses[:,i])
        ax.set_title(f"Epoch {i}")
    plt.tight_layout()
    plt.savefig(path / "losses.png")
    plt.close()

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
from data.dataloader_vae import make_dataloaders, load_prepared_file_list
from util_moduls.args import args
import numpy as np

from src.util_moduls.utils_functions import load_checkpoint_with_shape_match, SphericalGrid, create_tqdm_bar
from matplotlib import pyplot as plt
from util_moduls.loss_funcs import MaskedMSELoss, CLIPloss
from model.AE import AE
import tifffile


import numpy as np


if __name__ == "__main__":
    
    train_num_samples, val_num_samples = args.train_val_split
    # files_list = load_prepared_file_list(args.split)
    
    # no_rain, with_rain = 0, 0
    # for idx in range(train_num_samples):
    #     name = files_list[idx][6].split('/')[-1].split('.')[0][:-3]
    #     if "rain" in name:
    #         with_rain += 1
    #     else:
    #         no_rain += 1
    # print(no_rain / train_num_samples, with_rain/ train_num_samples)
    dataloaders = make_dataloaders(batchsize=1, split="train")
    train_dl = dataloaders["train"]
    
    loop = create_tqdm_bar(train_dl, desc="Extract Anchors") # tqdm is needed for visualization of the training progess in the terminal
    day_count, rain_count, num_anchors = 0, 0, 300
    day_num, rain_num = int(0.7 * num_anchors), int(0.3 * num_anchors)
    
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AE(input_channels=args.input_channels, mode="stage_0").to(device)
    args.checkpoint = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/stage_0/12/mlt_epoch_18_best_RMSE_2.5297970_aux_4.318e+00.pth"
    state = torch.load(args.checkpoint, map_location=device)
    load_checkpoint_with_shape_match(model, state["state_dict"])
    
    model.eval()
    with torch.no_grad():
        with torch.autocast('cuda'): 
            acc_latent = None 
            for idx, sample_dict in loop:
                name = sample_dict["name"][0][:-3]
                
                if "rain" in name:
                    if rain_count == rain_num:
                        continue
                    rain_count += 1
                else:
                    if day_count == day_num:
                        continue
                    day_count += 1
                    
                x = sample_dict["image"].to(torch.float32).to(device)
                gt_x = sample_dict["gt"]["depth"]["augmented_depth"].to(torch.float32).to(device)
                latent = model.depth_encoder(gt_x + x)
                
                if acc_latent is None:
                    acc_latent = latent
                else:
                    acc_latent = torch.cat((acc_latent, latent), dim=0)
                    
                if acc_latent.shape[0] == num_anchors:
                    print("DONE")
                    exit()
                print("Accumelated Anchors shape:", acc_latent.shape)
            
            print("Accumelated Anchors shape:", acc_latent.shape)
            path = Path("/home/ubuntu/Datasets") / "Anchors"
            os.makedirs(path, exist_ok=True)
            tifffile.imwrite(path / f"anchors_{num_anchors}.tiff", acc_latent.detach().cpu().numpy())
    
        
        
        
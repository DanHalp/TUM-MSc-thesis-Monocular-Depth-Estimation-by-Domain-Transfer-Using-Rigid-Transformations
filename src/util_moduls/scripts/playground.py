import os, sys
import random
import shutil
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

import torch
import cv2
from data.dataloader_vae import make_dataloaders
from util_moduls.args import args
import numpy as np
from model.AE import AE
from src.util_moduls.utils_functions import load_checkpoint_with_shape_match, create_tqdm_bar

from util_moduls.loss_funcs import MaskedMSELoss

from scipy.stats import norm, shapiro
import scipy.stats as stats
import tifffile


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

if __name__ == "__main__":
    
   
    args.checkpoint = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/stage_1/6/mlt_epoch_15_best_RMSE_2.7529848_aux_0.000e+00.pth"
 
    # If your GPU is currently occupied, you could simply use the CPU to visuazlie your recent predictions.
    use_gpu = True
    if use_gpu:
        cuda_id = 0
        with torch.cuda.device(cuda_id):
            model  = AE(input_channels=args.input_channels, mode="stage_1")
            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            state = torch.load(args.checkpoint, map_location=device)
            load_checkpoint_with_shape_match(model, state['state_dict'])
            print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
            model.eval()
            model.to(device)
    else:
        model  = AE(input_channels=args.input_channels,  mode="stage_1")
        device = torch.device('cpu')
        state = torch.load(args.checkpoint, map_location=device)
        load_checkpoint_with_shape_match(model, state['state_dict'])
        print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
        model.eval()
        model.to(device)
    
    ############################# Save the latent space ############################# 
        
    
   
    print(f"{args.hashtags_prefix} Extracting latent space from Depth model")
    batch_size = 8
    dataloaders = make_dataloaders(batchsize=batch_size, split="train")
    trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
    test_dl = make_dataloaders(batchsize=batch_size, split="test")["test"]
    splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
    splits_and_loaders = [(k, v) for k, v in zip(list(splits.keys()), list(splits.values()))]
    split, loader = [(k, v) for k, v in zip(list(splits.keys()), list(splits.values()))][1]
    orig_em_path = Path("/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/stage_1") / "depth_anchors" # Where should it be created?
    # orig_em_path = Path(args.output_dir) / "latent_spaces" # Where should it be created?
    os.makedirs(orig_em_path, exist_ok=True)
        
    # loop = create_tqdm_bar(loader, split)
    with torch.no_grad():
        acc_tensor = None
        for split, loader in splits_and_loaders:
            loop = create_tqdm_bar(loader, split)
            for i, batch in loop:
                gt_x = batch["gt"]["depth"]["lidar_depth"].to(torch.float32).to(device)
                depth_gt = batch["gt"]["depth"]["lidar_depth"].to(torch.float32).to(device)
                x = batch["image"].to(torch.float32).to(device)
                names = [".".join(n.split(".")[:-1]) + ".npy" for n in batch["name"]]
                pred_dict = model(x, gt_x)

                pc = pred_dict["projection"]["latents"]["depth_latent_space"].detach().cpu()
                pred_depth = pred_dict["proc_decoder_output"]["depth"]["final_depth"]
                rmse = torch.sqrt(MaskedMSELoss()(pred_depth, depth_gt)) * args.max_depth
                
                
                b, c, h, w = pc.shape
                pc = pc.permute(0, 2, 3, 1).reshape(b, -1, c)
                
                if acc_tensor is None:
                    acc_tensor = pc
                else:
                    acc_tensor = torch.cat((acc_tensor, pc), 0)
                    
                if acc_tensor.shape[0] >= 1000:
                    curr_gt_tensor = acc_tensor.numpy()
                    tifffile.imwrite(orig_em_path / "depth_anchors.tiff", curr_gt_tensor)
                    exit()
                loop.set_postfix({"RMSE": rmse.item(), "shape": pc.shape})
                
                # orig_e = model.encoder(x)[-1].detach().cpu()
                # for e, name in zip(orig_e, names):
                #     np.save(os.path.join(orig_em_path, name), e)
                
                # pred_depth = pred_dict["depth"]["final_depth"].detach().cpu()
                
                    
        
        
        
        
        
        
        
        
        
        
       
        
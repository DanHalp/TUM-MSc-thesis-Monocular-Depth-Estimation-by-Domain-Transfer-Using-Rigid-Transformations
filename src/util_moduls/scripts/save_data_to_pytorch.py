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
import cv2
from data.dataloader_vae import make_dataloaders
from util_moduls.args import args
import numpy as np

from src.util_moduls.utils_functions import load_checkpoint_with_shape_match, create_tqdm_bar



import torchvision.transforms as T
from PIL import Image
import tifffile


if __name__ == "__main__":
    """
    Visualize the GT depth maps of the nuscenes dataset.
    
    """
   
    ##################### Load the model #####################

    args.num_workers = 20
    ##################### Data #####################
    dataloaders = make_dataloaders(batchsize=1, split="test")
    # dataloaders = make_dataloaders(batchsize=1, split="test")
    test_dl = dataloaders["test"]
    dataloaders = make_dataloaders(batchsize=1, split="train", shuffle_training=False)
    trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
    splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
    keys, values = list(splits.keys()), list(splits.values())
    
        
    
    ##################################################################################################################################################################
    split, loader = [(k, v) for k, v in zip(keys, values)][0] # [0] is the train split, [1] is the val split, [2] is the test split
    num_samples = 120  # How many samples would you like to visualize?
    ##################################################################################################################################################################


    all_losses = []
    mean_losses = []
    save_path = "/home/ubuntu/Datasets/thesis_data"
    # save_path = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/bulshit"
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for split, loader in  [(k, v) for k, v in zip(keys, values)]:
        loop = create_tqdm_bar(loader, desc=f"{args.hashtags_prefix} {split}") # tqdm is needed for visualization of the training progess in the terminal
        for i, batch in loop:
            name = batch["name"]
            gt_tensor = batch["gt"]["depth"]["lidar_depth"]
            sky_seg = batch["gt"]["seg"]["sky_seg"]
            original_image = batch["gt"]["images"]["original"]
            image = batch["image"]
            
            for idx in range(gt_tensor.shape[0]):
                curr_dir_name = Path(save_path) / name[idx].split(".")[0]
                os.makedirs(curr_dir_name, exist_ok=True)
                
                # Convert from tensor to png:
                curr_gt_tensor = gt_tensor[idx].numpy()
                tifffile.imwrite(curr_dir_name / "lidar_depth.tiff", curr_gt_tensor)
                
                # curr_sky_seg = sky_seg[idx].numpy()
                # tifffile.imwrite(curr_dir_name / "sky_seg.tiff", curr_sky_seg)
                
                # curr_original_image = original_image[idx].numpy()
                # tifffile.imwrite(curr_dir_name / "original_image.tiff", curr_original_image)
                
      
                
                curr_original_image = (original_image[idx].cpu().numpy().transpose(1, 2, 0)).astype(np.uint)
                cv2.imwrite(str(curr_dir_name / "orig.png"), curr_original_image)
                # # curr_original_image = T.ToPILImage()(original_image[idx])
                # # curr_original_image.save(curr_dir_name / "original_image.png")\
                
                curr_sky_seg = sky_seg[idx].cpu().numpy().squeeze(0).squeeze(0).astype(np.uint8)
                cv2.imwrite(str(curr_dir_name / "sky_seg.png"), curr_sky_seg)
                # # curr_sky_seg.save(curr_dir_name / "sky_seg.png")
                
                curr_name = name[idx]
                curr_name = curr_name.split(".")[0]
                np.save(curr_dir_name / "name.npy", curr_name)
                    
                    
            #     save_dict = {
            #         "gt": {
            #             "depth": {
            #                 "lidar_depth": gt_tensor[idx].cpu(),
            #                 },
            #             "seg": {
            #                 # "seg_gt": seg_gt[idx],
            #                 "sky_seg": sky_seg[idx].cpu(),
            #                 },
            #             "images": {
            #                 "original": original_image[idx].cpu(),
            #                 },           
            #         },
            #         "image": image[idx].cpu(),
            #         "name": name[idx]
            #     }
            # # np.save(os.path.join(save_path, name[idx].split(".")[0] + ".npy"), save_dict)
            #     torch.save(save_dict, os.path.join(save_path, name[idx].split(".")[0] + ".pth"))

                
                                
                
            
    
    
        
        
      



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

from src.util_moduls.utils_functions import load_checkpoint_with_shape_match
from matplotlib import pyplot as plt
from util_moduls.loss_funcs import MaskedMSELoss
from src.util_moduls.utils_functions import save_point_clouds_as_obj
from model.AE import AE
import torch.nn.functional as F
from src.util_moduls.utils_functions import procrustes_align

import numpy as np
from scipy.spatial.distance import cdist


def find_closest_pixels(feature_map):
    # Get the indices of pixels with values greater than 0
    nonzero_indices = np.argwhere(feature_map > 0)

    # Calculate pairwise distances between nonzero pixels
    distances = cdist(nonzero_indices, nonzero_indices)

    # Exclude self-distances (diagonal elements)
    np.fill_diagonal(distances, np.inf)

    # Find the indices of the 8 closest pixels for each nonzero pixel
    closest_indices = np.argsort(distances, axis=1)[:, :8]

    # Get the coordinates of the closest pixels
    closest_pixels = nonzero_indices[closest_indices]

    return closest_pixels


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

    # args.checkpoint = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Depth/43/mlt_epoch_15_best_RMSE_2.6101025.pth"
    # args.model = "Depth"
    # args.input_channels = 5

    # args.checkpoint = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Embedding/20/mlt_epoch_32_best_RMSE_3.1687825.pth"
    # args.model = "Embedding"
    # args.input_channels = 3
    

    # If your GPU is currently occupied, you could simply use the CPU to visuazlie your recent predictions.
    use_gpu = True
    if use_gpu:
        cuda_id = 0
        with torch.cuda.device(cuda_id):
            # model  = CamRaDepth(input_channels=args.input_channels)
            # model = CamRaBin(input_channels=args.input_channels)
            model = AE(input_channels=args.input_channels)

            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            state = torch.load(args.checkpoint, map_location=device)
            load_checkpoint_with_shape_match(model, state['state_dict'])
            print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
            model.eval()
            model.to(device)
    else:
        # model  = CamRaDepth(input_channels=args.input_channels)
        # model = CamRaBin(input_channels=args.input_channels)
        model = AE(input_channels=args.input_channels)
        device = torch.device('cpu')
        state = torch.load(args.checkpoint, map_location=device)
        load_checkpoint_with_shape_match(model, state['state_dict'])
        print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
        model.eval()
        model.to(device)
        
    ##################### Data #####################
    
    dataloaders = make_dataloaders(batchsize=1, split="test")
    # dataloaders = make_dataloaders(batchsize=1, split="test")
    test_dl = dataloaders["test"]
    dataloaders = make_dataloaders(batchsize=1, split="train", shuffle_training=False)
    trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
    splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
    keys, values = list(splits.keys()), list(splits.values())
    
    ##################### Create an output folder #####################
    path = Path(args.output_dir) / "3d_meshes" # Where should it be created?
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
        
    
    ##################################################################################################################################################################
    split, loader = [(k, v) for k, v in zip(keys, values)][2] # [0] is the train split, [1] is the val split, [2] is the test split
    num_samples = 120  # How many samples would you like to visualize?
    ##################################################################################################################################################################

                 
    print(f"{args.hashtags_prefix} Visualizing {num_samples} samples")
   
    
    
    visited = set()
    names = {"00453_gt"}

    RMSE_arr = []
    for k, batch in enumerate(loader):
        plt.close()
        fig, axs = plt.subplots(2, 3, figsize=(20, 20))
        name = batch["name"][0].split(".")[0]
        
        if k == num_samples:
            break

        if split == "test" and True:
            
            if name not in names:
                continue
            
            visited.add(name)
           
            print(f"{args.hashtags_prefix} Processing {name}")
            print(f"{args.hashtags_prefix} Proccessed {len(visited)}/{len(names)}")
            
                
    
        with torch.no_grad():
            # model.eval()
            x = batch["image"].to(torch.float32).to(device)
            pred_dict = model(x, gt_image=batch["gt"]["depth"]["cropped_lidar_depth_enhance"].to(torch.float32).cuda())
            
        
        
    
        ####################################################### Depth #######################################################
        # depth_pred = pred["depth"]["final_depth"]
        # depth_pred = pred_dict["depth"]["final_depth"] depth_pred = pred["depth"]["depth_maps"][0]
        b = x.size(0)
        rgbd_pc = pred_dict["projection"]["pcs"]["rgbd_pc"]
        save_point_clouds_as_obj(rgbd_pc,path, path / "rgbd_pc")
        
        proc_pc = pred_dict["projection"]["pcs"]["proc_pc"]
        save_point_clouds_as_obj(proc_pc,path, path / "proc_pc")
        
        origin = torch.zeros(1, 1, 3)
        save_point_clouds_as_obj(origin,path, path / "origin")

      

        # # Let's save now the first point cloud in the source and target batches:
        # save_point_clouds_as_obj(rgb_pc ,path,  path / "rgb_pc")
        
        #
        print("Finished")


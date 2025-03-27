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
from model.CamRaDepth import CamRaDepth
from model.CamRaBin import CamRaBin
from src.util_moduls.utils_functions import load_checkpoint_with_shape_match
from matplotlib import pyplot as plt
from util_moduls.loss_funcs import MaskedMSELoss
from src.util_moduls.utils_functions import save_point_clouds_as_obj
from model.AE import AE
import torch.nn.functional as F
from src.util_moduls.utils_functions import procrustes_align

import numpy as np
from scipy.spatial.distance import cdist




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
    
    if args.model == "Depth":
        dir_path = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Depth"
    elif args.model == "Projection":
        dir_path = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Projection"
    elif args.model == "Embedding":
        dir_path = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Embedding"
    file = get_latest_file_in_folder(dir_path)

    args.checkpoint = file
    
    # args.checkpoint = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Projection/4/mlt_epoch_3_iter_2100_intermediate_3.4014_r-loss_9.61e-01.pth"
    # args.checkpoint = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Projection/4/mlt_epoch_3_iter_2400_intermediate_7.0177_r-loss_9.57e-01.pth"
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
    path = Path(args.output_dir) / "2D_point_clouds" # Where should it be created?
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

        if split == "test" and False:
            
            if name not in names:
                continue
            
            visited.add(name)
           
            print(f"{args.hashtags_prefix} Processing {name}")
            print(f"{args.hashtags_prefix} Proccessed {len(visited)}/{len(names)}")

        with torch.no_grad():
            # model.eval()
            x = batch["image"].to(torch.float32).to(device)
            pred_dict = model(x, gt_image=batch["gt"]["depth"]["lidar_depth_enhance"].to(torch.float32).cuda())
            
        R = pred_dict["transformation"]["trans_dict"]["rotations"]["proc"]["R"]
        R_gt = pred_dict["transformation"]["trans_dict"]["rotations"]["proc"]["R_gt"]
        print(R_gt); exit()
        
    
        ####################################################### Depth #######################################################
        trans_dict = pred_dict["transformation"]["trans_dict"]
        proc_pc = trans_dict["proc_pc"]
        proc_pc = proc_pc[:, :, :2]
        proc_pc = proc_pc - proc_pc.mean(dim=1, keepdim=True)
        proc_pc = proc_pc / proc_pc.norm(dim=2, keepdim=True)
        proc_pc = proc_pc.cpu().numpy()
        
        target_pc = trans_dict["target_pc"]
        target_pc = target_pc[:, :, :2]
        target_pc = target_pc - target_pc.mean(dim=1, keepdim=True)
        target_pc = target_pc / target_pc.norm(dim=2, keepdim=True)
        target_pc = target_pc.cpu().numpy()
        
        source_pc = trans_dict["source_pc"]
        source_pc = source_pc[:, :, :2]
        source_pc = source_pc - source_pc.mean(dim=1, keepdim=True)
        source_pc = source_pc / source_pc.norm(dim=2, keepdim=True)
        source_pc = source_pc.cpu().numpy()
        num_points = proc_pc.shape[1]
        print(f"num_points: {num_points}")  
        for i in range(25):
            # print(trans_dict['R_proc'][i])
            # print(trans_dict['R_pred'][i])

            x1, y1 = proc_pc[i][:, 0][:num_points], proc_pc[i][:, 1][:num_points]
            x2, y2 = target_pc[i][:, 0][:num_points], target_pc[i][:, 1][:num_points]
            x3, y3 = source_pc[i][:, 0][:num_points], source_pc[i][:, 1][:num_points]
            
            print(trans_dict["R_proc"][i])

                
            # Create a scatter plot for each point cloud
            plt.figure(figsize=(8, 8))
            plt.scatter(x2, y2, color='red', label='target_pc', s=10)
            plt.scatter(x1, y1, color='blue', label='proc_pc', s=60)
            plt.scatter(x3, y3, color='green', label='source_pc', s=40)
            
            # abc = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"][:num_points]
            # for idx, n in enumerate(abc):
            #     plt.annotate(n, (x1[idx], y1[idx]), xytext=(0, 3), textcoords="offset points", ha="center", va="baseline")
            #     plt.annotate(n, (x2[idx], y2[idx]), xytext=(0, 3), textcoords="offset points", ha="center", va="baseline")
            #     plt.annotate(n, (x3[idx], y3[idx]), xytext=(0, 3), textcoords="offset points", ha="center", va="baseline")
            
            plt.title('Point Cloud Visualization')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(path / f"{name}_{i}_2D_point_cloud.png")
            plt.close()
            print(f"{args.hashtags_prefix} Saved 2D point cloud visualization to {path / f'{name}_{i}_2D_point_cloud.png'}")
        break
    
print(f"FInished Visualizing and validating {args.checkpoint}")


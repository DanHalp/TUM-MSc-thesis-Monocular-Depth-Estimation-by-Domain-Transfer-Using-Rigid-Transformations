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
from util_moduls.loss_funcs import MaskedMSELoss
from model.AE import AE

import numpy as np



if __name__ == "__main__":
    """
    Visualize the GT depth maps of the nuscenes dataset.
    
    """
   
    ##################### Load the model #####################
    
    def plot_loss_curves(train_loss_values, val_loss_values, save_path):
        """
        Plots the training and validation loss curves over epochs and saves the plot to the given path.
        
        :param train_loss_values: List or array of training loss values
        :param val_loss_values: List or array of validation loss values
        :param save_path: Path to save the plot (e.g., 'path/to/save/loss_curve.png')
        """
        epochs = range(1, len(train_loss_values) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss_values, label='Training Loss', color='red', marker='o')
        plt.plot(epochs, val_loss_values, label='Validation Loss', color='blue', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.grid(True)
        plt.legend()
        
        # Save the plot to the specified path
        plt.savefig(save_path)
        plt.close()
    
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
    
    ####################### If you desire a specic checkpoint #######################
    # args.checkpoint = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/stage_0/22/mlt_epoch_29_best_RMSE_2.7455144_aux_4.453e+00.pth"


    # If your GPU is currently occupied, you could simply use the CPU to visuazlie your recent predictions.
    use_gpu = True
    if use_gpu:
        cuda_id = 0
        with torch.cuda.device(cuda_id):
          
            model = AE(input_channels=args.input_channels,  mode="stage_0")

            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            state = torch.load(args.checkpoint, map_location=device)
            load_checkpoint_with_shape_match(model, state['state_dict'])
            print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
            model.eval()
            model.to(device)
    else:
       
        model = AE(input_channels=args.input_channels, mode="stage_0")
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
    path = Path(args.output_dir) / "visualzisations" # Where should it be created?
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
     # print(state.keys()); exit()
    train_losses_array = state["train_losses"][:, 0]
    val_losses_array = state["val_losses"][:, 1]
    print(state["train_losses"][:, 0], state["train_losses"][:, 1])
    plot_loss_curves(state["train_losses"][:, 0], state["train_losses"][:, 1], path / "loss_curves.png")
        
    
    ##################################################################################################################################################################
    split, loader = [(k, v) for k, v in zip(keys, values)][2] # [0] is the train split, [1] is the val split, [2] is the test split
    num_samples = 120  # How many samples would you like to visualize?
    ##################################################################################################################################################################

    print(f"{args.hashtags_prefix} Visualizing {num_samples} samples")
    cur_split_path = path 
    orig_path = cur_split_path / "orig"
    depth_path = cur_split_path / "depth"
    seg_path = cur_split_path / "seg"
    gt_path = cur_split_path / "gt"
    radar_on_rgb_path = cur_split_path / "radar_on_rgb"
    radar_path = cur_split_path / "radar"
    seg_pred_path = cur_split_path / "seg_pred"
    collage_path = cur_split_path / "collage"
    depth_pred_path = cur_split_path / "depth_pred"
    
    # os.makedirs(radar_path, exist_ok=True)
    os.makedirs(collage_path, exist_ok=True)
    
    visited = set()
    names = {"00453_gt", "06209_gt", "06279_gt", "09648_gt", "12092_gt",  "12348_gt", "15409_rain_gt", "16140_gt", "16474_gt", "16734_gt", "16927_gt", "18630_rain_gt",
             "20387_gt", "21150_gt", "23802_gt", "24805_gt", "19732_rain_gt"}

    RMSE_arr = []
    to_break = False

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
            if len(visited) == len(names):
                to_break = True
                
        curr_img_path = cur_split_path / name
        os.makedirs(curr_img_path, exist_ok=True)
        img = batch["image"][:, :3]
        
                    
        # Orig image
        
        orig_img = batch["gt"]["images"]["original"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        orig_img = orig_img.astype(np.uint8)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(str(curr_img_path / "orig.png"), orig_img)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        
        # Lidar Groundtruth:
        lidar_gt = batch["gt"]["depth"]["lidar_depth"].cpu().numpy().squeeze()    
        plt.imsave(str(curr_img_path / "lidar_gt.png"), lidar_gt, cmap="jet")

        # Define the grid parameters
        gt_img_with_lidar =  orig_img.copy() / 255
        gt_colour = np.array(cv2.imread(str(curr_img_path / "lidar_gt.png")))
        gt_colour = cv2.cvtColor(gt_colour, cv2.COLOR_BGR2RGB) / 255
        gt_colour = gt_colour.astype(np.float32)
        gt_img_with_lidar[lidar_gt > 0] = gt_colour[lidar_gt > 0]
        plt.imsave(str(curr_img_path / "lidar_gt_on_rgb.png"), gt_img_with_lidar)
        
        import torch.nn.functional as F
        with torch.no_grad():
            with torch.autocast('cuda'):
                # model.train()
                x = batch["image"].to(torch.float32).to(device)
                gt_x = batch["gt"]["depth"]["augmented_depth"].to(torch.float32).to(device)
                pred = model(x, gt_x)

        b, c, h, w = x.shape
        depth_pred = pred["proc_decoder_output"]["depth"]["final_depth"]
        proj_pred = pred["proj_decoder_output"]["depth"]["final_depth"]
        rgb_pred = pred["rgb_decoder_output"]["depth"]["final_depth"]
        plt.imsave(str(curr_img_path / "rgbd_pred.png"),  depth_pred.detach().cpu().numpy().squeeze(), cmap="jet")

        
        rgb_latent = pred["projection"]["latents"]["rgb"]
        depth_latent = pred["projection"]["latents"]["depth"]


        RMSE = (torch.sqrt(MaskedMSELoss()(proj_pred, batch["gt"]["depth"]["lidar_depth"].to(device))) * args.max_depth).item()
        RMSE_arr.append(RMSE)
        print(f"RMSE LOSS for {name}: ", RMSE)
   
        gt_sky_seg = batch["gt"]["seg"]["sky_seg"].squeeze()
        
        indices = torch.logical_and(gt_sky_seg == 0, depth_pred.squeeze().cpu() < args.false_sky_thresh)
        sky_error = torch.zeros_like(gt_sky_seg)
        sky_error[indices] = 1
        sky_error = sky_error.cpu().numpy()
        

        # Transperent colormap on the rgb.
        depth_color = cv2.imread(str(curr_img_path / "rgbd_pred.png"))
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
  
  
        # Blend the colorized depth map with the RGB image
        # blended = orig_img * 0.8 + depth_color * 0.75
        # blended = blended / blended.max()
        blended = cv2.addWeighted(orig_img.astype(depth_color.dtype), 0.8, depth_color, 0.75, 0)
        blended = blended / blended.max()
        plt.imsave(str(curr_img_path / "depth_on_rgb.png"), blended)
        
        
        proj_pred =  pred["proj_decoder_output"]["depth"]["final_depth"].cpu().numpy().squeeze()
        rgb_pred = pred["rgb_decoder_output"]["depth"]["final_depth"].cpu().numpy().squeeze()
        depth_pred = pred["proc_decoder_output"]["depth"]["final_depth"].cpu().numpy().squeeze()
        plt.imsave(str(curr_img_path / "rgb_pred.png"),  proj_pred, cmap="jet")

            
        axs[0, 0].imshow(orig_img, cmap="jet")
        axs[0, 1].imshow(rgb_pred, cmap="jet")
        axs[0, 2].imshow(proj_pred, cmap="jet")
        axs[1, 0].imshow(depth_pred, cmap="jet")
        axs[1, 1].imshow(gt_img_with_lidar, cmap="jet")
        
        diff = np.zeros_like(depth_pred)
        diff[lidar_gt > 0] = np.abs(depth_pred[lidar_gt > 0] - lidar_gt[lidar_gt > 0])
        med = np.median(diff[lidar_gt > 0])
        mean = np.mean(diff[lidar_gt > 0])
        diff[diff <= med] = diff[diff <= med] ** 2
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        axs[1, 2].imshow(diff, cmap="jet")
    

        # Save the figure
        plt.savefig(str(collage_path / f"{split}_{batch['name'][0]}"))
        plt.close()
        
        print(f"{args.hashtags_prefix} Processed {name} -- {k + 1}/{num_samples} samples")
        if k + 1 == num_samples or to_break:
            print(f"{args.hashtags_prefix} Finished processing {num_samples} samples")
            break

    print(f"FInished Visualizing and validating {args.checkpoint}")
    print("RMSE: ", np.mean(RMSE_arr))
        
      



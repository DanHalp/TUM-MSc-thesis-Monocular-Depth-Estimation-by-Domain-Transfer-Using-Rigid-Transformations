import os, sys

import shutil
from pathlib import Path
from tqdm import tqdm
path = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/halperin-dan"
os.chdir(path + "/src") 
sys.path.append(os.getcwd())
# sys.path.append(os.getcwd() + "/Transformer/scripts/project_files/src")


import torch
import torch.functional as F
import torch.nn as nn
import cv2
from matplotlib import pyplot as plt
import numpy as np

from data.dataloader_vae import make_dataloaders
from util_moduls.args import args

from src.util_moduls.utils_functions import load_checkpoint_with_shape_match, SphericalGrid, create_tqdm_bar
from util_moduls.loss_funcs import MaskedMSELoss
from model.AE import AE

import time
import numpy as np
os.chdir("/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/halperin-dan") 
args.checkpoint = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/halperin-dan/mlt_epoch_35_best_RMSE_4.2187502_aux_3.545e+00.pth"

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

def plot_loss_curves(values_arrays, desc_arrays, save_path):
    """
    Plots the training and validation loss curves over epochs and saves the plot to the given path.
    
    :param train_loss_values: List or array of training loss values
    :param val_loss_values: List or array of validation loss values
    :param save_path: Path to save the plot (e.g., 'path/to/save/loss_curve.png')
    """
    plt.figure(figsize=(10, 6))
    for arr, desc in zip(values_arrays, desc_arrays):
        epochs  = np.arange(len(arr))

        plt.plot(epochs, arr, label=desc, marker=next(iter(["o", "s", "X", "^", "D", "v", "p", "H", "<", ">"])) )
    
    
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    # plt.title('Training and Validation Loss Curves')
    plt.grid(True)
    plt.legend(prop={'size': 27})
    
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    
    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()

def eval(model, test_dl):
    model.eval()
    all_losses = []
    mean_losses = []
    times = []
    
    with torch.no_grad():
        loop_eval = create_tqdm_bar(test_dl, f"Test") # tqdm is needed for visualization of the training progess in the terminal
        
        for i, batch in loop_eval:
            gt = batch["gt"]
            gt_depth_final = gt["depth"]['lidar_depth'].to(torch.float32).cuda()
            x = batch["image"].to(torch.float32).cuda()
            gt_x = gt["depth"]["augmented_depth"].to(torch.float32).cuda()

            with torch.autocast("cuda"):
                start = time.time()
                pred_dict = model(x, gt_image=gt_x)
                end = time.time()
                proj_pred_depth  = pred_dict["proj_decoder_output"]["depth"]["final_depth"]
                proj_rmse = torch.sqrt(MaskedMSELoss()(proj_pred_depth, gt_depth_final)).item() * args.max_depth
                
                rgb_pred_depth = pred_dict["rgb_decoder_output"]["depth"]["final_depth"]
                rgb_rmse = torch.sqrt(MaskedMSELoss()(rgb_pred_depth, gt_depth_final)).item() * args.max_depth
                
                zero_shot_depth = pred_dict["proc_decoder_output"]["depth"]["final_depth"]
                zero_shot_rmse = torch.sqrt(MaskedMSELoss()(zero_shot_depth, gt_depth_final)).item() * args.max_depth
  
            all_losses.append([
                rgb_rmse,
                proj_rmse,
                zero_shot_rmse,
                round(end - start, 2)
            ])

            
            progess_bool = (i + 1) % args.update_interval == 0 or (i + 1) == len(test_dl)
            if progess_bool:
                
                mean_losses = np.nanmean(all_losses, axis=0)
                
                loop_eval.set_postfix(
                    v_RMSE_proj= mean_losses[0],
                    v_RMSE_depth = "{:.6f}".format(mean_losses[1]),
                    v_RMSE_zero_shot = "{:.6f}".format(mean_losses[2]),
                    v_time = "{:.6f} seconds".format(mean_losses[-1]),
                    
                )
        
    print(f"rgb_rmse: {mean_losses[0]}, proj_rmse: {mean_losses[1]}, zero_shot_rmse: {mean_losses[2]}, time: {mean_losses[-1]}")
    
def visualize(model, test_dl):
    model.eval()
    split = "test"
    args.output_dir = os.getcwd()
    path = Path(args.output_dir) / "visualzisations" # Where should it be created?
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    collage_path = path / "collage"
    os.makedirs(collage_path, exist_ok=True)
    
    visited = set()
    names = {"00453_gt", "06209_gt", "06279_gt", "09648_gt", "12092_gt",  "12348_gt", "15409_rain_gt", "16140_gt", "16474_gt", "16734_gt", "16927_gt", "18630_rain_gt",
             "20387_gt", "21150_gt", "23802_gt", "24805_gt", "19732_rain_gt"}
    
    RMSE_arr = []
    to_break = False
    num_samples = 120
    for k, batch in enumerate(test_dl):
        plt.close()
        fig, axs = plt.subplots(2, 3, figsize=(20, 20))
        name = batch["name"][0].split(".")[0]
        
        if k == num_samples:
            break

        if name not in names:
            continue
        
        visited.add(name)
        
    
        if len(visited) == len(names):
            to_break = True
                
        curr_img_path = path / name
        os.makedirs(curr_img_path, exist_ok=True)
        img = batch["image"][:, :3]
               
        # Orig image
        
        orig_img = batch["gt"]["images"]["original"].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        orig_img = orig_img.astype(np.uint8)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite(str(curr_img_path / "orig.png"), orig_img)


        # Lidar Groundtruth:
        lidar_gt = batch["gt"]["depth"]["lidar_depth"].cpu().numpy().squeeze()
        
        augmented_depth_noise = batch["gt"]["depth"]["augmented_depth"].cpu().numpy().squeeze()[0]
        plt.imsave(str(curr_img_path / "lidar_gt.png"), lidar_gt, cmap="jet")
        plt.imsave(str(curr_img_path / "augmented_depth_noise.png"), augmented_depth_noise, cmap="jet")

        buf = 100
        # lidar groundtruth on rgb
        gt_img_with_lidar =  orig_img.copy() / 255
        gt_colour = np.array(cv2.imread(str(curr_img_path / "lidar_gt.png")))
        gt_colour = cv2.cvtColor(gt_colour, cv2.COLOR_BGR2RGB) / 255
        gt_colour = gt_colour.astype(np.float32)
        gt_img_with_lidar[lidar_gt > 0] = gt_colour[lidar_gt > 0]
        
        h, w = augmented_depth_noise.shape[:2]
        gt_img_with_lidar = gt_img_with_lidar[buf:h-buf, buf:w-buf]
        plt.imsave(str(curr_img_path / "lidar_gt_on_rgb.png"), gt_img_with_lidar)
        
        # augmented noise depth on rgb
        gt_img_with_lidar =  orig_img.copy() / 255
        gt_colour = np.array(cv2.imread(str(curr_img_path / "augmented_depth_noise.png")))
        gt_colour = cv2.cvtColor(gt_colour, cv2.COLOR_BGR2RGB) / 255
        gt_colour = gt_colour.astype(np.float32)
        gt_img_with_lidar[augmented_depth_noise > 0] = gt_colour[augmented_depth_noise > 0]
        
        h, w = augmented_depth_noise.shape[:2]
        gt_img_with_lidar = gt_img_with_lidar[buf:h-buf, buf:w-buf]
        plt.imsave(str(curr_img_path / "augmented_depth_noise_on_rgb.png"), gt_img_with_lidar)
        
      
  
        import torch.nn.functional as F
        with torch.no_grad():
            # model.train()
            x = batch["image"].to(torch.float32).to(device)
            gt_x = batch["gt"]["depth"]["augmented_depth"].to(torch.float32).to(device)
            pred = model(x, gt_x)

        b, c, h, w = x.shape
        proj_pred = pred["proj_decoder_output"]["depth"]["final_depth"]
        rgb_pred = pred["rgb_decoder_output"]["depth"]["final_depth"].cpu().numpy().squeeze()
        zero_pred = pred["proc_decoder_output"]["depth"]["final_depth"].squeeze().cpu().numpy()

        pred_name = name.split("_")[0]
        plt.imsave(str(curr_img_path / (pred_name + "-depth.png")),  proj_pred.detach().cpu().numpy().squeeze(), cmap="jet")
   
   
        gt_sky_seg = batch["gt"]["seg"]["sky_seg"].squeeze()
        indices = torch.logical_and(gt_sky_seg == 0, proj_pred.squeeze().cpu() < args.false_sky_thresh)
        sky_error = torch.zeros_like(gt_sky_seg)
        sky_error[indices] = 1
        sky_error = sky_error.cpu().numpy()
        

        # Transperent colormap on the rgb.
        depth_color = cv2.imread(str(curr_img_path / (pred_name + "-depth.png")))
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
  
      
        blended = cv2.addWeighted(orig_img.astype(depth_color.dtype), 0.8, depth_color, 0.75, 0)
        blended = blended / blended.max()
        plt.imsave(str(curr_img_path / "depth_on_rgb.png"), blended)
        
        plt.imsave(str(curr_img_path / (pred_name + ".png")),  rgb_pred, cmap="jet")
        proj_pred = pred["proj_decoder_output"]["depth"]["final_depth"].cpu().numpy().squeeze()
        rgb_pred = pred["rgb_decoder_output"]["depth"]["final_depth"].cpu().numpy().squeeze()
        zero_pred = pred["proc_decoder_output"]["depth"]["final_depth"].squeeze().cpu().numpy()

            
        axs[0, 0].imshow(orig_img, cmap="jet")
        axs[0, 1].imshow(rgb_pred, cmap="jet")
        axs[0, 2].imshow(zero_pred, cmap="jet")
        axs[1, 0].imshow(proj_pred, cmap="jet")
        axs[1, 1].imshow(gt_img_with_lidar, cmap="jet")
        
    
        diff = np.abs(rgb_pred- proj_pred)
        med = np.median(diff)
        mean = np.mean(diff)
        diff[diff <= med] = diff[diff <= med] ** 2
        # rescale diff to [0, 1]
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        axs[1, 2].imshow(diff, cmap="jet")
    

        # Save the figure
        plt.savefig(str(collage_path / f"{split}_{batch['name'][0]}"))
        
      
        plt.close()
        
        print(f"{args.hashtags_prefix} Processed {name} -- {k + 1}/{num_samples} samples")
        if k + 1 == num_samples or to_break:
            print(f"{args.hashtags_prefix} Finished processing {num_samples} samples")
            break
        # break

    print(f"FInished Visualizing and validating {args.checkpoint}")
    print("RMSE: ", np.mean(RMSE_arr))
    

def print_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_gb = total_params * 4 / (1024 ** 3)
    print(f"Total Parameters: {total_params_gb:,} ")


if __name__ == "__main__":
    model = AE(input_channels=args.input_channels, mode=args.model)
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    state = torch.load(args.checkpoint, map_location=device)
    load_checkpoint_with_shape_match(model, state["state_dict"])
    # args.learning_rate = state.get("lr", args.learning_rate)
    train_losses_list = state.get("train_losses", [])
    val_losses_list = state.get("val_losses", [])
    model = model.to(device).eval()
    dataloaders = make_dataloaders("test")
    test_dl = dataloaders["test"]

    ############################### 1. Num_params ###############################

    depth_rmse_array_train = train_losses_list[:, 1]
    depth_rmse_array_val = val_losses_list[:, 1]
    rgb_rmse_array_train = train_losses_list[:, 0]
    rgb_rmse_array_val = val_losses_list[:, 0]
    zero_rmse_array_train = train_losses_list[:, 2]
    zero_rmse_array_val = val_losses_list[:, 2]
    
    plot_loss_curves([rgb_rmse_array_val, depth_rmse_array_val, zero_rmse_array_val], ["RGB validation loss", "Depth validation loss", "zero-shot validation loss"], os.getcwd() + "/Validation losses.png")
    plot_loss_curves([rgb_rmse_array_train, depth_rmse_array_train], ["RGB training loss", "Depth training loss"], os.getcwd() + "/Training losses.png")
    # exit()
    print()
    print()
    print()
    
    eval(model, test_dl)
    
    print()
    print()
    print()
    
    visualize(model, test_dl)
    
    
    
    
    
    
        

    
    
    


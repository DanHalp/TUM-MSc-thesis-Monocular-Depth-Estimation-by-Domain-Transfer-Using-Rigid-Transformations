

from collections import namedtuple
import os, sys
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))



from tqdm import tqdm
import torch
from data.dataloader_vae import make_dataloaders
# from data.dataloader_v2 import make_dataloaders, load_prepared_file_list
from util_moduls.args import args
import numpy as np
from model.AE import VAE
from src.util_moduls.utils_functions import load_checkpoint_with_shape_match, NonZeroAvgPool2d


import numpy as np

def delete_dir(path):
    os.system("sudo rm -r %s" % path)

if __name__ == "__main__":
    """
    Visualize the GT depth maps of the nuscenes dataset.
    """
    ##################### Data #####################
    num_samples = 5 # How many samples would you like to visualize?
    dataloaders = make_dataloaders(batchsize=1, split="test")
    # dataloaders = make_dataloaders(batchsize=1, split="test")
    test_dl = dataloaders["test"]
    dataloaders = make_dataloaders(batchsize=1, split="train")
    trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
    splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
    keys, values = list(splits.keys())[::-1], list(splits.values())[::-1]
    
    ##################### Create an output folder #####################
    path = Path(args.output_dir) / "quasi_lidar_gt" # Where should it be created?
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    
    ##################### Load the model #####################
    
    # If your GPU is currently occupied, you could simply use the CPU to visuazlie your recent predictions.
    args.checkpoint = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/checkpoints/Thesis/AE/Best_AE/mlt_epoch_23_best_eval_loss_0.0003638.pth"
    use_gpu = True
    if use_gpu:
        cuda_id = 0
        with torch.cuda.device(cuda_id):
            model  = VAE(input_channels=args.input_channels)
            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            state = torch.load(args.checkpoint, map_location=device)
            load_checkpoint_with_shape_match(model, state['state_dict'])
            print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
            model.eval()
            model.to(device)
    else:
        model  = VAE(input_channels=args.input_channels)
        device = torch.device('cpu')
        state = torch.load(args.checkpoint, map_location=device)
        load_checkpoint_with_shape_match(model, state['state_dict'])
        print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
        model.eval()
        model.to(device)
    
    
    ##################### Visualize #####################
    print(f"{args.hashtags_prefix} Visualizing {num_samples} samples")
    for j, curr_dataloader in enumerate(values):
        
        loop = tqdm(enumerate(curr_dataloader), total=len(curr_dataloader), desc=f"Processing {keys[j]}")
        for i, batch in loop:
            name = batch["name"][0].split(".")[0].split("_")[0]
            curr_img_path = path / (str(name) + "_quasi_liadr.npy")
        
            # pred_Seg
            with torch.no_grad():
                pred = model(batch["image"].to(torch.float32).to(device)[:, :args.input_channels])

            # TODO: Depth prediction
            depth_pred = pred["depth"]["final_depth"]
            depth_pred = depth_pred.detach().cpu().numpy().squeeze()
            np.save(str(curr_img_path),  depth_pred)


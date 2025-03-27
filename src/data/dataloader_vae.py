import os
import sys
import time
from pathlib import Path


sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util_moduls.args import args
import torch.nn.functional as F
# from utils.ResizeRight.interp_methods import *
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from skimage import io

def add_noise(data, noise_std=0.4):
    
    noise = torch.randn_like(data) * noise_std
    noisy_data = data + noise
    return torch.clamp(noisy_data, 0., 1.) 

def make_dataloaders(split="train", train_part=-1, num_samples=-1, batchsize=-1, data_type="cross_val", shuffle_training=True, num_workers=args.num_workers):
    
    """
    Creates train, val, and test dataloaders according to the split and options.
    
    Args:
    split (str): Either "train" or "test".
    train_part (float): The proportion of the dataset to be used for training.
    num_samples (int): The number of samples to use from the dataset.
    batchsize (int): The batch size for the dataloaders.
    data_type (str): The type of dataset to use. Currently only "cross_val" is supported.
    shuffle_training (bool): Whether to shuffle the training dataset.
    num_workers (int): The number of workers to use for the dataloaders.
    
    Returns:
    A dictionary with the keys "train", "val", or "test", each containing the respective dataloader.
    """
    batchsize = args.batch_size if batchsize < 1 else batchsize
    
    args.num_samples = num_samples if num_samples > 0 else args.num_samples
    files_list = load_prepared_file_list(args.split)
    train_dl, val_dl, test_dl = None, None, None
    train_ds, val_ds = None, None
    
    if split == "train":
        files_list = files_list[0: args.num_samples]
        fractions = [int(round(train_part * args.num_samples)), int(round((1 - train_part) * args.num_samples))]  if 0 < train_part < 1 else args.train_val_split
        train_ds, val_ds = NuscenesDataset(list_with_all_files=files_list[0: fractions[0]]), NuscenesDataset(list_with_all_files=files_list[fractions[0]: fractions[0] + fractions[1]], mode="test")
        train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=shuffle_training, num_workers=num_workers, pin_memory=True )
        val_dl = DataLoader(val_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers, pin_memory=True)
        
    elif split == "test":
        
        index = 0 if args.mini_dataset else sum(args.train_val_split)
        files_list = files_list[index:]
        dataset = NuscenesDataset(list_with_all_files=files_list, mode="test")
        test_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)
        
    else:
        raise ValueError("Split must be either train or test")
    return {"train": train_dl, "val": val_dl, "test": test_dl}

def load_prepared_file_list(path):
    """
    If the data is already organized in bin files, we could simply load the the prepared data, instead of
    reorganizing each time, for a significant speed up (0.1 seconds vs 29 seconds)
    """
   
    path = Path(os.path.abspath(path))
    files = np.load(path, allow_pickle=True)
    print(f"{args.hashtags_prefix} Loaded split from {path}")
    return files
   

class NuscenesDataset(Dataset):
    def __init__(self, list_with_all_files, mode='train'):

        assert mode in ['train', 'test']
        self.list_with_all_files = np.array(list_with_all_files)
        self.mode = mode
        self.bins = [torch.linspace(0, 1, N + 1) for N in args.num_bins[::-1]]
        self.spatial_sizes = [np.array(args.image_dimension) // 2 ** (i) for i in range(3)]
        

       
    def __getitem__(self, index):
        """

        Returns: input tensor: [:, :3] - img
                               [:, 3:4] - radar
                               [:, 4:6] - radar flow (uv)
                               [:, 6:7] - radar velocity
       
        """
        
        name = self.list_with_all_files[index][6].split('/')[-1].split('.')[0]
        path = Path(args.dataset_path) / (name)
        
        gt_tensor = io.imread(str(path / "lidar_depth.tiff"))
        gt_tensor = torch.from_numpy(gt_tensor)
        
        # Add Gaussian noise and remove part of the LiDAR point clouds:
        lidar_clone = gt_tensor.clone()
        augmented_depth = torch.zeros_like(lidar_clone)
        non_zero_indices = lidar_clone != 0
        num_non_zero = non_zero_indices.sum().item()
        num_to_zero = int(num_non_zero * 1) 
        indices = torch.randperm(num_non_zero)[:num_to_zero]
        flat_tensor = augmented_depth.view(-1)
        non_zero_flat_indices = non_zero_indices.view(-1).nonzero(as_tuple=False).squeeze()
        flat_tensor[non_zero_flat_indices[indices]] = lidar_clone.view(-1)[non_zero_flat_indices[indices]]
        
        flat_tensor_noise = flat_tensor.clone()
        flat_tensor_noise[non_zero_flat_indices[indices]] = add_noise(flat_tensor[non_zero_flat_indices[indices]], noise_std=0.0)
        augmented_depth_orig = flat_tensor.view(augmented_depth.shape)
        augmented_depth = flat_tensor_noise.view(augmented_depth.shape)
        augmented_depth = augmented_depth.repeat(3, 1, 1)

        sky_seg = Image.open(str(path / "sky_seg.png"))
        sky_seg = torchvision.transforms.PILToTensor()(sky_seg).unsqueeze(0)
        sky_seg =  sky_seg *  (1 / 255)
        
        original_image = Image.open(str(path / "orig.png"))
        original_image = torchvision.transforms.PILToTensor()(original_image)
        image_tensor = original_image * (1 / 255)

        gt = {"depth": {'lidar_depth': gt_tensor, "augmented_depth": augmented_depth, "aug_depth_orig": augmented_depth_orig},
              "seg":{"sky_seg": sky_seg},
            "images": {"original": original_image, },
            }
        
        return {"image": image_tensor, "gt": gt, "name": name}
       
    def __len__(self):
        return len(self.list_with_all_files)
    
    


from tqdm import tqdm
def run():
    args.num_workers = 16
    dataloaders = make_dataloaders(batchsize=16, split="train")
    train_dl = dataloaders["train"]

        

if __name__ == "__main__":
    
    run()

    
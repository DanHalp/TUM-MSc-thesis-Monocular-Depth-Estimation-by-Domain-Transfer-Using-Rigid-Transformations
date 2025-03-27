import fileinput
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
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from util_moduls.args import args
from skimage.transform import resize
import pickle
import torchvision.transforms as T
import torch.nn.functional as F

import cv2


def make_dataloaders(split="train", train_part=-1, num_samples=-1, batchsize=-1, data_type="cross_val", shuffle_training=True):
    
    """
    Create train and val dataloaders out of the split (Given by nuscenes)
    
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
        train_dl = DataLoader(train_ds, batch_size=batchsize, shuffle=shuffle_training, num_workers=args.num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    elif split == "test":
        
        index = 0 if args.mini_dataset else sum(args.train_val_split)
        files_list = files_list[index:]
        dataset = NuscenesDataset(list_with_all_files=files_list, mode="test")
        test_dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        
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
   

def save_list(img_filename_list, radar_filename_list, filt_radar_filename_list, mseg_filename_list,
                im_uv_filename_list, rad_vel_filename_list, gt_filename_list, file_name, split="train"):
    
    # print(len(img_filename_list))
    # print(len(gt_filename_list))
    # print(len(radar_filename_list))
    # print(len(filt_radar_filename_list))
    # print(len(mseg_filename_list))
    # print(len(im_uv_filename_list))
    # print(len(rad_vel_filename_list))
    
    des_length = len(img_filename_list)
    assert len(gt_filename_list) >= des_length and len(radar_filename_list) >= des_length and len(filt_radar_filename_list) >= des_length  \
        and len(mseg_filename_list) >= des_length and len(im_uv_filename_list) >= des_length and len(rad_vel_filename_list) >= des_length

    if len(img_filename_list) == 0 or len(gt_filename_list) == 0:
        print("List(s) empty!")
        print("Number of images: {} \nNumber of ground truths: {} \n".format(len(img_filename_list), len(gt_filename_list)))
    else:
        if len(img_filename_list) == len(gt_filename_list):
            list_with_all_files = list(zip( img_filename_list,
                                            radar_filename_list,
                                            filt_radar_filename_list,
                                            mseg_filename_list,
                                            im_uv_filename_list,
                                            rad_vel_filename_list,
                                            gt_filename_list))
            
          
            path = Path("data") / (split + "_files")
            os.makedirs(path, exist_ok=True)
            path = path / file_name
            np.save(path, list_with_all_files)

        else:
            print("Number of images {} does not match number of ground truths {}!".format(len(img_filename_list), len(gt_filename_list)))

def create_new_split_file(current_split_path, new_dir_data, new_split_path):
    
    if not new_split_path.endswith(".npy"):
        new_split_path += ".npy"

    old_files = load_prepared_file_list(current_split_path)
    new_files = []
    for i, file in enumerate(old_files):
        new_dir_data = Path(new_dir_data)
        new_instance = [str(new_dir_data / file[j].split("/")[-1]) for j in range(len(file))]
        new_files.append(new_instance)
    
    # Check that the split is the same, just with a different path:
    # assert len(new_files) == len(old_files)
    # for i in range(len(old_files)):
    #     print("########################################")
    #     for j in range(len(old_files[0])):
    #         print(old_files[i][j], new_files[i][j])
    #     print("########################################")
    
    path = Path(new_dir_data)
    os.makedirs(path, exist_ok=True)
    print(new_split_path)
    np.save(new_split_path, new_files)
    print(load_prepared_file_list(str(new_split_path)))
    
    


def create_file_list(dir_data):
    """
    Organize the raw data as needed, and save the created list to dist
    so later on we could simply load the the prepared data, instead of
    reorganizing each time, for a significant speed up (0.1 seconds vs 29 seconds)
    """
    start = time.time()
    img_filename_list = glob.glob(dir_data + '*_im.jpg')
    img_filename_list.sort()
    radar_filename_list = glob.glob(dir_data + '*_radar.npy')
    radar_filename_list.sort()
    filt_radar_filename_list = glob.glob(dir_data + '*_radar_filtered.npy')
    filt_radar_filename_list.sort()
    mseg_filename_list = glob.glob(dir_data + '*_mseg.npy')
    mseg_filename_list.sort()
    im_uv_filename_list = glob.glob(dir_data + '*_im_uv.npy')
    im_uv_filename_list.sort()
    rad_vel_filename_list = glob.glob(dir_data + '*_rad_vel.npy')
    rad_vel_filename_list.sort()
    gt_filename_list = glob.glob(dir_data + '*_gt.npy')
    gt_filename_list.sort()
    save_list(img_filename_list, radar_filename_list, filt_radar_filename_list, mseg_filename_list,
                im_uv_filename_list, rad_vel_filename_list, gt_filename_list, "with_rain")
    end = time.time()
    print("with_rain: ", end - start, " seconds")
    
    # start = time.time()
    # img_filename_list = glob.glob(dir_data + '*_rain_im.jpg')
    # img_filename_list.sort()
    # radar_filename_list = glob.glob(dir_data + '*_rain_radar.npy')
    # radar_filename_list.sort()
    # filt_radar_filename_list = glob.glob(dir_data + '*_rain_radar_filtered.npy')
    # filt_radar_filename_list.sort()
    # mseg_filename_list = glob.glob(dir_data + '*_rain_mseg.npy')
    # mseg_filename_list.sort()
    # im_uv_filename_list = glob.glob(dir_data + '*_rain_im_uv.npy')
    # im_uv_filename_list.sort()
    # rad_vel_filename_list = glob.glob(dir_data + '*_rain_rad_vel.npy')
    # rad_vel_filename_list.sort()
    # gt_filename_list = glob.glob(dir_data + '*_rain_gt.npy')
    # gt_filename_list.sort()
    # save_list(img_filename_list, radar_filename_list, filt_radar_filename_list, mseg_filename_list,
    #             im_uv_filename_list, rad_vel_filename_list, gt_filename_list, "only_rain")
    # end = time.time()
    # print("only_rain: ", end - start, " seconds")


    # start = time.time()
    # img_filename_list = glob.glob(dir_data + '*[!rain]_im.jpg')
    # img_filename_list.sort()
    # radar_filename_list = glob.glob(dir_data + '*[!rain]_radar.npy')
    # radar_filename_list.sort()
    # filt_radar_filename_list = glob.glob(dir_data + '*[!rain]_radar_filtered.npy')
    # filt_radar_filename_list.sort()
    # mseg_filename_list = glob.glob(dir_data + '*[!rain]_mseg.npy')
    # mseg_filename_list.sort()
    # im_uv_filename_list = glob.glob(dir_data + '*[!rain]_im_uv.npy')
    # im_uv_filename_list.sort()
    # rad_vel_filename_list = glob.glob(dir_data + '*[!rain]_rad_vel.npy')
    # rad_vel_filename_list.sort()
    # gt_filename_list = glob.glob(dir_data + '*[!rain]_gt.npy')
    # gt_filename_list.sort()
    # save_list(img_filename_list, radar_filename_list, filt_radar_filename_list, mseg_filename_list,
    #             im_uv_filename_list, rad_vel_filename_list, gt_filename_list, "without_rain")
    # end = time.time()
    # print("without_rain: ", end - start, " seconds")
    exit()

   
class NuscenesDataset(Dataset):
    def __init__(self, list_with_all_files, mode='train'):

        assert mode in ['train', 'test']
        self.list_with_all_files = list_with_all_files
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
        # 
        
        def minpool(tensor):
            x = tensor.clone()
            # change zero value to high number (higher than max depth) such that it is ignored during min_pool
            x[tensor==0] = 255
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # min_pool
            x = -maxpool(-x)
            # rechange 255 to zero
            x[x==255] = 0
            return x

        name = self.list_with_all_files[index][6].split('/')[-1].split('.')[0] + ".png"
        # Standardize -> rescale data to a normal distribution with mean = 0 and std = 1
        image = cv2.imread(self.list_with_all_files[index][0])
        # transform_resize = torchvision.transforms.Resize(size=args.image_dimension)
        image = cv2.resize(image, args.image_dimension[::-1],  interpolation = cv2.INTER_NEAREST)   
        transform_img = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        # transform_resize,
                                                        torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                        std = [0.229, 0.224, 0.225])])                      
        image_tensor = transform_img(image)
        ### LIDAR DEPTH GT ###
        gt = np.load(self.list_with_all_files[index][6])
        gt = np.moveaxis(gt, -1, 0)
        # take only first channel of gt to get depth map and ignore uv
        gt_depth = gt[0, :, :]
       
        gt_max_depth = np.clip(gt_depth, 0, args.max_depth)
        
        # Inverse depth map
        gt_norm = gt_max_depth
        indices = gt_norm > 0
        gt_norm[indices] = (args.max_depth - gt_norm[indices]) * (1 / args.max_depth)

        gt_depth_tensor = torch.from_numpy(gt_norm)
        gt_depth_tensor = gt_depth_tensor.unsqueeze(0)
        if args.gt_uv:
            gt_uv_tensor = torch.from_numpy(gt[1:, :, :])
            gt_tensor = torch.cat((gt_depth_tensor, gt_uv_tensor), dim=0)
        else:
            gt_tensor = gt_depth_tensor
        
        
        ### SEGMENTATION GT and SEG_DEPTH GT###

        # Seg
        mseg = np.load(self.list_with_all_files[index][3])
        mseg= mseg[:416, :]
        seg_map = resize(mseg,(416, 800), order=0, preserve_range=True, anti_aliasing=False) #halve
        seg_map = torch.from_numpy(seg_map).to(torch.long)
        seg_map[seg_map == 255] = -1
        

       
        # Enhance lidar gt
        lidar_gt_enhanced =  cv2.dilate(gt_norm, np.ones((5, 5), np.uint8), iterations=1) #
        lidar_gt_enhanced[gt_norm > 0] = gt_norm[gt_norm > 0]
        lidar_gt_enhanced = torch.from_numpy(lidar_gt_enhanced).unsqueeze(0)
        # gt_tensor =lidar_gt_enhanced
        
        
        
        depth_map = gt_tensor.clone().squeeze(0)

        inter_depths_bins_gt = []
        depth_maps = [depth_map, depth_map[::2, ::2].contiguous(), depth_map[::4, ::4].contiguous()]
        for i, depth_map in enumerate(depth_maps):
            inter_depths_bins_gt.append((depth_map * (args.num_bins[::-1][i] - 1)).floor().to(torch.long))
            inter_depths_bins_gt[i][depth_map == 0] = -1
        
        depth_maps = [depth_map.unsqueeze(0) for depth_map in depth_maps]
        
        
        original_image = torch.from_numpy(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = torch.from_numpy(gray_image).unsqueeze(0)

        # Skys
        sky_seg = torch.ones_like(seg_map)
        # indices = torch.logical_and(seg_map == 14, (torch.rand(*sky_seg.shape) < 0.2))
        indices = seg_map == 14
        sky_seg[indices] = 0
        
        
        gt = {"depth": {'lidar_depth': gt_tensor,
                        "lidar_depth_enhance": lidar_gt_enhanced,
                        "depth_bins": inter_depths_bins_gt, 
                        "depth_maps": depth_maps},
                        # "sky_depth": sky,},
              'seg':{"seg_gt": seg_map, "sky_seg": sky_seg}
              }


        return {"image": image_tensor, "gt": gt, "name": name, "orig_img": original_image, "gray_img": gray_image}

    def __len__(self):
        return len(self.list_with_all_files)


if __name__ == "__main__": 
    
    directory = args.quasi_lidar_gt_path # Replace with the actual directory path

    # for file_name in os.listdir(directory):
    #     if 'liadr' in file_name:
    #         new_file_name = file_name.replace('liadr', 'lidar')
    #         old_file_path = os.path.join(directory, file_name)
    #         new_file_path = os.path.join(directory, new_file_name)
    #         os.rename(old_file_path, new_file_path)
    # exit()
    
    # print(np.load("/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/radar_depth/CamRaDepth/data/train_files/with_rain.npy"))
    # create_file_list("/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/nuscenes_mini/v1.0-mini/prepared_data/")
    dataloaders = make_dataloaders(split="train", train_part=0.8, num_samples=100, batchsize=1, data_type="cross_val", shuffle_training=True) 
    train_loader = dataloaders["train"]
    # mins = []
    # for b in train_loader:
    #     lidar_gt = b["gt"]["depth"]["lidar_depth"]
    #     cur_min = lidar_gt[lidar_gt > 0].min()
    #     mins.append(cur_min.item())
    # print(sorted(mins)[:100])


    
    # # create_new_split_file("/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/src/data/original_split.npy", "/home/ubuntu/Documents/students/DanHal/prepared_data", "/home/ubuntu/Documents/students/DanHal/local_split.npy")
    # seg_coeff = torch.tensor(args.seg_scheduler_values[0], requires_grad=False)
    # seg_optim = torch.optim.SGD([seg_coeff], lr=args.seg_scheduler_values[0])
    # seg_coeff_scheduler = torch.optim.lr_scheduler.OneCycleLR(seg_optim, max_lr=args.seg_scheduler_values[0], steps_per_epoch=len(train_loader), 
    #                                                          epochs=args.num_epochs, div_factor=args.seg_scheduler_values[0] / args.seg_scheduler_values[1], 
    #                                                          pct_start=args.pct_start / args.num_epochs, final_div_factor=args.seg_scheduler_values[1] / args.seg_scheduler_values[2])
    
    
    # for i, b in enumerate(train_loader):
    #     seg_optim.step()
    #     seg_coeff_scheduler.step()
    #     print(seg_coeff_scheduler.get_last_lr   () ,seg_coeff.item() )

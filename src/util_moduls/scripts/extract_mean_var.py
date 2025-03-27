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
from matplotlib import pyplot as plt
import torchvision.transforms as T

from util_moduls.loss_funcs import MaskedMSELoss


from scipy.stats import norm, shapiro
import scipy.stats as stats

def plot_histogram_2(source, target, feature_index, path):
    hashtag = args.hashtags_prefix
    mu1, std1 = source.mean(), source.std()
    mu2, std2 = target.mean(), target.std()
    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ feature_index: {feature_index} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"{hashtag} mu1: {mu1}, std1: {std1}")
    print(f"{hashtag} mu2: {mu2}, std2: {std2}")

    # Plot the PDF of the fitted normal distribution for source
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    xmin1, xmax1 = plt.xlim(source.min(), source.max())
    x1 = np.linspace(xmin1, xmax1, 100)
    p1 = norm.pdf(x1, mu1, std1)
    plt.plot(x1, p1, 'k', linewidth=2)
    plt.hist(source.flatten(), bins='auto', density=True, alpha=0.6, color='g')
    plt.title(f'Histogram of Feature {feature_index} (Source)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plot the PDF of the fitted normal distribution for target
    plt.subplot(1, 2, 2)
    xmin2, xmax2 = plt.xlim(target.min(), target.max())
    x2 = np.linspace(xmin2, xmax2, 100)
    p2 = norm.pdf(x2, mu2, std2)
    plt.plot(x2, p2, 'k', linewidth=2)
    plt.hist(target.flatten(), bins='auto', density=True, alpha=0.6, color='b')
    plt.title(f'Histogram of Feature {feature_index} (Target)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'histogram_feature_{feature_index}.png'))
    plt.close()  # Close the plot to free up resources

    alpha = 0.05

    stat1, p1 = shapiro(source.flatten())
    stat2, p2 = shapiro(target.flatten())
    
    print(f"{hashtag} Shapiro-Wilk Test (Source): stat1: {stat1}, p1: {p1}")
    print(f"{hashtag} Shapiro-Wilk Test (Target): stat2: {stat2}, p2: {p2}")

    if p1 > alpha:
        print(f"{Fore.GREEN}✅ Source appears to be normally distributed (p > 0.05){Style.RESET_ALL}")
        result1 = 1
    else:
        print(f"{Fore.RED}❌ Source does not appear to be normally distributed (p <= 0.05){Style.RESET_ALL}")
        result1 = 0

    if p2 > alpha:
        print(f"{Fore.GREEN}✅ Target appears to be normally distributed (p > 0.05){Style.RESET_ALL}")
        result2 = 1
    else:
        print(f"{Fore.RED}❌ Target does not appear to be normally distributed (p <= 0.05){Style.RESET_ALL}")
        result2 = 0

    return result1, result2


def plot_histogram_3(Source, Projection, Target, feature_index, path):
    hashtag = args.hashtags_prefix
    
    # Calculate statistics for Source, Projection, and Target
    mu1, std1 = Source.mean(), Source.std()
    mu2, std2 = Projection.mean(), Projection.std()
    mu3, std3 = Target.mean(), Target.std()
    
    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ feature_index: {feature_index} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    
    print(f"{hashtag} mu1: {mu1}, std1: {std1}")
    print(f"{hashtag} mu2: {mu2}, std2: {std2}")
    print(f"{hashtag} mu3: {mu3}, std3: {std3}")

    # Plot histograms and PDFs for Source, Projection, and Target
    plt.figure(figsize=(18, 4))
    # Source
    plt.subplot(1, 3, 1)
    xmin1, xmax1 = plt.xlim(Source.min(), Source.max())
    x1 = np.linspace(xmin1, xmax1, 100)
    p1 = norm.pdf(x1, mu1, std1)
    plt.plot(x1, p1, 'k', linewidth=2)
    
    plt.hist(Source.flatten(), bins='auto', density=True, alpha=0.6, color='g')
    plt.title(f'Histogram of Feature {feature_index} (Source)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Projection
    plt.subplot(1, 3, 2)
    xmin2, xmax2 = plt.xlim(Projection.min(), Projection.max())
    x2 = np.linspace(xmin2, xmax2, 100)
    p2 = norm.pdf(x2, mu2, std2)
    plt.plot(x2, p2, 'k', linewidth=2)
    plt.hist(Projection.flatten(), bins="auto", density=True, alpha=0.6, color='b')
    plt.title(f'Histogram of Feature {feature_index} (Projection)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Target
    plt.subplot(1, 3, 3)
    xmin3, xmax3 = plt.xlim(Target.min(), Target.max())
    x3 = np.linspace(xmin3, xmax3, 100)
    p3 = norm.pdf(x3, mu3, std3)
    plt.plot(x3, p3, 'k', linewidth=2)
    # plt.plot(torch.arange(Target.shape[0]), Target)
    
    plt.hist(Target.flatten(), bins='auto', density=True, alpha=0.6, color='r')
    plt.title(f'Histogram of Feature {feature_index} (Target)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(path, f'histogram_feature_{feature_index}.png'))
    plt.close()  # Close the plot to free up resources

    alpha = 0.05

    # Shapiro-Wilk tests for Source, Projection, and Target
    stat1, p1 = shapiro(Source.flatten())
    stat2, p2 = shapiro(Projection.flatten())
    stat3, p3 = shapiro(Target.flatten())
    
    def print_test_result(test_name, p_value):
        if p_value > alpha:
            print(f"{Fore.GREEN}✅ {test_name} appears to be normally distributed (p > 0.05){Style.RESET_ALL}")
            return 1
        else:
            print(f"{Fore.RED}❌ {test_name} does not appear to be normally distributed (p <= 0.05){Style.RESET_ALL}")
            return 0

    result1 = print_test_result(f"s1, {Source.shape}", p1)
    result2 = print_test_result(f"s2, {Projection.shape}", p2)
    result3 = print_test_result(f"s3, {Target.shape}", p3)

    return result1, result2, result3

def plot_histogram(data, feature_index, path):
    hashtag = args.hashtags_prefix
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"{hashtag} Feature {feature_index} {hashtag}")
    print(f"{hashtag} data.shape: {data.shape}")
    mu, std = data.mean(), data.std() 

    print(f"{hashtag} mu: {mu}, std: {std}")
    # Plot the PDF of the fitted normal distribution
    xmin, xmax = plt.xlim(-4, 4)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.hist(data.flatten(), bins='auto', density=True, alpha=0.6, color='g')
    plt.title(f'Histogram of Feature {feature_index}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.savefig(os.path.join(path, f'histogram_feature_{feature_index}.png'))
    plt.close()  # Close the plot to free up resources
    # Assuming 'data' is your dataset
    
    alpha = 0.05
    
    stat, p = shapiro(data.flatten())
    # w = shapiro_wilk_test(data)
    print(f"{hashtag} stat: {stat}, p: {p}")

    if p > alpha:
        print(f"{Fore.GREEN}✅ Data appears to be normally distributed (p > 0.05){Style.RESET_ALL}")
        result = 1
    else:
        print(f"{Fore.RED}❌ Data does not appear to be normally distributed (p <= 0.05){Style.RESET_ALL}")
        result = 0
    return result

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
    
    # args.model = "Depth"
    # args.input_channels = 5
    if args.model == "Depth":
        dir_path = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Depth"
        # dir_path = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Playground"
    elif args.model == "Projection":
        dir_path = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Projection"
    elif args.model == "Embedding":
        dir_path = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Embedding"
    args.checkpoint = get_latest_file_in_folder(dir_path)

    # args.checkpoint = file
    args.checkpoint = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Embedding/28/mlt_epoch_1_best_RMSE_2.6972832.pth"
    # args.model = "Depth"
    # args.input_channels = 5
    # args.model = "Embedding"
    # args.checkpoint = "/media/EDGAR/02_students/Studenten_Sauerbeck/Dan_Halperin/CamRaDepth/Output/Transformer/AE/Embedding/15/mlt_epoch_22_best_RMSE_2.9465398.pth"

    # If your GPU is currently occupied, you could simply use the CPU to visuazlie your recent predictions.
    use_gpu = True
    if use_gpu:
        cuda_id = 0
        with torch.cuda.device(cuda_id):
            model  = AE(input_channels=args.input_channels)
            device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
            state = torch.load(args.checkpoint, map_location=device)
            load_checkpoint_with_shape_match(model, state['state_dict'])
            print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
            model.eval()
            model.to(device)
    else:
        model  = AE(input_channels=args.input_channels)
        device = torch.device('cpu')
        state = torch.load(args.checkpoint, map_location=device)
        load_checkpoint_with_shape_match(model, state['state_dict'])
        print(f"{args.hashtags_prefix} Loaded model from {args.checkpoint}")
        model.eval()
        model.to(device)
    
    # dataloaders = make_dataloaders(batchsize=1, split="test")
    
    if False:
    
    ##################### Check for normality (gaussianity) #####################
        dataloaders = make_dataloaders(batchsize=1, split="test")
        # dataloaders = make_dataloaders(batchsize=1, split="test")
        test_dl = dataloaders["test"]
        dataloaders = make_dataloaders(batchsize=1, split="train", shuffle_training=False)
        trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
        splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
        keys, values = list(splits.keys()), list(splits.values())
        path = Path(args.output_dir) / "mean_and_var_ckpt"  # Where should it be created?
        
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        
        
        projection_flag = args.model == "Projection"
        
        # loop = create_tqdm_bar(trai_dl, "train")
        with torch.no_grad():
            
            mean, std = None, None
            counter = 0
            full_e = None
            full_e_d = None
            full_source = None
            acc_samples = 1
            for i, batch in enumerate(test_dl):
                # depth_embedding = batch["gt"]["depth"]["depth_embedding"].to(torch.float32).to(device)
                x = batch["image"].to(torch.float32).to(device)[:, :args.input_channels]
                pred = model(x, gt_image=batch["gt"]["depth"]["lidar_depth_enhance"].to(torch.float32).cuda())
                
                
                # pred_transform = pred["transformation"]["pred_transform"]
                # proc_transform = pred["transformation"]["proc_transform"]
                
                # R_pred, t_pred, s_pred = pred_transform["R"], pred_transform["t"], pred_transform["s"]
                # R_proc, t_proc, s_proc = proc_transform["R"], proc_transform["t"], proc_transform["s"]
                
                # # print(torch.det(R_pred), torch.det(R_proc))
        
                # print(R_pred[0])
                # print(R_proc[0])
                # exit()

                name =  [".".join(n.split(".")[:-1]) + ".npy" for n in batch["name"]][0]
                
                num_samples = 40
                
                # if projection_flag:
            
                #     e = pred["latent_spaces"]["projected_latent_space"].detach().cpu()
                #     full_e = e if full_e is None else torch.cat([full_e, e], dim=0)
                    
                #     full_e = full_e.flatten(2).permute(0, 2, 1)
                #     full_source = full_source.flatten(2).permute(0, 2, 1)
                #     full_e_d = full_e_d.flatten(2).permute(0, 2, 1)
                    
                #     if (i + 1) % acc_samples == 0:
                        
                #         acc_v_1 = 0
                #         acc_v_2 = 0
                #         acc_v_3 = 0
                #         for k in range(full_e.shape[0]):
                #             # for j in range(full_e.shape[1]):
                #             for j in range(num_samples):
                #                 if j == num_samples:
                #                     break
                #                 # acc_v += plot_histogram(full_e[k, j], f"{j}-{k+1}", path)
                #                 val_1, val_2, val_3 = plot_histogram_3(full_source[k, j], full_e[k, j], full_e_d[k, j], f"{j+1}-{k+1}", path)
                #                 acc_v_1 += val_1
                #                 acc_v_2 += val_2
                #                 acc_v_3 += val_3
                #         print(f"source: {acc_v_1 / (full_e.shape[0] * num_samples)}, projection: {acc_v_2 / (full_e.shape[0] * num_samples)}, target: {acc_v_3 / (full_e.shape[0] * num_samples)}")
                #         full_e = None
                #         break
                
                # else:
                    
                if (i + 1) % acc_samples == 0:
                    if args.model == "Depth":
                        x = pred["transformation"]["rgbd_pc"].reshape(1, 288, 80, 3)
                    elif args.model == "Embedding":
                        x = pred["transformation"]["trans_dict"]["rgbd_pc"].reshape(1, 288, 80, 3)
                    s_1 = x[:, :, :, 0].detach().cpu().numpy()
                    s_2 = x[:, :, :, 1].detach().cpu().numpy()
                    s_3 = x[:, :, :, 2].detach().cpu().numpy()
                    
                    # s_1 = pred["transformation"]["rgbd_latent"].flatten(2).detach().cpu().numpy()
                    # s_2 = pred["transformation"]["rgbd_latent"].flatten(2).detach().cpu().numpy()
                    # # s_2 = pred["transformation"]["rgb_latent"].flatten(2).permute(0, 2, 1).detach().cpu().numpy()
                    # s_3 = pred["transformation"]["rgbd_latent"].flatten(2).permute(0, 2, 1).detach().cpu().numpy()
                    # s_3 = pred["transformation"]["rgbd_latent_tanh"].flatten(2).permute(0, 2, 1).detach().cpu().numpy()
                    # s_1, s_2, s_3 = encoder_outputs[-1].flatten(2).permute(0, 2, 1).detach().cpu().numpy(), encoder_outputs[-2].flatten(2).permute(0, 2, 1).detach().cpu().numpy(), encoder_outputs[-3].flatten(2).permute(0, 2, 1).detach().cpu().numpy()
                    # s_1 = pred["transformation"]["s_latent"].flatten(2).detach().cpu().numpy()
                    # print(f"s1: {s_1.shape}, s2: {s_2.shape}, s3: {s_3.shape}"); exit()
                    

                    # print(f"full_source norms: {full_source.norm(dim=2)}, full_e_d norms: {full_e_d.norm(dim=2)}")
                    
                    first, second, third, = 0, 0, 0
                    for k in range(s_1.shape[0]):
                        for j in range(num_samples):
                            if j == num_samples:
                                break
                            val_1, val_2, val_3 = plot_histogram_3(s_1[k, j], s_2[k, j], s_3[k, j], f"{j+1}-{k+1}", path)
                            first += val_1
                            second += val_2
                            third += val_3
                            
                    print(f"s1: {first / (s_1.shape[0] * num_samples)}, s2: {second / (s_1.shape[0] * num_samples)}, s3: {third / (s_1.shape[0] * num_samples)}")
                    break
                
        print(args.checkpoint)
            
    ############################# Save the latent space ############################# 
    else:
        
        assert args.model in ["Depth", "Embedding"]
        
        if args.model == "Embedding":
            print(f"{args.hashtags_prefix} Extracting latent space from Depth model")
            batch_size = 8
            dataloaders = make_dataloaders(batchsize=batch_size, split="train")
            trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
            test_dl = make_dataloaders(batchsize=batch_size, split="test")["test"]
            splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
            splits_and_loaders = [(k, v) for k, v in zip(list(splits.keys()), list(splits.values()))]
            split, loader = [(k, v) for k, v in zip(list(splits.keys()), list(splits.values()))][1]
            orig_em_path = Path(args.output_dir) / "quasi_gt" # Where should it be created?
            # orig_em_path = Path(args.output_dir) / "latent_spaces" # Where should it be created?
            os.makedirs(orig_em_path, exist_ok=True)
            
            # loop = create_tqdm_bar(loader, split)
            with torch.no_grad():
                for split, loader in splits_and_loaders:
                    loop = create_tqdm_bar(loader, split)
                    for i, batch in loop:
                        gt_x = batch["gt"]["depth"]["lidar_depth_enhance"].to(torch.float32).to(device)
                        depth_gt = batch["gt"]["depth"]["depth_maps"][0].to(torch.float32).to(device)
                        x = batch["image"].to(torch.float32).to(device)
                        names = [".".join(n.split(".")[:-1]) + ".npy" for n in batch["name"]]
                        pred_dict = model(x, gt_x)

                        pc = pred_dict["transformation"]["trans_dict"]["rgbd_pc"].detach().cpu()
                        pred_depth = pred_dict["rgb_decoder_output"]["depth"]["final_depth"]
                        rmse = torch.sqrt(MaskedMSELoss()(pred_depth, depth_gt)) * args.max_depth
                        loop.set_postfix({"RMSE": rmse.item(), "shape": pc.shape})
                        # orig_e = model.encoder(x)[-1].detach().cpu()
                        # for e, name in zip(orig_e, names):
                        #     np.save(os.path.join(orig_em_path, name), e)
                        
                        # pred_depth = pred_dict["depth"]["final_depth"].detach().cpu()
                        for e, name in zip(pc, names):
                            
                            np.save(os.path.join(orig_em_path, name), e)
                       
                        
        # elif args.model == "Embedding":
        #     print(f"{args.hashtags_prefix} Extracting latent space from Embedding model")
        #     batch_size = 16
        #     dataloaders = make_dataloaders(batchsize=batch_size, split="train")
        #     trai_dl, val_dl = dataloaders["train"], dataloaders["val"]
        #     test_dl = make_dataloaders(batchsize=batch_size, split="test")["test"]
        #     splits = {"train": trai_dl, "val": val_dl, "test": test_dl}
        #     splits_and_loaders = [(k, v) for k, v in zip(list(splits.keys()), list(splits.values()))]
        #     split, loader = [(k, v) for k, v in zip(list(splits.keys()), list(splits.values()))][1]
        #     orig_em_path = Path(args.output_dir) / "latent_spaces_projected" # Where should it be created?
        #     processed_em_path = Path(args.output_dir) / "latent_spaces_processed" # Where should it be created? 
        #     os.makedirs(orig_em_path, exist_ok=True)
        #     os.makedirs(processed_em_path, exist_ok=True)
            
        #     # loop = create_tqdm_bar(loader, split)
        #     with torch.no_grad():
        #         for split, loader in splits_and_loaders:
        #             loop = create_tqdm_bar(loader, split)
        #             for i, batch in loop:
        #                 x = batch["image"].to(torch.float32).to(device)
        #                 names = [".".join(n.split(".")[:-1]) + ".npy" for n in batch["name"]]
        #                 pred_dict = model(x, batch["gt"]["depth"]["depth_embedding"].to(torch.float32).cuda())
                        
        #                 # e = pred_dict["latent_spaces"]["projected_latent_space"].detach().cpu()
        #                 # latent = pred_dict["latent_spaces"]["latent_space"].detach().cpu()
        #                 e = pred_dict["latent_spaces"]["target_3s"].detach().cpu()
        #                 latent = pred_dict["latent_spaces"]["source_3s"].detach().cpu()
        #                 loop.set_postfix({"proj_shape": e.shape, "latent_shape": latent.shape})
        #                 for e, name in zip(e, names):
        #                     np.save(os.path.join(orig_em_path, name), e)
        #                 for e, name in zip(latent, names):
        #                     np.save(os.path.join(processed_em_path, name), e)

        
        
        
        
        
        
        
        
        
        
       
        
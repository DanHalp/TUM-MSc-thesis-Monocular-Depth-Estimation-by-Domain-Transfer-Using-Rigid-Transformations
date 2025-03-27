
import os, sys
from pathlib import Path


sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[4]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))


import datetime
import torch
import torch.nn as nn
import math
import time
import numpy as np
from util_moduls.args import args 
from data.dataloader_vae import make_dataloaders

from src.model.AE import AE

from torch.optim import RAdam
from src.util_moduls.utils_functions import load_checkpoint_with_shape_match, create_tqdm_bar,  save_files, adjust_loss
from util_moduls.loss_funcs import MaskedFocalLoss, MaskedMSELoss, InfinityL1Loss, DepthLoss, CustomChi2Loss
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from roma import rotmat_geodesic_distance
import torch


class Trainer:
    
    def __init__(self, model, mode="train") -> None:
        
        self.device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
        self.training_steps = 0
        self.val_steps = 0
        print(f"{args.hashtags_prefix} Using device: {self.device}")
        self.model = model(input_channels=args.input_channels, mode=args.model).to(self.device)

        ################# Create a folder for the current run ################# 
        if args.save_model and mode == "train":
            out = Path(args.output_dir) / Path(args.arch_name) 
            os.makedirs(out, exist_ok=True)
            dirs = os.listdir(out)
            dirs += ["0"]
            index = str(max([int(x) for x in dirs if x.isdigit()]) + 1)
        
            # path = out / index if args.run_name is None else out / args.run_name / "Playground"
            path = out / index if args.run_name is None else out / args.run_name / args.model
            os.makedirs(path, exist_ok=True)
            
            dirs = os.listdir(path)
            dirs += ["0"]
            index = str(max([int(x) for x in dirs if x.isdigit()]) + 1)
            path = path / index 
            os.makedirs(path, exist_ok=True)
            
            self.new_log_path = path 
            self.new_run_path =  path
            # self.tb_logger = SummaryWriter(path, flush_secs=10)
            print(f"{args.hashtags_prefix} Saving to {self.new_run_path}")
            save_files(self.new_run_path)
            
            
        if mode == "test" and (args.checkpoint is None):
            raise ValueError("A checkpoint is needed for testing!")
        
        ################# Save comulative losses, for logging and later visualizing #################
        self.train_losses_list = []
        self.val_losses_list = []
        
        ################# Load a checkpoint if needed #################
        if args.checkpoint is not None and args.load_ckpt:
            if os.path.exists(args.checkpoint):
                device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
                state = torch.load(args.checkpoint, map_location=device, weights_only=False)
                load_checkpoint_with_shape_match(self.model, state["state_dict"])
                # args.learning_rate = state.get("lr", args.learning_rate)
                self.train_losses_list = state.get("train_losses", self.train_losses_list)
                self.val_losses_list = state.get("val_losses", self.val_losses_list)
                print(f"{args.hashtags_prefix} Loaded checkpoint from {args.checkpoint}")                    
                
            else:
                raise ValueError(f"Checkpoint not found at {os.path.abspath(args.checkpoint)}!")
            
        ################# Freezing differnet parts of the models #################
        
        # freeze_and_unfreeze_model(self.model, freeze=True)
        # freeze_and_unfreeze_model(self.model.discriminator, freeze=False)
        # freeze_and_unfreeze_model(self.model.rgb_encoder, freeze=True)
        # self.model.reset_decoder()
          
        self.model = self.model.to(self.device)
        
        ################# Create dataloaders #################
        if args.overfit:
            dataloaders = make_dataloaders(num_samples=1000, train_part=0.8, split=mode)
        else:
            dataloaders = make_dataloaders(mode)
            
        
        print(f"{args.hashtags_prefix} Mode: Train")
        if args.num_steps is not None:
            print(f"{args.hashtags_prefix} No. steps: {args.num_steps}")
        print(f"{args.hashtags_prefix} No. epochs: {args.num_epochs}")
        print(f"{args.hashtags_prefix} warmup epochs: {args.pct_start}")

        self.train_dataloader, self.val_dataloader = dataloaders["train"], dataloaders["val"]
        
        self.criterion = {'depth': DepthLoss(), 'seg': MaskedFocalLoss()}
        
        self.optimizer =  RAdam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, decoupled_weight_decay=True)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.learning_rate, steps_per_epoch=len(self.train_dataloader), 
                                                             epochs=args.num_epochs, div_factor=args.div_factor, pct_start=args.pct_start / args.num_epochs, final_div_factor=args.final_div_factor)
        
        self.scaler = torch.amp.GradScaler('cuda')
       
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            
        ################# Differnet schedulers for the differnet losses #################
        len_train = len(self.train_dataloader)
        sky_coeff = torch.tensor(args.sky_scheduler_values[0], requires_grad=False)
        self.sky_optim = torch.optim.SGD([sky_coeff], lr=args.sky_scheduler_values[0])
        self.sky_coeff_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.sky_optim, max_lr=args.sky_scheduler_values[0], steps_per_epoch=len(self.train_dataloader),
                                                            epochs=args.num_epochs, div_factor=args.sky_scheduler_values[0] / args.sky_scheduler_values[1], 
                                                            pct_start=args.pct_start / args.num_epochs, final_div_factor=args.sky_scheduler_values[1] / args.sky_scheduler_values[2])
    
        dist_coeff = torch.tensor(args.dist_scheduler_values[0], requires_grad=False)
        self.dist_optim = torch.optim.SGD([dist_coeff], lr=args.dist_scheduler_values[0])
        self.dist_coeff_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.dist_optim, max_lr=args.dist_scheduler_values[0], steps_per_epoch=len(self.train_dataloader),
                                                            epochs=args.num_epochs, div_factor=args.dist_scheduler_values[0] / args.dist_scheduler_values[1], 
                                                            pct_start=args.pct_start / args.num_epochs, final_div_factor=args.dist_scheduler_values[1] / args.dist_scheduler_values[2])
    def train_one_epoch(self, epoch):
        self.model.train()
        
        loop = create_tqdm_bar(self.train_dataloader, f"Training [{epoch + 1}/{args.num_epochs}]")
  
        counter = 0
        all_losses, final_losses = [], []
        self.optimizer.zero_grad() 
        MSE_loss = MaskedMSELoss()
        infinity_loss = InfinityL1Loss()
        deph_loss_classs = DepthLoss()

        cur_lr = self.optimizer.param_groups[0]['lr']
        best_loss_val = np.inf
       
        for i, batch in loop:
        
            # try: 
            gt = batch["gt"]
            gt_sky_seg = gt["seg"]["sky_seg"].to(torch.long).cuda()
            gt_depth_416x800 = gt["depth"]["lidar_depth"].to(torch.float32).cuda()
    
            x = batch["image"].to(torch.float32).cuda()
        
            gt_x = gt["depth"]["augmented_depth"].to(torch.float32).cuda()
            with torch.autocast('cuda'):
                
                pred_dict = self.model(x, gt_image=gt_x)
                
                ################# Both AE depth losses and infinity losses #################
                proc_pred_depth = pred_dict["proj_decoder_output"]["depth"]["final_depth"]
                proc_pred_logits = pred_dict["proj_decoder_output"]["depth"]["logits"]
                gt_sky_seg = gt_sky_seg.reshape(proc_pred_depth.shape)
                true_sky_indices = torch.logical_and(gt_sky_seg == 0, gt_depth_416x800 == 0)
                false_sky_indices = (proc_pred_depth < args.false_sky_thresh)
                indices = torch.logical_or(true_sky_indices, false_sky_indices)
                proc_sky_loss = infinity_loss(proc_pred_depth[indices], gt_sky_seg[indices])
                proc_depth_loss = deph_loss_classs.depth_loss(proc_pred_depth, gt_depth_416x800) 
                proc_RMSE = torch.sqrt(MSE_loss(proc_pred_depth, gt_depth_416x800)).item() * args.max_depth
                
                proj_pred_depth = pred_dict["rgb_decoder_output"]["depth"]["final_depth"]
                gt_sky_seg = gt_sky_seg.reshape(proj_pred_depth.shape)
                true_sky_indices = torch.logical_and(gt_sky_seg == 0, gt_depth_416x800 == 0)
                false_sky_indices = (proj_pred_depth < args.false_sky_thresh)
                indices = torch.logical_or(true_sky_indices, false_sky_indices)
                proj_sky_loss = infinity_loss(proj_pred_depth[indices], gt_sky_seg[indices])
                proj_depth_loss = deph_loss_classs.depth_loss(proj_pred_depth, gt_depth_416x800) 
                proj_RMSE = torch.sqrt(MSE_loss(proj_pred_depth, gt_depth_416x800)).item() * args.max_depth
                
                ################# Extra losses #################
                rgb_latent = pred_dict["projection"]["latents"]["rgb"]
                depth_latent = pred_dict["projection"]["latents"]["depth"]      
                coral_loss = F.mse_loss(rgb_latent, depth_latent)
                rot_loss = torch.tensor(0.0).to(self.device)
                
            beta = 1
            ################# Adjusting loss and setting coefficients #################
            proj_depth_coeff = adjust_loss(proc_depth_loss.item(), proj_depth_loss.item(), nominator = beta)
            proc_depth_coeff = adjust_loss(proc_depth_loss.item(), proc_depth_loss.item(), nominator = 1)
            proj_sky_coeff = adjust_loss(proc_depth_loss.item(), proj_sky_loss.item(), nominator = self.sky_coeff_scheduler.get_last_lr()[0] * beta)    
            proc_sky_coeff = adjust_loss(proc_depth_loss.item(), proc_sky_loss.item(), nominator = self.sky_coeff_scheduler.get_last_lr()[0])
            coral_coeff = adjust_loss(proc_depth_loss.item(), coral_loss.item(), nominator = self.dist_coeff_scheduler.get_last_lr()[0])
            rot_coeff = adjust_loss(proc_depth_loss.item(), rot_loss.item(), nominator = self.dist_coeff_scheduler.get_last_lr()[0])
            
  
            d_l = (proj_depth_loss * proj_depth_coeff + proc_depth_loss * proc_depth_coeff) 
            s_l = (proj_sky_loss * proj_sky_coeff + proc_sky_loss * proc_sky_coeff) / 2
            c_l = (coral_loss * coral_coeff + rot_loss * rot_coeff)
    
            ae_loss = d_l + s_l + c_l 
            loss = ae_loss 
            
            # For logging
            all_losses.append([
                            proc_depth_loss.item(), proc_sky_loss.item(), proc_RMSE,
                            proj_depth_loss.item(), proj_sky_loss.item(), proj_RMSE,
                            coral_loss.item(), rot_loss.item(),
                                ])
            
            
            # calculates gradients
            loss = loss / args.update_interval # Accumulated gradients need to be divided by the number of iterations
            self.scaler.scale(loss).backward()
            
            ################# Debugging: Print the gradient's norm, to catch exploding gradients. #################
            
            # Print the gradients, to ensure that they don't explode - for debugging purposes
            # if not args.save_model:
            #     total_norm = 0
            #     max_norm = 0
            #     print()

            #     for name, p in self.model.named_parameters():
            #         if p.grad is None:
            #             continue
                    
            #         param_norm = p.grad.data.norm(2)
                    
            #         if param_norm > max_norm:
            #             max_norm = param_norm
            #             print(name, param_norm.item())
                    
            #         max_norm = max(max_norm, param_norm.item())
            #         total_norm += param_norm.item() ** 2
            #     total_norm = total_norm ** (1. / 2)
            #     print(f"Total norm: {total_norm}, Max norm: {max_norm}")
            #     print("#####################################################")
            
            ################# If accumelative gradients are required - in case the hardware is not sufficient for a big batch #################
            # Perform the optimizer step after the required number of iterations for accumelated gradients (Could also be simply 1).
            progess_bool = (i + 1) % args.update_interval == 0 or (i + 1) == len(self.train_dataloader)
            
            if progess_bool:
                
                # Update the progress tqdm bar and Tensorboard.
                cutoff = 200
                cur_lr = self.optimizer.param_groups[0]['lr']
                all_losses = all_losses[-cutoff:]
                mean_losses = np.nanmean(all_losses, axis=0)
                final_losses = np.array([mean_losses[2], mean_losses[5], cur_lr])  
                
                loop.set_postfix(
                                    a_lr = "{:.2e}".format(cur_lr),
                                    c_RMSE = "{:.5f}/{:.5f}".format(mean_losses[2], mean_losses[5]),
                                    d_sky = "{:.2e}/{:.2e}/{:.2e}".format(mean_losses[1], mean_losses[4], self.sky_coeff_scheduler.get_last_lr()[0]),
                                    e_coral = "{:.2e}".format(mean_losses[-2]),
                                    f_rot = "{:.2e}".format(mean_losses[-1]),
                                    z_dist = "{:.2e}".format(self.dist_coeff_scheduler.get_last_lr()[0]),
                )

                if args.save_model:

                    if counter >= 300 :
                        best_loss_val = mean_losses[2]
                        state = {'state_dict': self.model.to('cpu').state_dict(), 'optimizer': self.optimizer.state_dict(), "lr": self.optimizer.param_groups[0]['lr'], "steps": [self.training_steps, self.val_steps], "train_losses": self.train_losses_list, "val_losses": self.val_losses_list}
                        path = os.path.join(self.new_run_path, "mlt" + '_epoch_' + str(epoch+1) + "_iter_" +  str(i) + "_intermediate_" + "{:.7f}".format(best_loss_val) +"_chi2_" +"{:.2e}".format(mean_losses[5]) + '.pth')
                        torch.save(state, path)
                        self.model.to(device=self.device)
                        counter = 0
 
                counter += 1
                self.training_steps += 1
                
                ################# Scheduler and optimizer steps #################     
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 500)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            # To prevent a scheduler step before the optimizer step.
            if (i + 1) > args.update_interval:
                self.scheduler.step()
                self.sky_optim.step()
                self.sky_coeff_scheduler.step()
                self.dist_optim.step()
                self.dist_coeff_scheduler.step()
        
        return final_losses
                   
            
    def eval(self, epoch):
        self.model.eval()
        all_losses = []
        mean_losses = []
      
        with torch.no_grad():
            loop_eval = create_tqdm_bar(self.val_dataloader, f"Val [{epoch + 1}/{args.num_epochs}]") # tqdm is needed for visualization of the training progess in the terminal
            for i, batch in loop_eval:
                
                gt = batch["gt"]
                gt_depth_final = gt["depth"]['lidar_depth'].to(torch.float32).cuda()
                x = batch["image"].to(torch.float32).cuda()
                gt_x = gt["depth"]["augmented_depth"].to(torch.float32).cuda()

                with torch.autocast("cuda"):
                    start = time.time()
                    pred_dict = self.model(x, gt_image=gt_x)
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
    
                
                ################# LOGGING #################
                progess_bool = (i + 1) % args.update_interval == 0 or (i + 1) == len(self.val_dataloader)
                if progess_bool:
                    
                    mean_losses = np.nanmean(all_losses, axis=0)
                    
                    loop_eval.set_postfix(
                        v_RMSE_rgb = mean_losses[0],
                        v_RMSE_proj= mean_losses[1],
                        v_RMSE_zero_shot = mean_losses[2],
                        time_s = "{:.6f}".format(mean_losses[-1]),
                    )
          
                    self.val_steps += 1
       
        return np.array(mean_losses)

    def train(self):

        start_training = time.time()
        early_stop_counter = 0
 

        # Create the models saving folder for the current run
  
        for epoch in range(args.num_epochs):
            train_losses = self.train_one_epoch(epoch=epoch)
            val_losses = self.eval(epoch=epoch)
            print("{} Eval loss: rgb: {:.3e}, proj: {:.3e}, zero_shot: {:.3e}, time: {:.2f}s".format(args.hashtags_prefix, *val_losses))
            
            if len(self.train_losses_list) == 0:
                self.train_losses_list = np.array(train_losses)[None, :]
                self.val_losses_list = np.array(val_losses)[None, :]
            else:
                try:
                    self.train_losses_list = np.vstack((self.train_losses_list, train_losses))
                    self.val_losses_list = np.vstack((self.val_losses_list, val_losses))
                except Exception as e:
                    self.train_losses_list = np.array(train_losses)[None, :]
                    self.val_losses_list = np.array(val_losses)[None, :]
                

            ################# LOGGING #################
            RMSE = val_losses[1]
            aux = val_losses[0]
            if args.save_model:
                try:
                    state = {'state_dict': self.model.to('cpu').state_dict(), 'optimizer': self.optimizer.state_dict(), "lr": self.optimizer.param_groups[0]['lr'], "steps": [self.training_steps, self.val_steps], "train_losses": self.train_losses_list, "val_losses": self.val_losses_list}
                    path = os.path.join(self.new_run_path, "mlt" + '_epoch_' + str(epoch+1) +"_best_RMSE_" + "{:.7f}".format(RMSE.item()) + "_aux_" + "{:.3e}".format(aux) + '.pth')
                    torch.save(state, path)
                    self.model.to(device=self.device)
                    print(f'{args.hashtags_prefix} Model saved to {self.new_run_path}')
                except Exception as e:
                    time.sleep(10)
                
                ################# Delete all the intermediate files #################
                for filename in os.listdir(self.new_run_path):
                    if "intermediate" in filename:
                        file_path = os.path.join(self.new_run_path, filename)
                        os.remove(file_path)
            
        stop_training = time.time()

        print('Training done.')
        print('Time for total training    : ', str(datetime.timedelta(seconds=stop_training - start_training)))


if __name__ == "__main__":
    
    
    # For easily tweaking the arguments. One could also follow the more conventional way of passing arguments to the command line,
    
    if args.run_mode == "train":
        Trainer(model=AE).train()
    elif args.run_mode == "test":
        Trainer(model=AE, mode=args.run_mode).test()
    else:
            raise ValueError("Invalid run mode. Please choose between 'train' and 'test'.")


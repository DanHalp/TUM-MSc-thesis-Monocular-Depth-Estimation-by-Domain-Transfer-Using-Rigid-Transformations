import os, sys
from pathlib import Path
import argparse
from easydict import EasyDict as edict
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))
 # Add the main directory to the python path

this_dir = this_dir = os.path.dirname(__file__)

args = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Radar Depth Completion')

# Dataset
args.add_argument("--max_depth", type=float, default=100, help="Maximum depth in meters")
args.add_argument("--max_distances", type=list, default=[100, 50], help="Maximum distances in meters")
args.add_argument("--mini_dataset", action="store_true", help="If set, the mini dataset is used")
# args.add_argument("--train_val_split", type=list, default=[17902, 2237], help="Splits the Dataset in train and validation set; \
#     added it must be equal to num_samples")
args.add_argument("--image_dimension", type=tuple, default=(416, 800), help="Image dimension")
args.add_argument("--val_test_size", type=int, default=2237, help="Size of the validation and test set")
args.add_argument("--num_workers", type=int, default=16, help="Number of workers for the dataloaders")
args.add_argument("--split", type=str, default="original_split.npy", help="Path to the split file")

# Semantic Segmentation
args.add_argument("--supervised_seg", action="store_true", help="If set, we deploy a supervised semantic segmentation branch")
args.add_argument("--unsupervised_seg", action="store_true", help="If set, we deploy an unsupervised semantic segmentation branch")
args.add_argument("--num_classes", type=int, default=21, help="Number of classes in the semantic segmentation")

# Optimization
args.add_argument("--batch_size", type=int, default=2, help="Batch size")
args.add_argument("--desired_batch_size", type=int, default=None, help="Desired batch size")
args.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
args.add_argument("--num_steps", type=int, default=None, help="If set, the training session will run for a number of epochs that correspond \
                    to the number of steps and the batch size.")
args.add_argument("--learning_rate", type=float, default=6e-05, help="Learning rate")
args.add_argument("--early_stopping_thresh", type=int, default=10, help="Number of epochs to wait before early stopping")
args.add_argument("--groupnorm_divisor", type=int, default=32, help="Divisor for the group normalization")
args.add_argument("--cuda_id", type=int, default=0, help="Which GPU to use")
args.add_argument("--distributed", action="store_false", help="If set, nn.Dataparallel is used")

# Optimizer and scheduler
args.add_argument("--div_factor", type=float, default=2, help="Divisor for the OneCyclicLR scheduler")

# Model
args.add_argument("--input_channels", type=int, default=7, help="Number of input channels")
args.add_argument("--rgb_only", action="store_true", help="If set, only the RGB channels are used as input")
args.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
args.add_argument("--model", type=str, default="base", help="Model to use")
args.add_argument("--load_ckpt", action="store_true", help="If set, the checkpoint is loaded")

# Hyperparameters tuning
args.add_argument("--random_search_dataset_size", type=int, default=500, help="Size of the dataset used for the random search")
args.add_argument("--random_search_num_trials", type=int, default=50, help="Number of trials for the random search")

# Output
args.add_argument("--output_dir", type=str, default="Output", help="Output directory")
args.add_argument("--quasi_lidar_gt_path", type=str, default="Output/quasi_lidar_gt", help="Depth quasi gt folder")
args.add_argument("--save_model", action="store_true", help="If set, the checkpoint and tensorboard logs are saved")
args.add_argument("--arch_name", type=str, default="Transformer", help="Name of the architecture")
args.add_argument("--run_name", type=str, default='current', help="Name of the run")
args.add_argument("--run_mode", type=str, default='train', help="Mode of the run")
args.add_argument("--summary", action="store_true", help="If set, the summary of the model is printed")

# Visualization
args.add_argument("--num_vis", type=int, default=25, help="Number of samples to visualize")

args = args.parse_args()
args = edict(vars(args))


# print(args.load_ckpt);exit()


############################ Manual settings ############################

# Uncomment the following section, in order to manually set the hyperparameters and relevant paths (Recommended).
# The current settings are set as follows:
# 1. exp_index = 0: Base (only RGB)
# 2. exp_index = 1: Base (RGB + Radar)
# 3. exp_index = 2: Supervised semantic segmentation
# 4. exp_index = 3: Unsupervised semantic segmentation
# 5. exp_index = 4: Supervised + Unsupervised semantic segmentation
# 6. exp_index = 5: Supervised + Unsupervised semantic segmentation (Only RGB)



exp_index = 0
args.load_ckpt = False
args.distributed = False 
args.run_mode = ["train", "test"][0]
args.mini_dataset = [False, True][0]
args.train_val_split = [17902, 2237]
args.split = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/halperin-dan/src/data/original_new_split.npy"
args.dataset_path = "/home/ubuntu/Dominik/Dan_Halperin/thesis_data"
# args.split ="/home/ubuntu/Documents/students/DanHal/local_split.npy"
           

args.arch_name = ["Debug", "Transformer"][1]  # 
args.run_name = "AE"
args.model = ["stage_0", "stage_1", "stage_2", "stage_3", "stage_4"][exp_index]


args.input_channels =     [3, 3, 3, 3, 3][exp_index]
args.batch_size =         [8, 8, 8, 8, 4][exp_index]
args.desired_batch_size = [8, 8,8, 8, 4][exp_index]

assert args.desired_batch_size % args.batch_size == 0
args.update_interval = args.desired_batch_size // args.batch_size

############ Values for the training session ############
# Usually, "from_scratch" is for a training from scratch, and the second option is when you want continue a training mid training.

args.num_workers = 12
args.from_scratch = True
if args.from_scratch:
    # args.num_steps = 250000
    args.num_epochs = 35 # Use that when you want to specify the number of epochs, and not the number of steps.
    args.pct_start = 6
    args.lr_scheduler_values = [2e-4, 6.5e-5, 8.5e-6] # learning rate, max_lr, min_lr
    # args.lr_scheduler_values = [6.5e-5, 2e-4, 8e-6] # learning rate, max_lr, min_lr
    # args.lr_scheduler_values = [5e-5, 3e-4, 1e-5] # learning rate, max_lr, min_lr
    args.lr_scheduler_values_decoder = args.lr_scheduler_values
    args.learning_rate, args.max_lr, args.min_lr = args.lr_scheduler_values
    args.div_factor = args.learning_rate / args.max_lr
    args.final_div_factor = args.max_lr / args.min_lr

    args.seg_scheduler_values = [1, 1e-2, 3] # Desired coeff, max_coeff, min_coeff
    args.sky_scheduler_values = [1e-1, 3e-1, 5e-3] # Desired coeff, max_coeff, min_coeff
    args.enhanced_scheduler_values = [1, 5e-2, 1] # Desired coeff, max_coeff, min_coeff
    # args.edge_scheduler_values = [2/16, 1, 1e-10] # Desired coeff, max_coeff, min_coeff
    args.edge_scheduler_values = [1, 1e-1, 1] # Desired coeff, max_coeff, min_coeff
    args.dist_scheduler_values = [1, 5e-3, 1]# Desired coeff, max_coeff, min_coeff
    # args.dist_scheduler_values = [0.1, 0.1, 0.1]# Desired coeff, max_coeff, min_coeff
else:
    # args.num_steps = 75000
    args.num_epochs = 20 # Use that when you want to specify the number of epochs, and not the number of steps.
    args.pct_start = 4
    
    args.lr_scheduler_values = [1e-4, 2e-5, 8.5e-06] # learning rate, max_lr, min_lr
    # args.lr_scheduler_values = [2e-4, 5e-4, 8e-6] # learning rate, max_lr, min_lr
    args.lr_scheduler_values_decoder = args.lr_scheduler_values
    args.learning_rate, args.max_lr, args.min_lr = args.lr_scheduler_values
    args.div_factor = args.learning_rate / args.max_lr
    args.final_div_factor = args.max_lr / args.min_lr

    args.seg_scheduler_values = [1, 1e-2, 3] # Desired coeff, max_coeff, min_coeff
    args.sky_scheduler_values = [2e-1, 3e-1, 5e-3] # Desired coeff, max_coeff, min_coeff
    # args.sky_scheduler_values = [6e-1, 1e-1, 6e-2] # Desired coeff, max_coeff, min_coeff
    args.enhanced_scheduler_values = [1, 5e-3, 1] # Desired coeff, max_coeff, min_coeff
    # args.edge_scheduler_values = [2/16, 1, 1e-10] # Desired hcoeff, max_coeff, min_coeff
    args.edge_scheduler_values = [1, 1e-1, 1e-1] # Desired coeff, max_coeff, min_coeff
    # args.dist_scheduler_values = [0.3, 0.1, 0.5]# Desired coeff, max_coeff, min_coeff
    args.dist_scheduler_values = [1, 1, 1]

args.overfit = False
args.save_model =  False  # Otherwise, it will save anything to disk.
args.ask_to_load = False  # If you want to load a model from disk, otherwise, it will skip this step automatically.


args.checkpoint = "/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth/halperin-dan/mlt_epoch_35_best_RMSE_4.2187502_aux_3.545e+00.pth"


##########################################################################################################################################################
# args.num_bins = [64, 128, 256]
# args.num_bins = [80, 96, 128]
args.num_bins = [32, 64, 100]
# args.loss_weights = [6, 0.75, 0.0005, 0.0005] # Depth, Enhance, Sky, seg
args.sky_thresh = 8e-03
args.false_sky_thresh = 1e-02
args.dropout_p = 0.1
args.weight_decay = 0.23
args.rgb = 3
args.depth = 1
args.rgb_flag = False
args.rot_coeff = 0.95
args.depth_thresh = 1 / args.max_depth

args.cuda_id = [0, 0, 0, 0, 0, 0, 0, 0, 0][exp_index]

args.stop_after = args.num_epochs # Skip the very last epochs, where the learning rate it too small.
args.rgb_only = [True, False, False, False, False, True, False, True, False][exp_index]
args.early_stopping_thresh = args.num_epochs
args.guidance_input_channels = 6

args.rasa_cfg = edict(
                    atrous_rates= None, # [1,3,5], # None, [1,3,5]
                    act_layer= 'nn.SiLU(True)',
                    init= 'kaiming',
                    r_num =2 ,
        )

args.vae_mode = False

############################ Set default values, if not set otherwise ############################
 
args.hashtags_prefix = "####################################"
if args.desired_batch_size is None:
    args.desired_batch_size = args.batch_size
else:
    assert args.desired_batch_size % args.batch_size == 0, "Desired batch size must be a multiple of batch size"
    
print(f"{args.hashtags_prefix} Run mode is {args.run_mode}")
    
args.update_interval = args.desired_batch_size // args.batch_size

if args.mini_dataset:
    assert args.run_mode == "test", "Mini dataset is only available for testing"
    
args.train_val_split = [0, 0] if args.mini_dataset else [17902, 2237]
    
args.num_samples = sum(args.train_val_split)
    
if args.num_steps is not None:
    args.num_epochs = int(np.ceil(args.num_steps * args.update_interval / (args.train_val_split[0] / args.batch_size)))
    print(f"{args.hashtags_prefix} Auto set number of epochs: {args.num_epochs}")
    
if args.checkpoint is not None:
    print(f"{args.hashtags_prefix} Auto set checkpoint to {args.checkpoint}")

args.split = Path(args.split)
assert args.split.exists(), f"Split file {args.split} does not exist"

if args.output_dir is not None:
    args.output_dir = Path("/home/ubuntu/DanHal/Dan_Halperin/CamRaDepth") / args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{args.hashtags_prefix} Auto set output directory to {args.output_dir}")
    

args.supervised_seg = args.model in ["sup_unsup_seg", "sup_unsup_seg (rgb)", "supervised_seg"]
args.unsupervised_seg = args.model in ["sup_unsup_seg","sup_unsup_seg (rgb)", "unsupervised_seg"]
print(f"{args.hashtags_prefix} Auto set supervised_seg to {args.supervised_seg}")
print(f"{args.hashtags_prefix} Auto set unsupervised_seg to {args.unsupervised_seg}")
    

if args.model in ["base (rgb)", "sup_unsup_seg (rgb)"] or args.rgb_only:
    args.input_channels = 3
    print(f"{args.hashtags_prefix} Auto set input channels to 3 (RGB only)")
    assert args.input_channels == 3, "Input channels must be 3 for base_rgb model"
    
print(f"{args.hashtags_prefix} Auto set model to '{args.model}'")

if args.save_model:
    assert args.run_name is not None, "If save_model is set, run_name must be set as well, to differentiate between different runs"

assert args.run_mode in ["train", "test"], "Run mode must be either train or test"

if args.checkpoint is not None and args.run_mode == "test":
    args.load_ckpt = True

if args.ask_to_load:
    if args.checkpoint is not None and not args.load_ckpt and args.run_mode == "train":
        user_input = input(f"{args.hashtags_prefix} Would you like to load the checkpoint file? [y/Y] for Yes, any other value for No. \n{args.hashtags_prefix} Answer: ")
        user_input = user_input.lower()
        args.load_ckpt = user_input in ["y", "yes"]
else:
    args.load_ckpt = False

print(f"{args.hashtags_prefix} save_model is set to {args.save_model}")
print(f"{args.hashtags_prefix} num workers is set to {args.num_workers}")



# Model arguments
args.transformer_depths =  {"0": (2, 2, 2, 2), "1": (2, 2, 2, 2), "1.5": (2, 2, 3, 3), 
                           "2": (3, 3, 6, 3), "2.5": (3, 4, 7, 3), "3": (3, 6, 8, 3),
                           "3.5": (3, 8, 10, 3),  "4": (3, 8, 12, 5), "5": (3, 10, 16, 5)}

# Dataloader args
args.sparse_lidar = False

args.filtered_radar = False
args.lidar_ratio = [0.75, 0.25]
args.sparse_depth_uv = True
args.im_uv = False
args.rad_vel = True
args.radar_uv = False
args.gt_uv = False


'''

'''

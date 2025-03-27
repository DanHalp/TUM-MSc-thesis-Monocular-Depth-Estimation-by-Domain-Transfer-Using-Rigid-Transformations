from pathlib import Path
import shutil
import sys, os
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

from util_moduls.args import args
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_c=192, ratio=8):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(in_c, in_c // ratio, 1, 1, 0),
            nn.Conv2d(in_c // 8, in_c, 1, 1, 0),
            nn.Sigmoid()
           
        )
        self.apply(weights_init_kaiming)

    def forward(self, x, ref):
        weights = self.fc(ref)
        return weights * x, weights


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, in_c=192, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 5, 7), 'Kernel size must be 3, 5 or 7'
        padding = kernel_size // 2

        self.merge = nn.Conv2d(2, 1, kernel_size, 1, padding)
        
        self.apply(weights_init_kaiming)


    def forward(self, x, ref):
        
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.max(ref, dim=1, keepdim=True)[0]
        cat = torch.cat([x_avg, x_max], dim=1)
        weights = self.merge(cat)
        weights = torch.sigmoid(weights)
    
        return weights * x, weights


# CBAM Block (Combining Channel and Spatial Attention)
class CBAM(nn.Module):
    def __init__(self, in_c, kernel_size=7):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_c, ratio=8)
        self.spatial_attention = SpatialAttention(in_c, kernel_size=kernel_size)

    def forward(self, x, y=None, return_weights=False, with_channels_attn=True):
        out = x
        channel_weights = None

        if with_channels_attn:
            out, channel_weights = self.channel_attention(out, x if y is None else y)
                    
        out, spatial_weights = self.spatial_attention(out,  out if y is None else y)
        if return_weights:
            return out, spatial_weights, channel_weights    
        return out + x

class Reltaive_Representation(nn.Module):
    
    def __init__(self, dim=192,num_anchors=192):
        super().__init__()
        self.num_anchors = num_anchors 
        
        self.anchors = nn.Parameter(torch.randn(dim, num_anchors), requires_grad=True)
      
        
    def forward(self, x, detach=False):

        x = F.normalize(x, p=2, dim=-1)
        a = F.normalize(self.anchors, p=2, dim=-2)
        
        if detach:
            a = a.detach()
        x = torch.matmul(x, a)
    
        return x
        
        

class SphericalGrid(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.register_buffer('grid', self.create_spherical_grid(latent_size))

    def create_spherical_grid(self, num_points):
        indices = torch.arange(0, num_points, dtype=torch.float32) + 0.5
        phi = torch.acos(1 - 2 * indices / num_points)
        theta = torch.pi * (1 + 5**0.5) * indices

        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)

        return nn.Parameter(torch.stack((x, y, z), dim=1), requires_grad=False)

    def forward(self, batch_size):
        return self.grid.repeat(batch_size, 1)
        

class  LinearResBlock(nn.Module):
    """
    A short dense blocks, with reducing channels as it goes deeper.
    """
    def __init__(self, in_d, out_d, mid_d=32, num_layers=3, as_final_block=False):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        reducuction_coeff = 1 / (num_layers)
        self.inter_out = [mid_d * (1 - (i * reducuction_coeff)) for i in range(num_layers - 1, -1 ,-1)]
        
        inp = int(in_d)
        
        for i in range(num_layers):
            out = int(self.inter_out[i])
            self.layers.append(
                self.create_linear_block(inp, out, as_final_block=False)
            )
            inp += out
        
        self.final = self.create_linear_block(inp, out_d, as_final_block=as_final_block)
        
        self.final_dim = inp
      

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat((x, out), dim=-1)
        x = self.final(x)
        return x
    
    def create_linear_block(self, in_d, out_d, as_final_block=False):
        
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_d, out_d))
        if not as_final_block:
            layers.append(nn.GELU())
        return nn.Sequential(*layers)


    
class FoldUnfold(nn.Module):
    
    def __init__(self,num_points=240, source_dim=64, target_dim=64, inter_dim=63, num_groups=16):
        super().__init__()
        
        self.pc_dim = inter_dim
        
        
    def unfold(self, x):
        x = x.unfold(2, 3, 3).permute(0, 2, 1, 3)
        return x
    
    def fold(self, x, b, c, d):
        x = x.permute(0, 2, 1, 3).reshape(b, c, d)
        return x

def rotmat_from_matrix(K, force_rotation=True):
    # U, s, V =torch.linalg.svd(K.float())
    U, s, V = torch.svd(K.float())
  
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], dtype=torch.float32).unsqueeze(0).to(K.device)
    Z = Z.repeat(U.shape[0],1,1).float()
    if force_rotation:
        Z[:,-1, -1] *= torch.det(torch.matmul(U, V.permute(0,2,1)).float())
  
    # Construct R.
    R = torch.matmul(V, torch.matmul(Z, U.permute(0,2,1))).float()
    R = torch.matmul(R, Z)
    
    return R

def procrustes_align(S1, S2, only_rot=False):
    
    # NOTE!!!!!: S1 and S2 must be of shape (batch_size, N, D), as the algorithm expects the transposed matrices, but this is dealt with inside the function.
    
    orig_dtype = S1.dtype
    
    
    S1 = S1.permute(0,2,1)
    S2 = S2.permute(0,2,1)
    
    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    
    mu2 = S2.mean(axis=-1, keepdims=True)
  
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    # var1 = torch.sum(X1**2, dim=1).sum(dim=-1)
    # var1[var1 == 0] = 1e-8

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0,2,1)).float()

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    # R = special_procrustes(K)
    R = rotmat_from_matrix(K)

    
    if only_rot:
        return R.permute(0,2,1).to(orig_dtype)

    # scale = 1
    scale = (X2.norm(dim=2) / (X1.norm(dim=2) + 1e-8)).unsqueeze(2)
    t = mu2 - (scale * (torch.matmul(R, mu1))).float()

    # 7. Error:

    S1_hat = scale * torch.matmul(R, S1) + t

    S1_hat = S1_hat.permute(0,2,1).to(orig_dtype)
    R = R.permute(0,2,1).to(orig_dtype)
    t = t.permute(0,2,1).to(orig_dtype)
    scale = scale.permute(0,2,1).to(orig_dtype)

    return S1_hat, R, t, scale


class Seg_Block(nn.Module):
    """
    Creates a segmnetation map out of an input block of logits.
    """
    def __init__(self, in_c, num_classes=21):
        super().__init__()
        self.conv = ConvLayer(in_c, num_classes, kernel_size=3, stride=1, padding=1, activation_flag=False, norm_flag=False)
        self.seg_num_classes = num_classes
    
    def forward(self, x):
        seg_logits = self.conv(x)
        seg_map =  torch.argmax(F.softmax(seg_logits, dim=1), dim=1, keepdim=True)
        seg_map = seg_map / self.seg_num_classes
        
        return seg_map, seg_logits

class DenseChain(nn.Module):
    
    def __init__(self, in_c=5, out_c=128, num_blocks=2, num_layers=2, as_final_block=False, norm_flag=True, activation_flag=True, dropout_p=0):
        super().__init__()
        
        assert num_blocks > 0, "num_blocks must be greater than 1"
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks - 1):
            self.blocks.append(DenseBlock(in_c, out_c, num_layers=num_layers, activation_flag=activation_flag))
            self.blocks.append(nn.Dropout2d(dropout_p))
            in_c = out_c
            
        self.blocks.append(DenseBlock(in_c, out_c, as_final_block=as_final_block, num_layers=num_layers, activation_flag=activation_flag))
        self.blocks = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        return self.blocks(x)

class DenseBlock(nn.Module):
    def __init__(self, in_c, out_c, num_layers=2, k=3, s=1, p=1, as_final_block=False, activation_flag=True, dropout_p=0):
        super().__init__()
        
        mid_c = max(in_c, out_c) # // 2
        
        self.layers = nn.ModuleList([ConvLayer(in_c, mid_c, k, s, p, norm_flag=as_final_block, activation_flag=activation_flag)])
        
        for idx in range(num_layers-2):
            in_c += mid_c
            self.layers.append(ConvLayer(in_c, mid_c, norm_flag=False))
            
        in_c += mid_c
        self.layers.append(ConvLayer(in_c, out_c, as_final_block=as_final_block))
        self.dropout = nn.Dropout2d(dropout_p)
        
           
    def forward(self, x):
        for idx in range(len(self.layers) - 1):
            layer = self.layers[idx]
            x = torch.cat((x, layer(self.dropout(x))), dim=1)
        x = self.layers[-1](self.dropout(x))
        return x
    
    
class Res_Chain(nn.Module):
        
    def __init__(self, in_dim, num_layers=3, as_final_block=False):
        super().__init__()
    
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(Res_Block(in_dim))
        self.layers.append(Res_Block(in_dim, as_final_block=as_final_block))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    
class Res_Block(nn.Module):
    def __init__(self, in_c, k=3, s=1, p=1, norm_flag=True, activation_flag=True, as_final_block=False):
        super().__init__()
        
        self.as_final_block = as_final_block
        
        self.conv = ConvLayer(in_c, in_c, k, s, p, activation_flag=False)
        if self.as_final_block:
            self.conv_2 = ConvLayer(in_c, in_c, 1, 1, 0, as_final_block=True)

        self.apply(weights_init_kaiming)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = F.gelu(x + identity)
        if self.as_final_block:
            x = self.conv_2(x)
        
        return x
      
 
class  ConvLayer(nn.Module):
    """
    A simple convolution layer with a norm layer and a non-linear activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation="gelu", bias=None, conv_groups=None, norm_divisor=None, dropout_p=0,
                 norm_layer = "groupnorm", maxpool_flag=False, maxpool_first=False, norm_flag=True, activation_flag=True, as_final_block=False, **kwargs):
        super().__init__()
        
        bias = not norm_flag if bias is None else bias
        self.norm_flag = norm_flag
        self.activation_flag = activation_flag
        self.layers = nn.ModuleList()
        
        n_groups = how_many_groups(out_channels)
        
        self.layers.append(nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity())
        self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                        stride=stride, padding=padding, bias=bias, groups=conv_groups if conv_groups is not None else 1)) # TODO: remove bias
        
        
        if as_final_block:
            self.norm_flag = False
            self.activation_flag = False
       
        if self.norm_flag:
            assert norm_layer in ["batchnorm", "groupnorm", "instancenorm", "layernorm", "pixelnorm"], "Norm layer must be one of: batchnorm, groupnorm, instancenorm, layernorm, pixelnorm"
            if norm_layer == "groupnorm":
                norm = nn.GroupNorm(n_groups, out_channels)
            elif norm_layer == "instancenorm":
                norm = nn.InstanceNorm2d(out_channels, affine=True)
            elif norm_layer == "batchnorm":
                norm = nn.BatchNorm2d(out_channels)
            else:
                raise NotImplementedError("Norm layer not implemented")
            self.layers.append(norm)
            
        if self.activation_flag:
            assert activation in ["elu", "relu", "leakyrelu", "gelu", "selu", "silu", "sigmoid", "swish", "tanh"], f"Activation {activation}  must be one of: elu, relu, gelu"
            self.activation = {"elu": nn.ELU, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU,"gelu": nn.GELU, "selu": nn.SELU,
                               "silu": nn.SiLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}[activation]()
            
            self.layers.append(self.activation)
        
        assert not (maxpool_first and not maxpool_flag), "maxpool_first can only be True if maxpool_flag is True"
        self.maxpool_bool = maxpool_flag
        self.maxpool_first = maxpool_first
        
        if maxpool_flag:
            self.maxpool = nn.MaxPool2d(2)
            
        self.layers = nn.Sequential(*self.layers)
        
        self.apply(weights_init_kaiming)

    def forward(self, x):
        if self.maxpool_bool and self.maxpool_first:
            x = self.maxpool(x)
        x = self.layers(x)
        if self.maxpool_bool and not self.maxpool_first:
            x = self.maxpool(x)
        return x

class Depth_Activation(nn.Module):
    """
    Create a depth map, by using a sigmoid activation, and then a linear convolution, for fine scaling and stretching.
    """
    def __init__(self, input_c, output):
        super().__init__()
        self.conv_1 = DenseBlock(input_c, input_c)
        self.conv_2 = ConvLayer(input_c, output, kernel_size=1, padding=0, as_final_block=True)
        
    def forward(self, x, eps=1e-6):
        # x = self.conv_1(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return  F.sigmoid(x), x



def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def weights_init_kaiming(m, activation="gelu"):
    if isinstance(m, nn.Conv2d):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        if activation == "selu":
            # Apply LeCun initialization to convolutional layer weights
            nn.init.normal_(m.weight, mean=0, std=1.0 / fan_in)
            if m.bias is not None:
                m.bias.data.zero_() 
        elif activation in ["relu", "gelu", "silu"]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif activation in ["sigmoid", "tanh"]:
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(activation))
            if m.bias is not None:
                m.bias.data.zero_()
                
    
    if isinstance(m, nn.ConvTranspose2d):
        
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        if activation == "selu":
            # Apply LeCun initialization to convolutional layer weights
            nn.init.normal_(m.weight, mean=0, std=1.0 / fan_in)
            if m.bias is not None:
                m.bias.data.zero_() 
        elif activation in ["relu", "gelu", "silu"]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif activation in ["sigmoid", "tanh"]:
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(activation))
            if m.bias is not None:
                m.bias.data.zero_()
    
    if isinstance(m, nn.Linear):
        fan_in = m.in_features
        if activation == "selu":
            # Apply LeCun initialization to linear layer weights
            nn.init.normal_(m.weight, mean=0, std=1.0 / fan_in)
            if m.bias is not None:
                m.bias.data.zero_() 
        elif activation in ["relu", "gelu", "silu"]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif activation in ["sigmoid", "tanh"]:
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain(activation))
            if m.bias is not None:
                m.bias.data.zero_()
   
    elif isinstance(m, (nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_kaiming_leaky(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def initialize_weights(model):
    """
    Initializes the weights of a PyTorch model according to the He et al. (2015) initialization scheme.

    Args:
        model (torch.nn.Module): The PyTorch model to initialize.

    Returns:
        None
    """

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        
        
################################ Helper Functions ################################

def freeze_and_unfreeze_model(model, freeze=True):
    """
    Freeze or unfreeze the weights of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to freeze or unfreeze.
        freeze (bool, optional): Whether to freeze or unfreeze the weights. Defaults to True.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = not freeze
        

def how_many_groups(num_c, divisor=args.groupnorm_divisor):
    """
    Calculate the number of groups for GroupNorm based on the input number of channels.

    Args:
        num_c (int): The number of input channels.
        divisor (int): The divisor to calculate the number of groups (default is args.groupnorm_divisor).

    Returns:
        int: The number of groups for GroupNorm.
    """
    if num_c // divisor > 0 and num_c % divisor == 0:
        n_groups = num_c // divisor
    else:
        div = args.num_c if num_c is None else num_c
        while num_c % div != 0:
            div //= 2
        n_groups = num_c // div
    return n_groups
    


def create_tqdm_bar(iterable, desc):
    """
    Creates a nice looking progress bar.
    """
    return tqdm(enumerate(iterable),total=len(iterable), ncols=240, desc=desc)

def load_without_module_state_dict(model, state_dict):
    """
    Load state dict without the 'module' prefix
    """
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

def load_checkpoint_with_shape_match(model, checkpoint_dict):
    """
    1. Load state dict without the 'module' prefix (If trained earlier with some distributed version)
    2. Load only matching layers from the checkpoint, while notifying the user about the mismatching layers.
    3. If a layer's weights have a shape mismatch, load as many weights as possible and leave the remaining weights uninitialized.
    """
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint_dict.items()}
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for key in model_state_dict.keys():
        if key in checkpoint:
            if checkpoint[key].shape == model_state_dict[key].shape:
                new_state_dict[key] = checkpoint[key]
            else:
                try:
                    checkpoint_shape = checkpoint[key].shape
                    model_shape = model_state_dict[key].shape
                    print(f"{args.hashtags_prefix} Shape mismatch in key: {key}", "checkpoint", checkpoint_shape, "new_model", model_shape)
                    # new_state_dict[key] = model_state_dict[key]
                    mismatched_shape = [min(c1, c2) for c1, c2 in zip(checkpoint_shape, model_shape)]
                    min_slices = [slice(None, size) for size in mismatched_shape]          
                    new_state_dict[key] = model_state_dict[key]
                    new_state_dict[key][tuple(min_slices)] = checkpoint[key][tuple(min_slices)]    
                except:
                    pass          
        else:
            print(f"{args.hashtags_prefix} Key not in checkpoint: ", key)
            new_state_dict[key] = model_state_dict[key]

    for key in checkpoint.keys():
        if key not in model_state_dict:
            print(f"{args.hashtags_prefix} Key in checkpoint, but not in model: ", key)
    model.load_state_dict(new_state_dict, strict=True)


def load_mean_var(mean_path, var_path):
    """
    Load mean and variance tensors from a given path.
    """
    mean = torch.load(mean_path)
    var = torch.load(var_path)
    return mean, var
    


def adjust_loss(x, y, nominator=1):
    
    return (x * nominator) / (y + 1e-5) 

def find_coeff(x, y, nominator=1):
    """
    Find the coefficient to multipy y by, in order to make it's exponent smaller than x's exponent by at least 1.
    """
    
    x_exp = find_exponent_of_fraction(x)
    y_exp = find_exponent_of_fraction(y)
    
    
    if y_exp >= x_exp:
        exp = nominator / (10 ** ((y_exp - x_exp)))
        if y * exp > x:
            return adjust_loss(x, y, nominator)
        return  exp
    return nominator
    

def find_exponent_of_fraction(fraction):
    
    if fraction == 0:
        return 0

    abs_fraction = np.abs(fraction)
    exponent = np.floor(np.log10(abs_fraction))
    return exponent

def standraize(numbers, new_min=0, new_max=1):
    current_min = torch.min(numbers)
    current_max = torch.max(numbers)
    normalized = ((numbers - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min
    return normalized

def save_files(output_path):
    """
    If you decide to use this functionality, you'll have to set the relevant paths first.
    Saving locally the relevant files of each run, which is quite hard to keep track with git. If you want to reverse to a better run,
    now you could simply access the relevant files and run them.
    """
    
    def copy_dir(src_dir, dst_dir):
          
        # Traverse the directory tree using os.walk()
        for dirpath, dirnames, filenames in os.walk(src_dir):
            # Create the corresponding subdirectories in the destination directory
            rel_dir = os.path.relpath(dirpath, src_dir)
            dst_subdir = os.path.join(dst_dir, rel_dir) if rel_dir != "." else dst_dir
            os.makedirs(dst_subdir, exist_ok=True)

            # Copy each file to the destination directory
            for filename in filenames:
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(dst_subdir, filename)
                shutil.copyfile(src_file, dst_file)
          
    
    project_files_path = Path(output_path) / "project_files"
    os.makedirs(project_files_path, exist_ok=True)
    
    this_dir = os.path.dirname(__file__)
        
    
    src_dir = Path(this_dir) / "../../src"
    dst_dir = project_files_path / "src"
    
    copy_dir(src_dir, dst_dir)
    

    
def save_point_clouds_as_obj(point_clouds, folder_path, file_name):
    """
    Take a batch of point clouds of shape (B, C, 3), create a folder, and save each point cloud as a .obj file
    """
    os.makedirs(folder_path, exist_ok=True)
    for i in range(point_clouds.shape[0]):
        point_cloud = point_clouds[i].detach().cpu().numpy()
        point_cloud = point_cloud.astype(np.float32)
        file_path = os.path.join(folder_path, f"{file_name}_{i}.obj")
        with open(file_path, 'w') as obj_file:
            for point in point_cloud:
                obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")

        
        break
    
    
    
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
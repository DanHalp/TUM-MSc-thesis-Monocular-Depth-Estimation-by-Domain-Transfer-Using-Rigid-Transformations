
import torch
import torch.nn as nn
from src.util_moduls.utils_functions import how_many_groups, weights_init_kaiming, ConvLayer
import numpy as np
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be divisible by 2"
        while in_channels < reduction or in_channels % reduction != 0:
            reduction = in_channels // 2
        
        
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x, dim=(2, 3), keepdim=True)
        out = self.fc1(out)
        out = F.gelu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

class OptimizedResBlock(nn.Module):
    def __init__(self, in_c, stride=1, reduction=16, ):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv(in_c, in_c, stride=stride)
        self.gn1 = nn.GroupNorm(how_many_groups(in_c), in_c)
        self.gelu = nn.GELU()
        self.conv2 = DepthwiseSeparableConv(in_c, in_c)
        self.gn2 = nn.GroupNorm(how_many_groups(in_c), in_c)
        self.se = SEBlock(in_c, reduction=reduction)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)

        out += identity
        out = self.gelu(out)

        return out
    
    
class ResChain(nn.Module):
        
    def __init__(self, in_dim, num_layers=2, as_final_block=False, dropout_p=0):
        super().__init__()
    
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(ResBlock(in_dim, dropout_p=dropout_p))
        self.layers.append(ResBlock(in_dim, as_final_block=as_final_block))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c, as_final_block=False, dropout_p=0):
        super().__init__()
        
        self.as_final_block = as_final_block
        
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.norm = nn.GroupNorm(how_many_groups(in_c), in_c)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        if self.as_final_block:
            self.conv_2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

        self.apply(weights_init_kaiming)

    def forward(self, x):
        identity = x
        x = self.conv(self.dropout(x))
        x = self.norm(x)
        x = F.gelu(x + identity)
        if self.as_final_block:
            x = self.conv_2(x)
        return x    
    
class ResBlockBottleNeck(nn.Module):
    
    def __init__(self, in_c, reduce_rate=4, as_final_block=False, dropout_p=0):
        super().__init__()
        
        self.as_final_block = as_final_block
        
        self.inner_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c // reduce_rate, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(in_c // reduce_rate, in_c // reduce_rate, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(in_c // reduce_rate, in_c, 1, 1, 0),
        )
        self.norm = nn.GroupNorm(how_many_groups(in_c), in_c)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        
        if self.as_final_block:
            self.conv_2 = nn.Conv2d(in_c, in_c, 3, 1, 1)

        self.apply(weights_init_kaiming)
        
    def forward(self, x):
        identity = x
        x = self.inner_conv(self.dropout(x))
        x = F.gelu(self.norm(x + identity))
        if self.as_final_block:
            x = self.conv_2(x)
        return x
    

    
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
    def __init__(self, in_c, out_c, mid_c=48, num_layers=2, as_final_block=False, dropout_p=0):
        super().__init__()
        
        mid_c = min(mid_c, in_c)
        # mid_c = min(in_c, out_c)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers-1):
            self.layers.append(Bottleneck(in_c, mid_c))
            in_c += mid_c

        self.layers.append(Bottleneck(in_c, out_c, as_final_block=as_final_block))
        self.dropout = nn.Dropout2d(dropout_p) 
        
           
    def forward(self, x):
        for idx in range(len(self.layers) - 1):
            layer = self.layers[idx]
            x = torch.cat((x, layer(self.dropout(x))), dim=1)
        x = self.layers[-1](self.dropout(x))
        return x
    
class Bottleneck(nn.Module):
    
    def  __init__(self, in_c, out_c, ratio=2, as_final_block=False):
        super().__init__()
        
        if in_c > ratio * out_c:
            self.conv = nn.Sequential(
                ConvLayer(in_c, out_c * ratio, 1, 1, 0, norm_flag=False),
                ConvLayer(out_c * ratio, out_c, 3, 1, 1, as_final_block=as_final_block),
            )
        else:
            self.conv = ConvLayer(in_c, out_c, 3, 1, 1, as_final_block=as_final_block)
            
    def forward(self, x):
        return self.conv(x)
import sys, os
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
import torch.nn.init as init
from util_moduls.args import args
from src.util_moduls.utils_functions import ConvLayer, DenseChain
from src.util_moduls.utils_functions import weights_init_kaiming, CBAM

from roma import rigid_vectors_registration, special_procrustes
import roma

class Tnet(nn.Module):
    
    def __init__(self, in_c=3, rot_dim=3, latent_dim = 64, dropout_p=0.0):
        super().__init__()
        
        self.rot_dim = rot_dim
        
        self.latent_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_c, self.latent_dim // 2),
            nn.GELU(),
            # nn.Dropout1d\\\(dropout_p),
            nn.Linear(self.latent_dim // 2, self.latent_dim),
            nn.GELU(),
            # nn.Dropout1d(dropout_p),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        if self.latent_dim > self.rot_dim ** 2:
            self.regress_R = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 2),
                nn.GELU(),
                nn.Linear(self.latent_dim // 2, self.rot_dim ** 2),
            )
            
        else:
            self.regress_R = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim * 2),
                nn.GELU(),
                nn.Linear(self.latent_dim * 2, self.rot_dim ** 2),
            )
            
    
        self.apply(weights_init_kaiming)
        
    def forward(self, x):
        shape = x.shape
        x = x + torch.randn_like(x) * 0.2
        x = self.mlp(x) 
        x = torch.max(x, dim=-2, keepdim=True)[0]
        
        R = self.regress_R(x)
        R = R.reshape(x.shape[0], -1, self.rot_dim, self.rot_dim) if len(shape) > 3 else R.reshape(x.shape[0], self.rot_dim, self.rot_dim)
        # R = self.relative_rep(R)
        return special_procrustes(R), R


class TnetConv(nn.Module):
    
    def __init__(self, in_c, rot_dim = 3):
        super().__init__()
        self.rot_dim = rot_dim
        self.layers = nn.Sequential(
            ConvLayer(in_c, in_c // 2, 1, 1, 0),
            nn.Dropout2d(args.dropout_p),
            ConvLayer(in_c // 2, in_c // 4, 1, 1, 0),
            nn.Dropout2d(args.dropout_p),
            ConvLayer(in_c // 4, rot_dim ** 2, 1, 1, 0),
        )
        
        self.numerical_anchors = NumericalAnchors()

    def forward(self, x):
        b, c, h, w = x.shape
        R = self.layers(x).permute(0, 2, 3, 1).reshape(b, -1, self.rot_dim ** 2)
  
        return special_procrustes(R.reshape(b, -1, self.rot_dim, self.rot_dim)) #, F.normalize(R, p=2, dim= -1)

class DepthAttention(nn.Module):
    
    def __init__(self, in_c, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_c, num_heads=num_heads, batch_first=True)
        
        self.q = nn.Linear(in_c, in_c)
        self.k = nn.Linear(in_c, in_c)
        self.v = nn.Linear(in_c, in_c)
        
        self.norm_1 = nn.LayerNorm(in_c)
        self.norm_2 = nn.LayerNorm(in_c)
        
        # Feedforward layer for post-attention
        self.fc_out = nn.Linear(in_c, in_c)
        self.anchors = None
    
    def forward(self, x, y):
        
        b, c, h, w = x.shape
        
        x = x.permute(2, 3, 0, 1).reshape(h*w*b, 1, c)
        y = y.permute(2, 3, 0, 1).reshape(h*w, b, c)
        
        a = self.anchors
        
        if a is None:
            a = y
        elif not self.training:
            a = self.anchors.to(x.device)
        else:
            a = torch.cat((y, a), dim=-2)
        
        a = a.repeat(b, 1, 1)
        
        q = self.q(self.norm_1(x))
        k = self.k(self.norm_1(a))
        v = self.v(self.norm_1(a))
        
        attn_output, _ = self.multihead_attn(q, k, v)
        # x = (attn_output + x)  # Reshape back to (B, C, N)
        # out = self.fc_out(self.norm_2(x)) + x  # Reshape back to (B, C, N)
        out = self.norm_2(self.fc_out(attn_output))
        
        
        out = out.reshape(h, w, b, c).permute(2, 3, 0, 1)
        
        if self.training:
            self.anchors = nn.Parameter(a[:h*w, :100].clone(), requires_grad=False).to(x.device)
        
        return out
        
        
class AE(nn.Module):
    def __init__(
            self,
            img_size=(384, 768),
            enc_embed_dims=[64, 128, 144, 192, 192, 512],
            rot_embed_dims=[64, 72, 96, 128, 192, 512],
            dec_embed_dims=[12, 96, 128, 144, 144, 512],
            input_channels=None,
            mode=args.model,
            **kwargs
    ):
    
        super().__init__()
        self.img_size = np.array(img_size)
        self.input_channels = input_channels if input_channels is not None else args.input_channels
        
        assert input_channels > 0, 'input_channels must be > 0'
        
        # Architecture
        self.num_layers = 4
        self.enc_embed_dims = enc_embed_dims[:self.num_layers]
        self.rot_embed_dims = rot_embed_dims[:self.num_layers]
        self.dec_embed_dims = dec_embed_dims[:self.num_layers]
        self.mode = mode
        self.depth_encoder = ResEncoder(embed_dims=[3] + self.enc_embed_dims, res_layers=[1, 1, 1, 1, 1][:self.num_layers], dropout_p=0.4)
        self.rgb_encoder = ResEncoder(embed_dims=[3] + self.enc_embed_dims, dropout_p=0, res_layers=[1, 2, 2, 2, 2][:self.num_layers])
        self.rgb_dec = DenseDecoder(embed_dims=[self.enc_embed_dims[-1]] + self.dec_embed_dims[::-1],  res_layers=[2, 2, 1, 1, 1][:self.num_layers])
        self.depth_dec = DenseDecoder(embed_dims=[self.enc_embed_dims[-1]] + self.dec_embed_dims[::-1], res_layers=[2, 2, 1, 1, 1][:self.num_layers])
        self.projection = AE_projector(in_c=self.enc_embed_dims[-1], mode=self.mode)  
        self.apply(weights_init_kaiming)
        
    def forward(self, x, gt_image=None, reduced=None):
        
        b = x.shape[0]
        
        if args.summary:
            self.train()
            gt_image = x[:, 0].unsqueeze(1)

        rgb_latent, rot_embedding = None, None
        rgb_enc_outs, depth_enc_outs = None, None
        depth_latent = None
        rgb_latent, outs = self.rgb_encoder(x)
        if gt_image is not None:
            depth_latent, _ = self.depth_encoder(gt_image + x)
        
        proj_dict = self.projection(rgb_latent=rgb_latent, depth_latent=depth_latent, rot_embedding=rot_embedding)
        
        depth_ret_dict = {}
        proj_ret_dict = {}
        proc_ret_dict = {}
        rgb_ret_dict = {}
        
       
        proj_latent = proj_dict["latents"]["proj"]
        proj_ret_dict = self.depth_dec(proj_latent)
        
        rgb_latent =  proj_dict["latents"]["rgb"]
        rgb_ret_dict = self.rgb_dec(rgb_latent)
        
        if not self.training:
            proc_ret_dict = self.depth_dec(proj_dict["latents"]["rgb"])

        if args.summary:
            return rgb_latent
        
        

        return {
            "projection": proj_dict,
            "depth_decoder_output": depth_ret_dict,
            "rgb_decoder_output": rgb_ret_dict,
            "proj_decoder_output": proj_ret_dict,
            "proc_decoder_output": proc_ret_dict,
            "rgb_enc_outs": rgb_enc_outs,
            "depth_enc_outs": depth_enc_outs
        }
        
    def reset_decoder(self):
        self.depth_dec = DenseDecoder(embed_dims=[self.enc_embed_dims[-1]] + self.dec_embed_dims[::-1], res_layers=[2, 2, 1, 1, 1][:self.num_layers])

 

class AE_projector(nn.Module):
    
    def __init__(self, in_c=192, latent_dim=192, mode=args.model):
        
        super().__init__()
        
        self.mode = mode
        self.rot_dim = 16
        self.full_dim = in_c
        self.latent_dim = latent_dim
      
        self.num_anchors = 300
        self.rgb_tnet = Tnet(in_c=self.rot_dim, rot_dim=self.rot_dim, latent_dim=128)

        
        
      
        self.apply(weights_init_kaiming)
      
    def forward(self, rgb_latent=None, depth_latent=None, rot_embedding=None):
        
        pred_dict = {}
        
        b, c, h, w = rgb_latent.shape
        if self.mode == "stage_0":
            
            rgb_pc = rgb_latent.permute(0, 2, 3, 1).reshape(b, h*w,-1,  self.rot_dim)
            depth_pc =  depth_latent.permute(0, 2, 3, 1).reshape(b, h*w, -1, self.rot_dim)
            
            R_rgb, R_rgb_raw = self.rgb_tnet(rgb_pc)
            R_depth = rigid_vectors_registration(rgb_pc, depth_pc).transpose(-2, -1)
            R_depth_raw = None
            
            proj_pc_proj = torch.matmul(rgb_pc, R_depth)
            rgb_pc_proj = torch.matmul(rgb_pc, R_rgb)
            rgb_latent_projected = rgb_pc_proj.reshape(b, h, w, c).permute(0, 3, 1, 2)
            depth_latent_projected = depth_pc.reshape(b, h, w, c).permute(0, 3, 1, 2)
            proj_latent_projected = proj_pc_proj.reshape(b, h, w, c).permute(0, 3, 1, 2)
            

            pred_dict = {
                "latents": {
                    "rgb": rgb_latent_projected, "depth": proj_latent_projected, "proj": proj_latent_projected
                },
            
                
                "R": {
                    "rgb": R_rgb, "depth": R_depth, #"rgb_raw": R_rgb_raw, "depth_raw": R_depth_raw
                },
 
            }

        return pred_dict
    
    def add_noise(self, data, noise_std=0.4):
        noise = torch.randn_like(data) * noise_std
        noisy_data = data + noise
        return torch.clamp(noisy_data, 0., 1.) 

class ResEncoder(nn.Module):
    
    def __init__(self,  embed_dims=[64, 128, 192, 240], res_layers=[1, 2, 2, 2], dropout_p=0):
        super().__init__()
        
        
        self.stages = nn.ModuleList()
        for idx in range(len(embed_dims) -1):
            in_block = nn.ModuleList()
            if idx == 0:
            
                in_block.append(ConvLayer(embed_dims[idx], embed_dims[idx + 1], kernel_size=7, stride=4, padding=3))
            else:
                in_block.append(ConvLayer(embed_dims[idx], embed_dims[idx + 1], kernel_size=3, stride=2, padding=1))
            
            for i in range(1):
                in_block.append(DenseChain(embed_dims[idx + 1], embed_dims[idx + 1], num_blocks=res_layers[idx], num_layers = 2))
                in_block.append(CBAM(embed_dims[idx + 1]))
                
         
            self.stages.append(nn.Sequential(*in_block))
        self.stages.append(nn.Conv2d(embed_dims[-1], embed_dims[-1], kernel_size=3, stride=1, padding=1))
        self.stages = nn.Sequential(*self.stages)
    def forward(self, x):
        outs = []
        
        for layer in self.stages:
            x = layer(x)
            outs.append(x)

        return x, outs

class simpleEncoder(nn.Module):
    
    def  __init__(self, embed_dims=[64, 128, 192, 240], dropout_p=0, double_downsample=False):
        super().__init__()
        self.stages = nn.ModuleList()
        for idx in range(len(embed_dims) -1):
            in_block = nn.ModuleList()
            if idx == 0 and double_downsample:
                    in_block.append(ConvLayer(embed_dims[idx], embed_dims[idx + 1], kernel_size=7, stride=4, padding=3))
            else:
                in_block.append(ConvLayer(embed_dims[idx], embed_dims[idx + 1], kernel_size=3, stride=2, padding=1))
            self.stages.append(nn.Sequential(*in_block))
        self.stages.append(nn.Conv2d(embed_dims[-1], embed_dims[-1], kernel_size=3, stride=1, padding=1))
           
            
    def forward(self, x):
        outs = []
        for layer in self.stages:
            x = layer(x)
            outs.append(x)
        return x, outs
    

class simpleDecoder(nn.Module):
    
    def  __init__(self, embed_dims=[64, 128, 192, 240], dropout_p=0, res_layers=[2, 2, 1, 1]):
        super().__init__()
        self.stages = nn.ModuleList()
        for idx in range(0, len(embed_dims) -1):
           
            upscale_factor = 2 if idx < len(embed_dims) - 2 else 4
            self.stages.append(nn.Sequential(
                ConvLayer(embed_dims[idx], embed_dims[idx + 1], kernel_size=3, stride=1, padding=1, activation="relu"),
                ConvLayer(embed_dims[idx + 1], embed_dims[idx + 1], kernel_size=3, stride=1, padding=1, activation="relu"),
                nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)
            
            ))
                
    
        self.deph_activation = ConvLayer(embed_dims[-1], 1, kernel_size=3, stride=1, padding=1, as_final_block=True)
            
    def forward(self, x):
        for layer in self.stages:
            x = layer(x)
        x = self.deph_activation(x)
        final_depth, logits = F.sigmoid(x), x
        return {
                "depth": {"final_depth": final_depth, "logits": logits}, 
        }
            

class DenseDecoder(nn.Module):
    def __init__(self, embed_dims=[64, 128, 192, 240], dropout_p=0, res_layers=[2, 2, 2, 1]):
        super().__init__()
        
        self.stages = nn.ModuleList()
        for idx in range(0, len(embed_dims) -1):
            in_block = nn.ModuleList()
            
            upscale_factor = 2 if idx < len(embed_dims) - 2 else 4
            in_block.append(nn.Sequential(
                nn.ConvTranspose2d(embed_dims[idx], embed_dims[idx + 1], kernel_size=upscale_factor, stride=upscale_factor),
                nn.GELU(),
            
            ))
                
            if idx < len(embed_dims) - 2:
                in_block.append(CBAM(embed_dims[idx + 1]))
                in_block.append(DenseChain(embed_dims[idx + 1], embed_dims[idx + 1], num_blocks=res_layers[idx], num_layers=2))
                

            in_block = nn.Sequential(*in_block)
            self.stages.append(in_block)
        self.stages = nn.Sequential(*self.stages)
        self.depth_activation = ConvLayer(embed_dims[-1], 1, kernel_size=1, stride=1, padding=0, as_final_block=True)
      
        
    def forward(self, x):
        outs = [x]
        for layer in self.stages:
            x = layer(x)
            outs.append(x)
            
        x = self.depth_activation(x)
        final_depth, logits = F.sigmoid(x), x
        return {
                "depth": {"final_depth": final_depth, "logits": logits}, 
                "outs": outs
        }
            

if __name__ == "__main__":
    args.summary = True 
    model = AE(input_channels=args.input_channels)
    # summary(model, (4, args.input_channels, 384, 768), verbose=1)
    
    # x, y = torch.randn(2, 3, 384, 768), torch.randn(2, 3, 384, 768)
    # model(x, y)
    
    # Pring each layer's parameters
    for name, param in model.named_parameters():
        if param.numel() > 100000:
            print(name, f": {param.numel():,}")
    
    print("#############################################################")
    total_params = sum(p.numel() for p in model.parameters())
    total_params_gb = total_params * 4 / (1024 ** 3)
    # Print the total params nicely, with commas:
    print(f"Total Parameters: {total_params:,}")
    
    
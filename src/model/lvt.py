import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math
import sys, os
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(__file__))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[3]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))
sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

from src.util_moduls.utils_functions import Depth_Activation, ConvLayer, weights_init_kaiming, Res_Chain, Res_Block, how_many_groups
from util_moduls.args import args
from easydict import EasyDict as edict

from simplified_attention import Attention_MaxPool, Block
__all__ = ['lvt']

class DenseAttention(nn.Module):
    
    def __init__(self, in_c, out_c, sr_ratio=1, num_heads=1, as_final_block=False):
        super().__init__()
        mid = in_c // 2
        self.conv_1 = ConvLayer(in_c, mid, 3, 1, 1)
        self.attn = Attention(mid, num_heads=num_heads, sr_ratio=sr_ratio, 
                              rasa_cfg=edict(
                                atrous_rates= None, # [1,3,5], # None, [1,3,5]
                                act_layer= 'nn.SiLU(True)',
                                init= 'kaiming',
                                r_num =2)
        )
        self.conv_2 = ConvLayer(in_c + mid, out_c, 3, 1, 1, as_final_block=as_final_block)
        
    def forward(self, x):
        
        a = self.attn(self.conv_1(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.cat([x, a], dim=1)
        x = self.conv_2(x)
        return x

class ConvAttention(nn.Module):
    
    def __init__(self, in_c, out_c, k=3, s=1, p=1, sr_ratio=1, num_heads=1, as_final_block=False):
        super().__init__()
        self.conv_1 = ConvLayer(in_c, out_c, kernel_size=k, stride=s, padding=p)
        self.attn = Transformer_block(out_c, num_heads=num_heads, sr_ratio=sr_ratio, mlp_ratio=3, with_depconv=True
                            #   rasa_cfg=edict(
                            #     atrous_rates= None, # [1,3,5], # None, [1,3,5]
                            #     act_layer= 'nn.SiLU(True)',
                            #     init= 'kaiming',
                            #     r_num =2 ,
                            #     )          
                           )
        self.conv_2 = ConvLayer(out_c, out_c, 3, 1, 1, as_final_block=as_final_block)
        
    def forward(self, x):
        
        x = self.conv_1(x).permute(0, 2, 3, 1)
        x = (x + self.attn(x)).permute(0, 3, 1, 2)
        x = x + self.conv_2(x)
        return x
        
        

class Comb_Attnetion(nn.Module):
    
    def __init__(self, dim, num_heads, sr_ratio=1):
        super().__init__()
        self.csa = CSA(dim, dim, num_heads=num_heads)
        self.rasa = Attention(dim, num_heads=num_heads, qkv_bias=True, sr_ratio=sr_ratio, 
                              rasa_cfg=edict(
                                atrous_rates= None, # [1,3,5], # None, [1,3,5]
                                act_layer= 'nn.SiLU(True)',
                                init= 'kaiming',
                                r_num =2 ,
                                )     
                              )
        # self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x + self.csa(x)
        x = x + self.rasa(x)
        return x
    

class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, 
                 dilation=[1,3,5], groups=1, bias=True, head_size=None,
                 act_layer='nn.SiLU(True)', init='kaiming'):
        super().__init__()
        assert in_planes%groups==0
        assert kernel_size==3, 'only support kernel size 3 now'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        
    
        self.layer_norm = nn.GroupNorm(num_groups=how_many_groups(out_planes), num_channels=out_planes)
        self.act = eval(act_layer)
        self.init = init
        self._initialize_weights()

    def _initialize_weights(self):
        if self.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
        if self.with_bias:
            if self.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.init == 'kaiming':
                bound = self.groups / (self.kernel_size**2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = 0
        for dil in self.dilation:
            output += self.act(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
                
            )
        self.layer_norm(output)
        return output

class CSA(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel_size=3, padding=1, stride=2,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5
        
        self.attn = nn.Linear(in_dim, kernel_size**4 * num_heads)
        self.attn_drop = nn.Dropout(attn_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        
        self.csa_group = 1
        assert out_dim % self.csa_group == 0
        self.weight = nn.Conv2d(
            self.kernel_size*self.kernel_size*out_dim, 
            self.kernel_size*self.kernel_size*out_dim, 
            1, 
            stride=1, padding=0, dilation=1, 
            groups=self.kernel_size*self.kernel_size*self.csa_group, 
            bias=qkv_bias,
        )
        assert qkv_bias == False
        fan_out = self.kernel_size*self.kernel_size*self.out_dim
        fan_out //= self.csa_group
        self.weight.weight.data.normal_(0, math.sqrt(2.0 / fan_out)) # init
        
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, H, W, C = x.shape
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        
        x = x.permute(0, 3, 1, 2)
        attn = self.pool(x).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4) # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        
        v = x.permute(0, 3, 1, 2) # B,C,H, W
        v = self.unfold(v).reshape(
            B, self.out_dim, self.kernel_size*self.kernel_size, h*w
        ).permute(0,3,2,1).reshape(B*h*w, self.kernel_size*self.kernel_size*self.out_dim, 1, 1)
        v = self.weight(v)
        v = v.reshape(B, h*w, self.kernel_size*self.kernel_size, self.num_heads, 
                      self.out_dim//self.num_heads).permute(0,3,1,2,4).contiguous() # B,H,N,kxk,C/H
        
        x = (attn @ v).permute(0, 1, 4, 3, 2)
        x = x.reshape(B, self.out_dim * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, 
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0., with_depconv=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_depconv = with_depconv
        
        if self.with_depconv:
            self.fc1 = nn.Conv2d(
                in_features, hidden_features, 1, stride=1, padding=0, dilation=1, 
                groups=1, bias=True,
            )
            self.depconv = nn.Conv2d(
                hidden_features, hidden_features, 3, stride=1, padding=1, dilation=1, 
                groups=hidden_features, bias=True,
            )
            self.act = act_layer()
            self.fc2 = nn.Conv2d(
                hidden_features, out_features, 1, stride=1, padding=0, dilation=1, 
                groups=1, bias=True,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        if self.with_depconv:
            x = x.permute(0,3,1,2).contiguous()
            x = self.fc1(x)
            x = self.depconv(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.permute(0,2,3,1).contiguous()
            return x
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
    
class Attention(nn.Module):
    def __init__(
        self, 
        dim, num_heads=8, qkv_bias=False, 
        qk_scale=None, attn_drop=0., 
        proj_drop=0., 
        rasa_cfg=None, sr_ratio=1, 
        linear=False, permute=False
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.permute = permute
        self.linear = linear
        self.rasa_cfg = rasa_cfg
        self.use_rasa = rasa_cfg is not None
        self.sr_ratio = sr_ratio
        self.layer_norm = nn.LayerNorm(dim)
        
        
        if not linear:
            if sr_ratio > 1:
                # self.sr = nn.Sequential(
                #     nn.Conv2d(dim, dim // 4, 1, 1, 0),
                #     nn.GELU(),
                #     nn.Conv2d(dim // 4, dim // 4, kernel_size=sr_ratio, stride=sr_ratio, groups=dim // 4),
                #     nn.GELU(),
                #     nn.Conv2d(dim // 4, dim, 1, 1, 0),
                # )
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
                self.norm = nn.LayerNorm(dim)
               
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        
        if self.use_rasa:
            if self.rasa_cfg.atrous_rates is not None:
                self.ds = ds_conv2d(
                    dim, dim, kernel_size=3, stride=1, 
                    dilation=self.rasa_cfg.atrous_rates, groups=dim, bias=qkv_bias, 
                    act_layer=self.rasa_cfg.act_layer, init=self.rasa_cfg.init,
                )
            if self.rasa_cfg.r_num > 1:
                self.silu = nn.SiLU(True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _inner_attention(self, x):
        if self.permute:
            x = x.permute(0,2,3,1).contiguous()
        B, H, W, C = x.shape
        x = self.layer_norm(x)
        q = self.q(x).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
       
        if self.use_rasa:
            if self.rasa_cfg.atrous_rates is not None:
                q = q.permute(0,1,3,2).reshape(B, self.dim, H, W).contiguous()
                q = self.ds(q)
                q = q.reshape(B, self.num_heads, self.dim//self.num_heads, H*W).permute(0,1,3,2).contiguous()
                
        
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0,3,1,2)
                x_ = self.sr(x_).permute(0,2,3,1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                
        else:
              raise NotImplementedError
        
        k, v = kv[0], kv[1]

        try:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
        except Exception as e:
            print()
            print(e)
            print(x.shape)
            print(q.shape, k.transpose(-2, -1).shape)
            exit()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.permute:
            x = x.permute(0,3,1,2).contiguous()
        return x
                
    def forward(self, x):
        if self.use_rasa:
            x_in = x
            x = self._inner_attention(x)
            if self.rasa_cfg.r_num > 1:
                x = self.silu(x)
            for _ in range(self.rasa_cfg.r_num-1):
                x = x + x_in
                x_in = x
                x = self._inner_attention(x)
                x = self.silu(x)
        else:
            x = self._inner_attention(x)
        return x

class Transformer_block(nn.Module):
    def __init__(self, dim,
                 num_heads=1, mlp_ratio=3., attn_drop=0.,
                 drop_path=0., sa_layer='sa', rasa_cfg=None, sr_ratio=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 qkv_bias=False, qk_scale=None, with_depconv=False, chunks=1):
        super().__init__()
        self.sa_layer = sa_layer
        self.num_chunks = chunks
        if sa_layer == 'csa':
            self.num_chunks = 1
            self.attn = CSA(
                dim, dim, num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop)
        elif sa_layer in ['rasa', 'sa']:
            self.num_chunks = 1
            self.attn = Attention(
                dim, num_heads=num_heads, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, rasa_cfg=rasa_cfg, sr_ratio=sr_ratio)
        else:
            raise NotImplementedError
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            with_depconv=with_depconv)

    def forward(self, x):
    
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
            
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, out_c=768, k=7, s=4, idx=0):
        super().__init__()
        patch_size = to_2tuple(k)
        self.patch_size = patch_size
       
        # self.proj = downsample_block(in_c, out_c, k=k, s=s, p=k // 2, as_final_block=True)
        # self.proj = nn.Sequential(
        #     ConvLayer(in_c, out_c, kernel_size=k, stride=s, padding=k//2, as_final_block=True),
        # )
        # self.proj = Res_Block(in_c, out_c, k=k, s=s, p=k // 2, as_final_block=True)
        self.proj = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=k//2),
                                  nn.GELU(),
                                #   nn.Conv2d(
                                #             out_c, out_c, 3, stride=1, padding=1, dilation=1, 
                                #             groups=out_c, bias=True,
                                #     )
        )
        
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)#.permute(0, 2, 3, 1)
        #x = self.norm(x).permute(0, 3, 1, 2)
        return x

class lite_vision_transformer(nn.Module):
    
    def __init__(self, layers, in_chans=3, num_classes=1000, patch_size=4,
                 embed_dims=None, num_heads=None,
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=None,
                 mlp_ratios=None, mlp_depconv=None, sr_ratios=[1,1,1,1], 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True,
                 chunks=[1, 1, 1, 4, 8, 8], **kwargs):

        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_depconv = mlp_depconv
        self.sr_ratios = sr_ratios
        self.layers = layers
        self.num_classes = num_classes
        self.sa_layers = sa_layers
        self.rasa_cfg = rasa_cfg
        self.with_cls_head = with_cls_head # set false for downstream tasks
        network = []
        for stage_idx in range(len(layers)):
        
            _patch_embed = OverlapPatchEmbed(    
                    in_c = in_chans if stage_idx == 0 else self.embed_dims[stage_idx-1],
                    out_c=self.embed_dims[0] if stage_idx == 0 else self.embed_dims[stage_idx],
                    k=7 if stage_idx == 0 else 3, s = 4 if stage_idx == 0 else 2,
                    idx=stage_idx
            )
            
            _blocks = []
            for block_idx in range(layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(layers[:stage_idx])) / (sum(layers) - 1)
                # _blocks.append(Transformer_block(
                #     embed_dims[stage_idx], 
                #     num_heads=num_heads[stage_idx], 
                #     mlp_ratio=mlp_ratios[stage_idx],
                #     sa_layer=sa_layers[stage_idx],
                #     rasa_cfg=self.rasa_cfg[stage_idx],
                #     sr_ratio=sr_ratios[stage_idx],
                #     qkv_bias=qkv_bias, qk_scale=qk_scale, 
                #     attn_drop=attn_drop_rate, drop_path=block_dpr,
                #     with_depconv=mlp_depconv[stage_idx],
                #     chunks=chunks[stage_idx])
                # )
                _blocks.append(Block(dim=embed_dims[stage_idx], num_heads=num_heads[stage_idx], mlp_ratio=1, sr_ratio=sr_ratios[stage_idx]))
                
  
            _blocks = nn.Sequential(*_blocks)
            
            # _conv_head = nn.Sequential(ResidualBlock(embed_dims[stage_idx], embed_dims[stage_idx], mid_c=min(embed_dims[stage_idx] // 4, 64), reduction=2), LayerNormConv())
            
            network.append(nn.ModuleList([
                _patch_embed, 
                _blocks,
# 
                # ConvLayer(embed_dims[stage_idx], embed_dims[stage_idx], kernel_size=1, stride=1, padding=0)
                Res_Chain(embed_dims[stage_idx], embed_dims[stage_idx], num_layers=4)
                
            ]))
            
        
        # backbone
        self.backbone = nn.ModuleList(network)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
    
        outs = []
        for idx, stage in enumerate(self.backbone):
            x = stage[0](x)#.permute(0, 2, 3, 1).contiguous()
            x = stage[1](x)#.permute(0, 3, 1, 2).contiguous()
            x = stage[2](x)
            outs.append(x)
        return outs


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_c=64, skip_size=0, upscale_factor=2):
        super().__init__()
        self.skip_size = skip_size
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upscale_factor, stride=upscale_factor),
            nn.GELU()
        )
        self.conv = ShortResBlock(out_channels, out_channels, mid_c=min(mid_c, in_channels), k=3, s=1, p=1, as_final_block=True)
        self.norm = nn.LayerNorm(out_channels)
        # self.conv = ResidualBlock(in_channels, out_channels, mid_c=min(mid_c, in_channels),  k=3, s=1, p=1)
    
    def forward(self, x, skip=None):
        
        x = self.upsample(x)
        
        if self.skip_size > 0:
            assert skip is not None, "Skip connection is required"
            assert skip.size(1) == self.skip_size, "Skip size mismatch, expected {} but got {}".format(self.skip_size, skip.size(1))
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x).permute(0, 2, 3, 1).contiguous()
        x = self.norm(x).permute(0, 3, 1, 2).contiguous()
        return x

class lite_vision_transformer_decoder(nn.Module):
    def __init__(self, layers, in_chans=3, num_classes=1000, patch_size=4,
                 embed_dims=None, num_heads=None, with_skip=True, skip_sizes=None,
                 sa_layers=['csa', 'rasa', 'rasa', 'rasa'], rasa_cfg=None,
                 mlp_ratios=None, mlp_depconv=None, sr_ratios=[1, 1, 1, 1],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, with_cls_head=True,
                 chunks=[1, 1, 1, 4, 8, 8],
                  **kwargs):
            
        super().__init__()
        self.embed_dims = embed_dims
        self.num_layers = len(layers)
        
        idx = len(sa_layers) - self.num_layers
        
        self.num_heads = num_heads[idx:]
        self.mlp_depconv = mlp_depconv[idx:]
        self.sr_ratios = sr_ratios[idx:]
        self.layers = layers
        self.sa_layers = sa_layers[idx:]
        self.rasa_cfg = rasa_cfg
        self.with_skip = with_skip
        self.mlp_ratios = mlp_ratios[idx:]

        dec_channels = [in_chans] + self.embed_dims + [8]
        self.skip_sizes = skip_sizes
        self._upsample_blocks = nn.ModuleList()
        
        for stage_idx in range(len(self.layers) + 1):
            in_c = dec_channels[stage_idx]
            if self.with_skip:
                in_c += self.skip_sizes[stage_idx]
    
            _upsample_block = DecoderBlock(
                in_channels=in_c,
                out_channels=dec_channels[stage_idx + 1],
                skip_size=self.skip_sizes[stage_idx] if self.with_skip else 0,
            )
               
            self._upsample_blocks.append(_upsample_block)

        self._trans_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for stage_idx in range(len(layers)):

            _blocks = nn.ModuleList()
            for block_idx in range(self.layers[stage_idx]):
                block_dpr = drop_path_rate * (block_idx + sum(self.layers[:stage_idx])) / (sum(self.layers) - 1)
                
                _blocks.append(Transformer_block(
                    self.embed_dims[stage_idx],
                    num_heads=self.num_heads[stage_idx],
                    mlp_ratio=self.mlp_ratios[stage_idx],
                    sa_layer=self.sa_layers[stage_idx],
                    rasa_cfg=self.rasa_cfg[stage_idx],
                    sr_ratio= self.sr_ratios[stage_idx],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop_rate, drop_path=block_dpr,
                    with_depconv=self.mlp_depconv[stage_idx],
                    chunks=chunks[stage_idx]))

            _blocks = nn.Sequential(*_blocks)
            self._trans_blocks.append(_blocks)
            self.norms.append(nn.LayerNorm(self.embed_dims[stage_idx]))
            
        self.depth_activation = Depth_Activation(dec_channels[-1] , 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, skip_features=None):
        
        if self.with_skip:
            assert skip_features is not None, "Skip features are required"

        for idx in range(len(self._upsample_blocks)):
            x = self._upsample_blocks[idx](x, skip_features[-(idx + 2)] if self.with_skip else None)
            
            if idx < len(self._upsample_blocks) - 1:
                x = self._trans_blocks[idx](x.permute(0,2,3,1).contiguous())
                x = self.norms[idx](x).permute(0,3,1,2).contiguous()

        final_depth = self.depth_activation(x)
                
        if args.summary:
            return final_depth 
        
        
        return {
                "depth": {"final_depth": final_depth, "quant_layer": x}, 
        }


       
       
       
@register_model 
class lvt(lite_vision_transformer):
    def __init__(self, 
                in_chans=3,
                rasa_cfg=[args.rasa_cfg, args.rasa_cfg, args.rasa_cfg, args.rasa_cfg, args.rasa_cfg, None], #args.rasa_cfg,
                # rasa_cfg=[None, None, None, None, None, None], #args.rasa_cfg,
                with_cls_head=False, 
                layers=[2, 2, 2, 3, 4, 1],
                embed_dims=[64, 128, 192, 288, 768, 1024],
                # num_heads=[2, 4, 8, 8, 8, 2],
                num_heads=[2, 4, 4, 4, 4, 2],
                mlp_ratios=[4, 4, 4, 4, 2],
                # mlp_ratios=[4, 4, 4, 4, 4, 2],
                mlp_depconv=[True, True, True, True, True, False],
                # sr_ratios=[32, 32, 8, 4, 1, 1],
                sr_ratios=[16, 8, 4, 2, 1, 1],
                # sa_layers=['csa', 'rasa', 'rasa', 'rasa', 'rasa', 'hsa'],
                sa_layers=['csa', 'rasa', 'rasa', 'rasa', 'rasa', 'hsa'],
                chunks = [1, 1, 1, 1, 8, 8],
                **kwargs):
        
        super().__init__(
            in_chans=in_chans,
            layers=layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            mlp_depconv=mlp_depconv,
            sr_ratios=sr_ratios,
            sa_layers=sa_layers,
            rasa_cfg=rasa_cfg,
            with_cls_head=with_cls_head,
            chunks=chunks,
            **kwargs
        ) 

@register_model
class lvt_decoder(lite_vision_transformer_decoder):
    def __init__(self, 
                 rasa_cfg=[None, None, None, None], 
                 embed_dims=[96, 64, 32, 16], 
                 layers=[2, 2, 1, 1, 1, 1],
                 num_heads=[4, 4, 4, 4, 2, 2],
                 mlp_ratios=[2, 2, 2, 2, 2, 2],
                mlp_depconv=[True, True, True, False],
                sr_ratios=[2, 4, 8, 16],
                sa_layers=['hsa', 'hsa', 'rasa', 'rasa', 'rasa', 'rasa'],
                skip_sizes=[480, 240, 192, 128, 64, 3, 3],
                chunks=[1, 1, 1, 1],
                 with_skip=True, 
                 **kwargs):
        
        super().__init__(
            layers=layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            mlp_depconv=mlp_depconv,
            sr_ratios=sr_ratios,
            sa_layers=sa_layers,
            rasa_cfg=rasa_cfg,
            with_skip=with_skip,
            skip_sizes=skip_sizes,
            chunks=chunks,
            **kwargs
        )


# class CrossModalChain(nn.Module):
    
#     def __init__(self, d_model=16, num_heads=4, num_layers=2):
        
#         super().__init__()
#         self.layers = nn.ModuleList([CrossModalAttention(d_model, num_heads) for _ in range(num_layers)])
        
#     def forward(self, x, y):
#         for layer in self.layers:
#             x = layer(x, y)
#         return x
    
class CrossModalAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, sr_ratio=1):
        super().__init__()
        
        # Multi-head attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        
        # Layer normalization
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        # Feedforward layer for post-attention
        self.fc_out = nn.Linear(d_model, d_model)
        
        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio, groups=d_model)
            # self.sr = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        

    def forward(self, x):
        # Save the input for the skip connection
        y = x
        bx, cx, hx, wx = x.shape
        # Apply multi-head attention (expects input shape: (N, B, C))
        x = x.permute(2, 3, 0, 1).reshape(hx*wx, bx, cx)  # Reshape to (N, B, C)
        
        if self.sr_ratio > 1:
            y = self.sr(y)
        
        by, cy, hy, wy = y.shape
        y = y.permute(2, 3, 0, 1).reshape(hy*wy, by, cy)  # Reshape to (M, B, C)
        
        residual = x
        x = self.norm_1(x)
        y = self.norm_1(y)
        attn_output, _ = self.multihead_attn(x, y, y)  # (N, B, C)
        # Skip connection and layer normalization

        x = (attn_output + residual)  # Reshape back to (B, C, N)
        
        # Apply output layer and another normalization
        output = self.fc_out(x)
        output = self.norm_2(output + x)  # Add skip connection again
        
        output = output.reshape(hx, wx, bx, cx).permute(2, 3, 0, 1)
        
        return output
    
class StackedCrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        
        # Stack multiple CrossModalAttention layers
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x, y)
        return x
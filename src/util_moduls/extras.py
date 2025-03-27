
class Decoder(nn.Module):
    """
    A simple upsampling layer, further processed with a convolutional block of choice.
    """
    def __init__(self, in_c, out_c, skip_size=None, 
                 upscale_factor=2, reduced=False, **kwargs):
        super().__init__()
        self.out_channels = out_c
        self.incoming_skip = skip_size is not None
                    
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=upscale_factor, stride=upscale_factor),
            nn.GroupNorm(how_many_groups(out_c), out_c),
            nn.GELU()

             
        )
        in_c = out_c + (skip_size if self.incoming_skip else 0)
            
        if reduced:
            self.conv = nn.Sequential(
            ConvLayer(in_c, out_c, 1, 1, 0),
        )
        else:
            self.conv = nn.Sequential(
                Res_Chain(in_c, out_c, num_layers=3)
            )
       
        self.apply(weights_init_kaiming)
        
    def forward(self, x, skip=None):
        
        x = self.upsample(x)
        if self.incoming_skip:
            x = torch.cat((x, skip), dim=1)
        out = self.conv(x)
        return out

class AE_Decoder(nn.Module):
    
    def __init__(self,
            num_layers=5,
            latent_dim=16,
            embed_dims=[8, 32, 64, 128, 192],     
            num_heads = [None, 1, 2, 4, 4, 4],
            mlp_ratios = [2, 2, 2, 2, 2, 2],
            mlp_depconv=[False, True, True, True, True, True],
            sr_ratios=[None, 8, 4, 2, 2, 1], 
            skip_sizes = None,
            with_skip=True,
            in_chans=None,
            layer_types=["rasa", "rasa", "rasa", "csa"],
            **kwargs
        ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims[:self.num_layers]
        self.skip_sizes = skip_sizes if with_skip else [None] * self.num_layers
        self.mid_ch = [8, 32, 64, 64, 64][:self.num_layers]
        
        self.sr_ratios = sr_ratios[:self.num_layers]
        self.num_heads = num_heads[:self.num_layers]
        self.mlp_ratios = mlp_ratios[:self.num_layers]
        self.mlp_depconv = mlp_depconv[:self.num_layers]     

        self.with_skip = with_skip
        self.latent_dim = latent_dim
        
        self.depth_upsample = nn.ModuleList()
        self.trans_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for idx in range(self.num_layers):
            in_c = in_chans if idx == (self.num_layers - 1) else self.embed_dims[idx + 1]
            out_c = self.embed_dims[idx]
            if idx > 0:
                upscale_fcator = 2
                self.trans_blocks.append(
                    Transformer_block(
                        self.embed_dims[idx],
                        num_heads=self.num_heads[idx],
                        mlp_ratio=1,
                        sr_ratio= self.sr_ratios[idx],
                        with_depconv=self.mlp_depconv[idx],
                        sa_layer="rasa",
                        rasa_cfg=args.rasa_cfg
                    ) if idx > 1 else nn.Identity()
                )
               
                self.depth_upsample.append(Decoder(in_c=in_c, out_c=out_c, skip_size=self.skip_sizes[idx] if self.with_skip else None, upscale_factor=upscale_fcator))
                # 
            else:
                upscale_fcator = 4
                self.trans_blocks.append(
                    nn.Identity()
                )
                self.depth_upsample.append(Decoder(in_c=in_c, out_c=out_c, skip_size=self.skip_sizes[idx] if self.with_skip else None, upscale_factor=upscale_fcator, reduced=True))
        self.depth_activation = Depth_Activation(self.embed_dims[0], 1)
   
    def forward(self, latent, skip_features=[None] * 6, only_latent=False, depth_detach=False):
        outs = []
        outs.append(latent)
        
        if only_latent:
            return {"outs": outs, "depth": {}}
        
        skip_features = skip_features[::-1][1:]
        for idx in range(1, self.num_layers + 1):
            skipi = skip_features[idx - 1] if skip_features[idx - 1] is not None else None
            latent = self.depth_upsample[-idx](latent, skipi)
            latent = self.trans_blocks[-idx](latent)
            if idx < self.num_layers:
                outs.append(latent)

        if depth_detach:
            latent = latent.detach()
        
        final_depth, logits = self.depth_activation(latent)
           
        if args.summary:
            return final_depth 
        
        return {
                "depth": {"final_depth": final_depth, "logits": logits}, 
                "outs": outs

        }
        
    
    
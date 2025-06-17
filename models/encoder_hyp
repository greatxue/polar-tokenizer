import torch
import torch.nn as nn
import geoopt
from models.hyp_utils import lorentz, lorentz_swish, LorentzConv2d, LorentzResBlock
# Assuming lorentz, lorentz_swish, LorentzConv2d, LorentzResBlock are defined above or imported
# from .hyperbolic_utils import lorentz, lorentz_swish, LorentzConv2d, LorentzResBlock # Example

# (Paste or import common hyperbolic modules here if not in a separate utils file)
# For brevity, assuming they are defined in the same scope for this snippet.
# Initialize Lorentz manifold (typically done once)

class EncoderLorentz(nn.Module):
    def __init__(self, in_channels_euc, h_dim_tan, n_res_layers):
        super().__init__()
        self.h_dim_tan = h_dim_tan
        # Initial Euclidean convolution to get to h_dim_tan for the first expmap
        self.conv_in_euc_to_tan = nn.Conv2d(in_channels_euc, h_dim_tan, kernel_size=3, stride=1, padding=1)
        
        ch_mult = (1, 2, 2, 4)  # Channel multipliers for tangent space dimensions
        num_resolutions = len(ch_mult)
        
        self.down_blocks = nn.ModuleList()
        curr_tan_channels = h_dim_tan
        
        for i_level in range(num_resolutions):
            level_modules = nn.ModuleList()
            out_tan_channels_level = h_dim_tan * ch_mult[i_level]
            for _ in range(n_res_layers):
                level_modules.append(LorentzResBlock(curr_tan_channels, out_tan_channels_level))
                curr_tan_channels = out_tan_channels_level
            
            if i_level != num_resolutions - 1:
                # Downsampling: LorentzConv2d with stride
                level_modules.append(
                    LorentzConv2d(curr_tan_channels, curr_tan_channels, kernel_size=4, stride=2, padding=1)
                )
            self.down_blocks.append(level_modules)
            
        self.mid_block1 = LorentzResBlock(curr_tan_channels, curr_tan_channels)
        self.mid_block2 = LorentzResBlock(curr_tan_channels, curr_tan_channels)
        self.final_tan_channels = curr_tan_channels # To be used by VQVAE
            
    def forward(self, x_euc):
        # x_euc: (B, C_euc, H, W)
        x_tan = self.conv_in_euc_to_tan(x_euc) # (B, h_dim_tan, H, W)
        x_hyp = lorentz.expmap0(x_tan)        # (B, h_dim_tan+1, H, W)
        
        for level_modules in self.down_blocks:
            for layer in level_modules:
                x_hyp = layer(x_hyp)
        
        x_hyp = self.mid_block1(x_hyp)
        x_hyp = self.mid_block2(x_hyp)
        
        return x_hyp

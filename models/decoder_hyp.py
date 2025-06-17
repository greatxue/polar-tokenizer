import torch
import torch.nn as nn
import geoopt
from models.hyp_utils import lorentz, lorentz_swish, LorentzConv2d, LorentzResBlock
# Assuming lorentz, lorentz_swish, LorentzConv2d, LorentzResBlock are defined or imported
# from .hyperbolic_utils import lorentz, lorentz_swish, LorentzConv2d, LorentzResBlock # Example

# (Paste or import common hyperbolic modules here if not in a separate utils file)
# For brevity, assuming they are defined in the same scope for this snippet.


class LorentzUpsampler(nn.Module):
    """Upsampling layer for features on the Lorentz manifold using PixelShuffle on tangent space."""
    def __init__(self, in_tan_channels, scale_factor=2):
        super().__init__()
        # Target tangent channels after upsampling (typically halving for scale_factor=2)
        self.out_tan_channels = in_tan_channels // 2
        # Conv output channels for PixelShuffle must be out_tan_channels * scale_factor^2
        conv_out_intermediate_tan = self.out_tan_channels * (scale_factor**2)
        
        self.conv_euc = nn.Conv2d(in_tan_channels, conv_out_intermediate_tan, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x_hyp):
        if not lorentz.check_point_on_manifold(x_hyp, atol=1e-3, rtol=1e-3).all():
            x_hyp = lorentz.projx(x_hyp)
        x_tan = lorentz.logmap0(x_hyp)
        x_tan_conv = self.conv_euc(x_tan)
        x_tan_shuffled = self.pixel_shuffle(x_tan_conv)
        return lorentz.expmap0(x_tan_shuffled)

class DecoderLorentz(nn.Module):
    def __init__(self, in_quantizer_tan_channels, h_dim_tan, n_res_layers, out_channels_euc=3):
        super().__init__()
        
        # Channel multipliers from original Encoder, used in reverse for Decoder
        ch_mult_encoder_equivalent = (1, 2, 2, 4) 
        num_resolutions = len(ch_mult_encoder_equivalent)

        # Initial conv: input from quantizer (embedding_dim_tan) to highest channel dim of decoder
        initial_decoder_tan_channels = h_dim_tan * ch_mult_encoder_equivalent[-1] # e.g. h_dim_tan * 4
        self.conv_in = LorentzConv2d(in_quantizer_tan_channels, initial_decoder_tan_channels, kernel_size=3, padding=1)
        
        curr_tan_channels = initial_decoder_tan_channels
        
        self.mid_block1 = LorentzResBlock(curr_tan_channels, curr_tan_channels)
        self.mid_block2 = LorentzResBlock(curr_tan_channels, curr_tan_channels)
        
        self.up_blocks = nn.ModuleList()
        # Loop from high-channel-dim stages to low-channel-dim stages
        for i_level_idx_encoder in reversed(range(num_resolutions)): # Iterates 3, 2, 1, 0
            level_modules = nn.ModuleList()
            # Target tangent channels for ResBlocks at this upsampling stage
            out_tan_channels_for_resblocks = h_dim_tan * ch_mult_encoder_equivalent[i_level_idx_encoder]
            
            for _ in range(n_res_layers + 1): # As per original Euclidean decoder
                level_modules.append(LorentzResBlock(curr_tan_channels, out_tan_channels_for_resblocks))
                curr_tan_channels = out_tan_channels_for_resblocks
            
            if i_level_idx_encoder != 0: # Add upsampler if not the final (highest spatial res) stage
                level_modules.append(LorentzUpsampler(curr_tan_channels, scale_factor=2))
                # LorentzUpsampler with scale_factor=2 is designed to halve tangent channels
                curr_tan_channels = curr_tan_channels // 2 
            
            self.up_blocks.append(level_modules)
            
        # Final output to Euclidean space
        num_groups_final = min(32, curr_tan_channels // 4 if curr_tan_channels >=4 else 1)
        if curr_tan_channels < num_groups_final : num_groups_final =1
        self.norm_out_tan = nn.GroupNorm(num_groups_final, curr_tan_channels, eps=1e-6)
        self.conv_out_tan_to_euc = nn.Conv2d(curr_tan_channels, out_channels_euc, kernel_size=3, padding=1)
            
    def forward(self, x_hyp_quantized):
        # x_hyp_quantized: (B, in_quantizer_tan_channels+1, H_q, W_q) - Output from VQ, on manifold
        x_hyp = self.conv_in(x_hyp_quantized)
        
        x_hyp = self.mid_block1(x_hyp)
        x_hyp = self.mid_block2(x_hyp)
        
        for level_modules in self.up_blocks:
            for layer in level_modules:
                x_hyp = layer(x_hyp)
                
        # Final projection to Euclidean image
        if not lorentz.check_point_on_manifold(x_hyp, atol=1e-3, rtol=1e-3).all():
            x_hyp = lorentz.projx(x_hyp)
        x_tan_final = lorentz.logmap0(x_hyp)
        x_tan_norm_final = self.norm_out_tan(x_tan_final)
        # Apply final Swish in tangent space before Euclidean conv
        x_tan_swish_final = x_tan_norm_final * torch.sigmoid(x_tan_norm_final)
        x_euc_out = self.conv_out_tan_to_euc(x_tan_swish_final)
        
        return x_euc_out

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def swish(x):
    # swish activation function
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, use_conv_shortcut=False):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        
        self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)
        
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        
        if in_filters != out_filters:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), bias=False)
                
    def forward(self, x):
        residual = x
        
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.shortcut(residual)
            else:
                residual = self.shortcut(residual)
                
        return x + residual

def depth_to_space(x, block_size):
    """Depth-to-Space实现"""
    if x.dim() < 3:
        raise ValueError("Input tensor must have at least 3 dimensions")
        
    c, h, w = x.shape[-3:]
    
    s = block_size**2
    if c % s != 0:
        raise ValueError(f"Channel dimension ({c}) must be divisible by block_size^2 ({s})")
        
    outer_dims = x.shape[:-3]
    
    # 从通道维度拆分出两个额外维度
    x = x.view(-1, block_size, block_size, c // s, h, w)
    
    # 将两个新维度放置在H和W位置
    x = x.permute(0, 3, 4, 1, 5, 2)
    
    # 将新维度与H和W合并
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)
    
    return x

class Upsampler(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        
        self.conv = nn.Conv2d(dim, dim_out * 4, kernel_size=3, padding=1)
        self.block_size = 2
        
    def forward(self, x):
        x = self.conv(x)
        x = depth_to_space(x, self.block_size)
        return x

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.
    
    This improved version uses GroupNorm, Swish, and depth-to-space upsampling.
    
    Inputs:
    - in_dim : the input dimension (channels)
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        
        # 配置上采样路径
        ch_mult = (1, 2, 2, 4)  # 通道数乘数（与编码器相反方向）
        num_resolutions = len(ch_mult)
        
        # 初始化通道数
        self.conv_in = nn.Conv2d(in_dim, h_dim * ch_mult[-1], kernel_size=3, padding=1)
        
        # 中间块
        curr_res = h_dim * ch_mult[-1]
        self.mid_block1 = ResBlock(curr_res, curr_res)
        self.mid_block2 = ResBlock(curr_res, curr_res)
        
        # 上采样块
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            block = nn.ModuleList()
            out_ch = h_dim * ch_mult[i_level]
            
            # 每个分辨率上的ResBlock数量
            for i_block in range(n_res_layers+1):
                block.append(ResBlock(curr_res, out_ch))
                curr_res = out_ch
                
            # 上采样（除最低分辨率外）
            if i_level != 0:
                block.append(Upsampler(curr_res, curr_res//2))
                curr_res = curr_res // 2
                
            self.up_blocks.append(block)
            
        # 输出卷积
        self.norm_out = nn.GroupNorm(32, curr_res)
        self.conv_out = nn.Conv2d(curr_res, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 初始特征转换
        x = self.conv_in(x)
        
        # 中间处理
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        
        # 上采样路径
        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
                
        # 最终输出
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        
        return x
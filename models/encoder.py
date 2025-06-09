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

class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.
    
    This improved version uses GroupNorm, Swish and deeper residual blocks.
    
    Inputs:
    - in_dim : the input dimension (channels)
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """
    
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        
        # 配置下采样路径
        ch_mult = (1, 2, 2, 4)  # 通道数乘数
        num_resolutions = len(ch_mult)
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1)
        
        # 下采样块
        self.down_blocks = nn.ModuleList()
        curr_res = h_dim
        
        # 每个分辨率级别
        for i_level in range(num_resolutions):
            block = nn.ModuleList()
            out_channels = h_dim * ch_mult[i_level]
            
            # 每个分辨率上的ResBlock数量
            for i_block in range(n_res_layers):
                block.append(ResBlock(curr_res, out_channels))
                curr_res = out_channels
                
            # 下采样 (除最后一层外)
            if i_level != num_resolutions - 1:
                block.append(
                    nn.Conv2d(curr_res, curr_res, kernel_size=4, stride=2, padding=1)
                )
            
            self.down_blocks.append(block)
            
        # 中间块 (额外的ResBlock处理)
        self.mid_block1 = ResBlock(curr_res, curr_res)
        self.mid_block2 = ResBlock(curr_res, curr_res)
            
    def forward(self, x):
        # 初始特征提取
        x = self.conv_in(x)
        
        # 下采样路径
        for i_level, block in enumerate(self.down_blocks):
            for layer in block:
                x = layer(x)
        
        # 中间处理
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        
        return x
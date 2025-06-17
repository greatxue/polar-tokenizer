# Common imports for both encoder_lorentz.py and decoder_lorentz.py
import torch
import torch.nn as nn
import geoopt

# Initialize Lorentz manifold (typically done once)
# Curvature 'k' is negative for hyperbolic spaces.
# For VQ-VAE, k=-1.0 is a common fixed choice.
lorentz = geoopt.Lorentz(k=-1.0) # Or k=geoopt.ManifoldParameter(torch.tensor(-1.0)) for learnable k

def lorentz_swish(x_hyp):
    """Swish activation for features on the Lorentz manifold."""
    if not lorentz.check_point_on_manifold(x_hyp, atol=1e-3, rtol=1e-3).all(): # check with tolerance
        # Attempt to project if slightly off, common after arithmetic ops
        x_hyp = lorentz.projx(x_hyp)
    x_tan = lorentz.logmap0(x_hyp)
    x_tan_swish = x_tan * torch.sigmoid(x_tan)
    return lorentz.expmap0(x_tan_swish)

class LorentzConv2d(nn.Module):
    """Convolutional layer for features on the Lorentz manifold."""
    def __init__(self, in_channels_tan, out_channels_tan, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # Convolutions operate on the tangent space representation (d dimensions)
        self.conv_euc = nn.Conv2d(in_channels_tan, out_channels_tan, kernel_size, stride, padding, bias=bias)

    def forward(self, x_hyp):
        # x_hyp: (B, D_in_tan+1, H, W)
        if not lorentz.check_point_on_manifold(x_hyp, atol=1e-3, rtol=1e-3).all():
            x_hyp = lorentz.projx(x_hyp)
        x_tan = lorentz.logmap0(x_hyp)    # (B, D_in_tan, H, W)
        out_tan = self.conv_euc(x_tan)    # (B, D_out_tan, H', W')
        return lorentz.expmap0(out_tan)   # (B, D_out_tan+1, H', W')

# filepath: /ext/work/polar/models/hyp_utils.py
class LorentzResBlock(nn.Module):
    """Residual Block for features on the Lorentz manifold."""
    def __init__(self, channels_tan, out_channels_tan=None):
        super().__init__()
        if out_channels_tan is None:
            out_channels_tan = channels_tan
            
        # 确保num_groups能被通道数整除
        def get_safe_num_groups(channels):
            if channels == 0: return 1
            for grp in [32, 16, 8, 4, 2]:
                if channels % grp == 0:
                    return grp
            return 1
            
        num_groups_in = get_safe_num_groups(channels_tan)
        num_groups_out = get_safe_num_groups(out_channels_tan)

        self.norm1 = nn.GroupNorm(num_groups_in, channels_tan, eps=1e-6)
        self.conv1 = LorentzConv2d(channels_tan, out_channels_tan, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels_tan, eps=1e-6)
        self.conv2 = LorentzConv2d(out_channels_tan, out_channels_tan, kernel_size=3, padding=1, bias=False)
        
        # 处理通道数变化的shortcut连接
        self.shortcut = None
        if channels_tan != out_channels_tan:
            self.shortcut = LorentzConv2d(channels_tan, out_channels_tan, kernel_size=1, bias=False)

    def forward(self, x_hyp):
        residual_hyp = x_hyp
        if self.shortcut is not None:
            residual_hyp = self.shortcut(residual_hyp)

        if not lorentz.check_point_on_manifold(x_hyp, atol=1e-3, rtol=1e-3).all(): x_hyp = lorentz.projx(x_hyp)
        x_tan = lorentz.logmap0(x_hyp)
        x_tan_norm = self.norm1(x_tan)
        x_hyp_norm = lorentz.expmap0(x_tan_norm)
        x_hyp_swish = lorentz_swish(x_hyp_norm)
        x_hyp_conv1 = self.conv1(x_hyp_swish)

        if not lorentz.check_point_on_manifold(x_hyp_conv1, atol=1e-3, rtl=1e-3).all(): x_hyp_conv1 = lorentz.projx(x_hyp_conv1)
        x_tan_conv1 = lorentz.logmap0(x_hyp_conv1)
        x_tan_norm2 = self.norm2(x_tan_conv1)
        x_hyp_norm2 = lorentz.expmap0(x_tan_norm2)
        x_hyp_swish2 = lorentz_swish(x_hyp_norm2)
        x_hyp_conv2 = self.conv2(x_hyp_swish2)
        
        if not lorentz.check_point_on_manifold(x_hyp_conv2, atol=1e-3, rtl=1e-3).all(): x_hyp_conv2 = lorentz.projx(x_hyp_conv2)
        if not lorentz.check_point_on_manifold(residual_hyp, atol=1e-3, rtl=1e-3).all(): residual_hyp = lorentz.projx(residual_hyp)
        out_hyp = lorentz.mobius_add(x_hyp_conv2, residual_hyp, k=lorentz.k)
        return lorentz.projx(out_hyp)

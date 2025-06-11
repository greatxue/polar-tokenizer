import torch
import torch.nn as nn
import torch.nn.functional as F
def hyperboloid_to_poincare(u):
    """将洛伦兹模型中的点转换到Poincaré球模型"""
    return u[..., 1:] / (u[..., 0:1] + 1.0)
def poincare_to_hyperboloid(x):
    x_norm = torch.sum(x * x, dim=-1, keepdim=True)
    
    # 只对超出安全阈值的点进行裁剪
    too_close_to_boundary = x_norm > 0.999  # 提高阈值
    
    # 选择性裁剪
    x_safe = torch.where(
        too_close_to_boundary,
        x * 0.999 / torch.sqrt(x_norm.clamp(min=1e-8)),
        x
    )
    
    # 其余转换代码不变
    sq_norm = torch.sum(x_safe * x_safe, dim=-1, keepdim=True)
    denom = 1.0 - sq_norm
    denom = torch.clamp(denom, min=1e-6)
    
    u0 = (1.0 + sq_norm) / denom
    u_spatial = 2.0 * x_safe / denom
    return torch.cat([u0, u_spatial], dim=-1)

def exp_map_to_lorentz(v, min_radius=0.1, max_radius=20.0):
    """使用分位数映射获得更均匀的半径分布"""
    # 计算范数
    # 计算范数
    norm = torch.norm(v, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-8)
    
    # 使用固定参考范围
    norm_ref_min = 2   # 根据您数据的最小范数
    norm_ref_max = 200  # 根据您数据的最大范数
    
    # 计算归一化范数
    norm_normalized = (norm - norm_ref_min) / (norm_ref_max - norm_ref_min)
    norm_normalized = torch.clamp(norm_normalized, min=0.0, max=1.0)
    
    # 映射到目标半径范围
    target_radius = min_radius + (max_radius - min_radius) * norm_normalized
    
    # 计算缩放因子
    scale_factor = target_radius / norm
    v_scaled = v * scale_factor
    
    # 标准双曲函数计算
    norm_scaled = torch.norm(v_scaled, dim=-1, keepdim=True)
    cosh_norm = torch.cosh(norm_scaled)
    sinh_norm = torch.sinh(norm_scaled)
    u0 = cosh_norm
    u_spatial = sinh_norm * (v_scaled / norm_scaled.clamp(min=1e-8))
    
    return torch.cat([u0, u_spatial], dim=-1)
def lorentz_inner(u, v):
    # u,v: (..., e_dim+1)
    return -u[...,0]*v[...,0] + torch.sum(u[...,1:]*v[...,1:], dim=-1)

def lorentz_distance(u, v):
    # on hyperboloid: d(u,v) = arccosh( -⟨u,v⟩ )
    prod = -lorentz_inner(u, v)
    prod = torch.clamp(prod, min=1.0 + 1e-5)  # 防止数值不稳定
    return torch.acosh(prod)

def from_polar(r, w):
    """
    r: (...), hyperbolic radius
    w: (..., e_dim), unit vector in tangent/Euc space
    returns x in Poincaré ball: x = tanh(r/2) * w
    """
    return torch.tanh(r.unsqueeze(-1) / 2.0) * w

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, radial_bins=16, max_radius=18.0):
        super().__init__()
        assert n_e % radial_bins == 0, "n_e 必须被 radial_bins 整除"
        self.n_e          = n_e
        self.e_dim        = e_dim           # Poincaré 球的空间维度
        self.beta         = beta
        self.radial_bins  = radial_bins
        self.angular_bins = n_e // radial_bins
        self.max_radius   = max_radius

        # 径向中心（超曲半径 r）
        log_r = torch.linspace(0., torch.log(torch.tensor(max_radius)), radial_bins)
        self.r_centres = nn.Parameter(torch.exp(log_r))  # (radial_bins,)

        # 角度中心：R^e_dim 上的单位向量
        self.angular_codebook = nn.Embedding(self.angular_bins, self.e_dim)
        with torch.no_grad():
            v = torch.randn(self.angular_bins, self.e_dim)
            self.angular_codebook.weight.copy_(F.normalize(v, dim=-1))

    def forward(self, z):
        """
        z: (B, C=e_dim, H, W)
        returns: loss, z_q (B,C,H,W), perplexity, one-hot, indices
        """
        B, C, H, W = z.shape
        assert C == self.e_dim

        # flatten to (N, e_dim)
        z_flat = z.permute(0,2,3,1).reshape(-1, C)  # N = B*H*W

        # --- 1) 投影到 Poincaré 球 ---
        u_hyp = exp_map_to_lorentz(z_flat)
             # (..., e_dim+1)

        # 超曲半径 r = arccosh(u0)
        u0 = u_hyp[..., 0]                                     # (...,)
        r  = torch.acosh(torch.clamp(u0, min=1.0 + 1e-5))      # (...,)

        # 方向单位向量（在 Poinc 球里）
        w = F.normalize(u_hyp[..., 1:], dim=-1)
        
        x_poinc = hyperboloid_to_poincare(u_hyp)                     # (..., e_dim)

        # --- 3) 径向量化 ---
        r_centres = torch.clamp(self.r_centres, min=1e-2, max=self.max_radius)  # (radial_bins,)
        dist_r2   = (r.unsqueeze(-1) - r_centres)**2                             # (..., radial_bins)
        r_idx     = dist_r2.argmin(dim=-1)                                        # (...,)
        r_q       = r_centres[r_idx]                                             # (...,)

        # --- 4) 角度量化 ---
        sim       = torch.matmul(w, self.angular_codebook.weight.t())  # (..., angular_bins)
        w_idx     = sim.argmax(dim=-1)                                # (...,)
        w_q       = self.angular_codebook(w_idx)                      # (..., e_dim)

        # --- 5) 重建 Poincaré 点 & 超曲面用于 loss ---
        x_q_poinc = from_polar(r_q, w_q)                                              # (..., e_dim)
        u_q_hyp   = poincare_to_hyperboloid(x_q_poinc)               # (..., e_dim+1)

        # --- 6) 量化损失 ---
        commit_loss   = torch.mean(lorentz_distance(u_hyp.detach(),   u_q_hyp)) 
        codebook_loss = torch.mean(lorentz_distance(u_hyp, u_q_hyp.detach()))
        loss = commit_loss + self.beta * codebook_loss

        # --- 7) Straight-through estimator 回流 ---
        x_q_poinc = x_poinc + (x_q_poinc - x_poinc).detach()

        # reshape 回 (B, C, H, W)
        z_q = x_q_poinc.view(B, H, W, C).permute(0,3,1,2).contiguous()

        # --- 8) Perplexity & one-hot ---
        N = z_flat.shape[0]
        one_hot = torch.zeros(N, self.n_e, device=z.device)
        ids = r_idx * self.angular_bins + w_idx               # (...,)
        one_hot.scatter_(1, ids.view(-1,1), 1.0)
        e_mean     = torch.mean(one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        used_codes = (one_hot.sum(dim=0) > 0).float().sum()
        codebook_usage = used_codes / self.n_e
        
        with torch.no_grad():
            self.angular_codebook.weight.data.copy_(
                F.normalize(self.angular_codebook.weight.data, dim=-1)
            )        

        return loss, z_q, perplexity, one_hot, ids.view(-1,1), codebook_usage, e_mean




import torch
import torch.nn as nn
import torch.nn.functional as F

def poincare_to_hyperboloid(x):
    """
    x: (..., e_dim), ‖x‖ < 1 (Poincaré ball)
    returns u: (..., e_dim+1) s.t. -u0^2 + sum(u_i^2) = -1
    """
    sq_norm = torch.sum(x * x, dim=-1, keepdim=True)  # (...,1)
    denom   = 1.0 - sq_norm                            # (...,1)
    u0      = (1.0 + sq_norm) / denom                  # (...,1)
    u_spatial = 2.0 * x / denom                        # (..., e_dim)
    return torch.cat([u0, u_spatial], dim=-1)

def lorentz_inner(u, v):
    # u,v: (..., e_dim+1)
    return -u[...,0]*v[...,0] + torch.sum(u[...,1:]*v[...,1:], dim=-1)

def lorentz_distance(u, v):
    # on hyperboloid: d(u,v) = arccosh( -⟨u,v⟩ )
    prod = -lorentz_inner(u, v)
    prod = torch.clamp(prod, min=1.0 + 1e-5)
    return torch.acosh(prod)

def from_polar(r, w):
    """
    r: (...), hyperbolic radius
    w: (..., e_dim), unit vector in tangent/Euc space
    returns x in Poincaré ball: x = tanh(r/2) * w
    """
    return torch.tanh(r.unsqueeze(-1) / 2.0) * w

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, radial_bins=8, max_radius=12.0):
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
        norm_euc = torch.clamp(torch.norm(z_flat, dim=-1, keepdim=True), min=1e-5)
        x_poinc  = (z_flat / norm_euc) * torch.tanh(norm_euc)  # (..., e_dim)

        # --- 2) 球 → 超曲面 用于距离计算 & 半径提取 ---
        u_hyp = poincare_to_hyperboloid(x_poinc)               # (..., e_dim+1)

        # 超曲半径 r = arccosh(u0)
        u0 = u_hyp[..., 0]                                     # (...,)
        r  = torch.acosh(torch.clamp(u0, min=1.0 + 1e-5))      # (...,)

        # 方向单位向量（在 Poinc 球里）
        w = F.normalize(x_poinc, dim=-1)                       # (..., e_dim)

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
        x_q_poinc = from_polar(r_q, w_q)                              # (..., e_dim)
        u_q_hyp   = poincare_to_hyperboloid(x_q_poinc)               # (..., e_dim+1)

        # --- 6) 量化损失 ---
        commit_loss   = torch.mean(lorentz_distance(u_hyp.detach(),   u_q_hyp)**2)
        codebook_loss = torch.mean(lorentz_distance(u_hyp, u_q_hyp.detach())**2)
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

        return loss, z_q, perplexity, one_hot, ids.view(-1,1), codebook_usage

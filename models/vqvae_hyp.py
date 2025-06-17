import torch
import torch.nn as nn
import geoopt
from models.encoder_hyp import EncoderLorentz
from models.decoder_hyp import DecoderLorentz
from models.quantizer import VectorQuantizer, poincare_to_hyperboloid  # 使用双曲量化器

# 全局洛伦兹流形定义
try:
    from models.hyp_utils import lorentz
except ImportError:
    print("警告: 未找到hyp_utils.lorentz，本地定义lorentz")
    lorentz = geoopt.Lorentz(k=-1.0)

class VQVAE_HYP(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False,
                 use_ema=False, ema_decay=0.99, 
                 radial_bins=16, max_radius=20.0):  # 添加双曲量化器参数
        super(VQVAE_HYP, self).__init__()
        
        # 双曲编码器 - h_dim是切空间维度
        self.encoder = EncoderLorentz(in_channels_euc=3, h_dim_tan=h_dim, n_res_layers=n_res_layers)
        
        # 预量化卷积 - 在切空间操作
        self.pre_quantization_conv = nn.Conv2d(
            self.encoder.final_tan_channels, embedding_dim, kernel_size=1, stride=1)
            
        # 双曲量化器
        self.vector_quantization = VectorQuantizer(
            n_e=n_embeddings, e_dim=embedding_dim, beta=beta,
            radial_bins=radial_bins, max_radius=max_radius,
            use_ema=use_ema, ema_decay=ema_decay)
            
        # 双曲解码器
        self.decoder = DecoderLorentz(
            in_quantizer_tan_channels=embedding_dim,
            h_dim_tan=h_dim,
            n_res_layers=n_res_layers,
            out_channels_euc=3
        )
        
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        # 1. 编码器: 欧几里得 → 洛伦兹流形
        z_e_hyp = self.encoder(x)  # 特征在洛伦兹流形上
        
        # 2. 映射到切空间进行预量化卷积
        if not lorentz.check_point_on_manifold(z_e_hyp, atol=1e-3, rtol=1e-3).all():
            z_e_hyp = lorentz.projx(z_e_hyp)
        z_e_tan = lorentz.logmap0(z_e_hyp)  # 切空间表示
        
        # 3. 预量化卷积
        z_e_quant_input = self.pre_quantization_conv(z_e_tan)  # 欧几里得操作
        
        # 4. 量化: 欧几里得 → 洛伦兹 → 量化 → 庞加莱球
        embedding_loss, z_q_poincare, perplexity, _, _, codebook_usage, e_mean = self.vector_quantization(
            z_e_quant_input)
            
        # 5. 转换回洛伦兹流形用于解码
        B, C_emb, H_q, W_q = z_q_poincare.shape
        z_q_poincare_flat = z_q_poincare.permute(0, 2, 3, 1).reshape(-1, C_emb)
        
        # 使用quantizer中的函数将庞加莱球点转回洛伦兹流形
        z_q_lorentz_flat = poincare_to_hyperboloid(z_q_poincare_flat)
        z_q_lorentz = z_q_lorentz_flat.reshape(B, H_q, W_q, C_emb + 1).permute(0, 3, 1, 2).contiguous()
        
        # 6. 解码: 洛伦兹流形 → 欧几里得
        x_hat = self.decoder(z_q_lorentz)  # 输出重建图像
        
        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape (hyp):', z_e_hyp.shape)
            print('quantizer input shape (tan):', z_e_quant_input.shape)
            print('quantized shape (poincare):', z_q_poincare.shape)
            print('decoder input shape (hyp):', z_q_lorentz.shape)
            print('recon data shape:', x_hat.shape)

        return embedding_loss, x_hat, perplexity, codebook_usage, e_mean

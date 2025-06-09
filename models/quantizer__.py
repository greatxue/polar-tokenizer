import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        # 改进初始化方式
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)
        
        # 添加EMA更新
        self.register_buffer('ema_cluster_size', torch.zeros(n_e))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
        self.register_buffer('ema_updated', torch.zeros(1))
        self.decay = 0.99
        self.eps = 1e-5

    def forward_(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # 归一化处理
        z_flattened_norm = F.normalize(z_flattened, dim=-1)
        embedding_norm = F.normalize(self.embedding.weight, dim=-1)
        
        # 计算余弦相似度而不是欧氏距离
        d = 1 - torch.matmul(z_flattened_norm, embedding_norm.t())
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e,
            device=z.device
        ).scatter_(1, min_encoding_indices, 1)

        # EMA 更新码本
        if self.training:
            self._ema_update(z_flattened, min_encodings)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # 改进的损失计算
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = commitment_loss + self.beta * codebook_loss

        # 改进的梯度传播
        z_q = z + (z_q - z).detach()

        # 计算困惑度和码本使用率
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        used_codes = (min_encodings.sum(0) > 0).float().sum()
        codebook_usage = used_codes / self.n_e

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices, codebook_usage

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # 归一化处理
        z_flattened_norm = F.normalize(z_flattened, dim=-1)
        embedding_norm = F.normalize(self.embedding.weight, dim=-1)
        
        # 计算余弦相似度而不是欧氏距离
        d = 1 - torch.matmul(z_flattened_norm, embedding_norm.t())
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e,
            device=z.device
        ).scatter_(1, min_encoding_indices, 1)

        # EMA 更新码本
        if self.training:
            self._ema_update(z_flattened, min_encodings)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # 先计算码本使用率
        e_mean = torch.mean(min_encodings, dim=0)
        used_codes = (min_encodings.sum(0) > 0).float().sum()
        codebook_usage = used_codes / self.n_e
        
        # 然后计算损失
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        usage_loss = 1.0 - codebook_usage  # 现在可以安全使用 used_codes
        loss = commitment_loss + self.beta * codebook_loss +  usage_loss #0.5
        
        # 计算困惑度
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        # 改进的梯度传播
        z_q = z + (z_q - z).detach()
        
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return loss, z_q, perplexity, min_encodings, min_encoding_indices, codebook_usage
    
    def _ema_update(self, z, encodings):
        """EMA update for the embedding vectors"""
        with torch.no_grad():
            # 计算 EMA cluster size
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                  (1 - self.decay) * encodings_sum
            
            # 计算 EMA weights
            dw = torch.matmul(encodings.t(), z)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            
            # 更新 embedding weights
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.eps) / \
                         (n + self.n_e * self.eps) * n
            
            encode_average = self.ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(encode_average)
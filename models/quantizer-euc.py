import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, use_ema=False, ema_decay=0.99):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # 初始化嵌入向量
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)
        
        # EMA相关变量
        if self.use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(n_e))
            self.register_buffer('ema_w', torch.zeros(n_e, e_dim))
            self.register_buffer('ema_initialized', torch.zeros(1, dtype=torch.bool))

    def calculate_perplexity(self, indices):
        """计算困惑度"""
        encodings = F.one_hot(indices, self.n_e).float().to(device)
        avg_probs = encodings.mean(0)
        return torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    
    def embed_code(self, embed_id):
        """通过嵌入ID返回对应的嵌入向量"""
        return F.embedding(embed_id, self.embedding.weight)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) 并 flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # 计算欧几里得距离
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # 找最近编码
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e,
            device=z.device
        ).scatter_(1, min_encoding_indices, 1)

        # 量化向量
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # EMA更新码本 (如果启用)
        if self.use_ema and self.training:
            # 计算每个码本向量的使用数量和编码总和
            encodings_sum = min_encodings.sum(0)
            encodings_batch = min_encodings.transpose(0, 1) @ z_flattened
            
            # 使用ema更新
            if not self.ema_initialized:
                self.ema_cluster_size.data = encodings_sum
                self.ema_w.data = encodings_batch
                self.ema_initialized.data.fill_(True)
            else:
                self.ema_cluster_size.data = self.ema_cluster_size * self.ema_decay + \
                                          encodings_sum * (1 - self.ema_decay)
                self.ema_w.data = self.ema_w * self.ema_decay + \
                                encodings_batch * (1 - self.ema_decay)
            
            # 计算归一化权重并更新码本
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + self.n_e * 1e-5) * n
            
            # 不使用梯度更新权重
            with torch.no_grad():
                normalized_weights = self.ema_w / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(normalized_weights)
        
        # 损失计算
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = commitment_loss + self.beta * codebook_loss

        # 梯度直通
        z_q = z + (z_q - z).detach()

        # 使用率统计
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        used_codes = (min_encodings.sum(0) > 0).float().sum()
        codebook_usage = used_codes / self.n_e

        # reshape 回 (batch, channel, height, width)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # 确保返回7个值，与超双曲量化器接口匹配
        return loss, z_q, perplexity, min_encodings, min_encoding_indices, codebook_usage, e_mean

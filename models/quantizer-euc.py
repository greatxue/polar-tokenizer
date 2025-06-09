import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        # 初始化嵌入向量，不使用 EMA 更新
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)

    def forward_(self, z):
        # reshape z -> (batch, height, width, channel) 并 flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # 归一化
        z_flattened_norm = F.normalize(z_flattened, dim=-1)
        embedding_norm = F.normalize(self.embedding.weight, dim=-1)

        # 计算余弦相似度
        d = 1 - torch.matmul(z_flattened_norm, embedding_norm.t())

        # 找最近编码
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e,
            device=z.device
        ).scatter_(1, min_encoding_indices, 1)

        # 量化向量
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # 损失计算
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = commitment_loss + self.beta * codebook_loss

        # 梯度直通
        z_q = z + (z_q - z).detach()

        # 困惑度 & 使用率
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        used_codes = (min_encodings.sum(0) > 0).float().sum()
        codebook_usage = used_codes / self.n_e

        # reshape 回 (batch, channel, height, width)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices, codebook_usage

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) 并 flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # 归一化
        z_flattened_norm = F.normalize(z_flattened, dim=-1)
        embedding_norm = F.normalize(self.embedding.weight, dim=-1)

        # 计算余弦相似度
        d = 1 - torch.matmul(z_flattened_norm, embedding_norm.t())

        # 找最近编码
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e,
            device=z.device
        ).scatter_(1, min_encoding_indices, 1)

        # 量化向量
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # 损失计算：去掉 usage_loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = commitment_loss + self.beta * codebook_loss

        # 梯度直通
        z_q = z + (z_q - z).detach()

        # 困惑度 & 使用率
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        used_codes = (min_encodings.sum(0) > 0).float().sum()
        codebook_usage = used_codes / self.n_e

        # reshape 回 (batch, channel, height, width)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices, codebook_usage

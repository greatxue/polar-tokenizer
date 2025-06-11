
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        
        
        # 使用新的编码器和解码器
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        
        # 修改这一行，将128改为512，匹配新编码器的输出通道数
        # 计算编码器输出通道数：h_dim * ch_mult[-1] = h_dim * 4 = 128 * 4 = 512
        self.pre_quantization_conv = nn.Conv2d(
            512, embedding_dim, kernel_size=1, stride=1)  # 原来是128
            
        # 量化器保持不变
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
            
        # 解码器保持相同接口
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        
        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None


    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _, codebook_usage, e_mean  = self.vector_quantization(
            z_e)

        #embedding_loss, z_q, perplexity, _, _, codebook_usage = self.vector_quantization(
           #z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, codebook_usage, e_mean
        #return embedding_loss, x_hat, perplexity, codebook_usage

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
from torch.serialization import add_safe_globals
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
import lpips

# 添加路径
sys.path.append('/ext/work')
sys.path.append('/ext/work/polar')

# 添加安全全局变量
add_safe_globals([
    'numpy._core.multiarray._reconstruct',
    'numpy.core.multiarray._reconstruct',
    'numpy.ndarray'
])

# 导入模型
from polar.models.vqvae import VQVAE

# 自定义数据集类，处理扁平文件夹
class FlatFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None, extensions=('.jpg', '.jpeg', '.png')):
        """
        参数:
            folder_path: 图像文件夹路径
            transform: 图像转换
            extensions: 支持的文件扩展名
        """
        self.folder_path = folder_path
        self.transform = transform
        
        # 获取所有支持的图像文件
        self.image_paths = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(extensions):
                self.image_paths.append(os.path.join(folder_path, filename))
                
        if not self.image_paths:
            raise RuntimeError(f"在 {folder_path} 中未找到任何图像文件")
            
        print(f"找到了 {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # 保存原始尺寸
        
        if self.transform:
            image_tensor = self.transform(image)
        
        # 返回原始图像尺寸，便于后处理
        return image_tensor, os.path.basename(img_path), orig_size

def load_model(checkpoint_path, device):
    """加载VQVAE模型"""
    print(f"从 {checkpoint_path} 加载模型...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 创建模型实例
    model = VQVAE(
        h_dim=128, 
        res_h_dim=32,
        n_res_layers=2, 
        n_embeddings=8192, 
        embedding_dim=64, 
        beta=0.25,
        use_ema=True,
        ema_decay=0.99
    ).to(device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.eval()
    return model

def calculate_metrics(original, reconstruction):
    """计算多种图像质量指标"""
    # 转为numpy数组，值范围0-1
    orig_np = original.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    recon_np = reconstruction.cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    
    # 裁剪值确保在0-1范围内
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)
    
    # MSE
    mse = np.mean((orig_np - recon_np) ** 2)
    
    # PSNR
    psnr_value = psnr(orig_np, recon_np)
    
    # SSIM
    ssim_value = ssim(orig_np, recon_np, channel_axis=2, data_range=1.0)
    
    return {
        'mse': mse,
        'psnr': psnr_value,
        'ssim': ssim_value
    }

def evaluate_folder(model, folder_path, output_dir, batch_size=1, device='cuda'):
    """评估文件夹中所有图像，与inference.py完全一致的重建流程"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 与inference.py完全相同的数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # 获取所有图像路径
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_paths.append(os.path.join(folder_path, filename))
    
    print(f"找到了 {len(image_paths)} 张图像")
    
    # 初始化LPIPS
    lpips_model = lpips.LPIPS(net='alex').to(device) if 'lpips' in sys.modules else None
    
    # 存储所有结果
    all_metrics = []
    all_original = []
    all_reconstructions = []
    
    # 按照inference.py的方式逐张处理图像
    for img_path in tqdm(image_paths, desc="评估图像"):
        # ==== 4. 加载原图 ==== (与inference.py完全一致)
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0)  # 保留原始尺寸
        x = x.to(device)  # 确保在正确的设备上
        
        # 保存文件名
        filename = os.path.basename(img_path)
        
        # ==== 5. 推理 ==== (与inference.py完全一致)
        with torch.no_grad():
            loss, recon_img_tensor, perplexity, codebook_usage, _ = model(x)
            
            # 收集特征用于FID
            all_original.append(x.cpu())
            all_reconstructions.append(recon_img_tensor.cpu())
        
        # ==== 6. 反归一化并直接转成图像 ==== (与inference.py完全一致)
        recon_img_tensor = recon_img_tensor.squeeze(0).cpu()
        recon_img_tensor = recon_img_tensor * 0.5 + 0.5
        recon_img_tensor = torch.clamp(recon_img_tensor, 0, 1)
        recon_img = transforms.ToPILImage()(recon_img_tensor)  # 不 resize！
        
        # ==== 7. 尺寸确认 ==== (与inference.py完全一致)
        print(f"原图尺寸: {img.size}")
        print(f"重构图尺寸: {recon_img.size}")
        
        # 强制重构图与原图尺寸一致（防止细微差异）
        recon_img = recon_img.resize(img.size, Image.BILINEAR)
        
        # 计算评估指标
        # 转为numpy数组用于PSNR和SSIM
        orig_np = np.array(img) / 255.0
        recon_np = np.array(recon_img) / 255.0
        
        # MSE
        mse = np.mean((orig_np - recon_np) ** 2)
        
        # PSNR
        psnr_value = psnr(orig_np, recon_np)
        
        # SSIM
        ssim_value = ssim(orig_np, recon_np, channel_axis=2, data_range=1.0)
        
        # LPIPS
        lpips_value = 0
        if lpips_model is not None:
            # 转换为LPIPS需要的格式
            orig_tensor = transform(img).unsqueeze(0).to(device)
            recon_tensor = transform(recon_img).unsqueeze(0).to(device)
            lpips_value = lpips_model(orig_tensor, recon_tensor).item()
        
        # 收集指标
        metrics = {
            'mse': mse,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'lpips': lpips_value,
            'perplexity': perplexity.item(),
            'codebook_usage': codebook_usage.item(),
            'filename': filename
        }
        
        all_metrics.append(metrics)
        
        # 保存可视化结果
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        
        # 原图
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[0].set_aspect('equal')
        
        # 重建图
        axes[1].imshow(recon_img)
        axes[1].set_title(f'Recon\nPSNR: {psnr_value:.2f}dB')
        axes[1].axis('off')
        axes[1].set_aspect('equal')
        
        # 差异图
        diff = np.abs(orig_np - recon_np) * 5  # 放大差异
        axes[2].imshow(np.clip(diff, 0, 1))
        axes[2].set_title(f'Diff (5x)\nSSIM: {ssim_value:.4f}')
        axes[2].axis('off')
        axes[2].set_aspect('equal')
        
        plt.savefig(os.path.join(output_dir, f"recon_{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存原始和重建图像
        img.save(os.path.join(output_dir, f"orig_{filename}"))
        recon_img.save(os.path.join(output_dir, f"recon_{filename}"))
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key != 'filename':  # 跳过非数值字段
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # 保存汇总结果
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
        f.write("=== 图像重建质量评估 (与inference.py完全一致) ===\n\n")
        
        # 写入平均指标
        f.write("平均指标:\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write("\n")
        
        # 写入各图像详细指标
        f.write("各图像指标:\n")
        for i, metrics in enumerate(all_metrics):
            f.write(f"\n图像 {i+1}: {metrics['filename']}\n")
            for key, value in metrics.items():
                if key != 'filename':
                    f.write(f"{key}: {value:.6f}\n")
    
    # 创建指标比较图
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['mse', 'psnr', 'ssim']
    if lpips_model is not None:
        metrics_to_plot.append('lpips')
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        values = [m[metric] for m in all_metrics]
        filenames = [m['filename'] for m in all_metrics]
        
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), filenames, rotation=45)
        plt.title(f"{metric.upper()}")
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=200)
    
    return all_metrics, all_original, all_reconstructions
def calculate_fid(real_features, fake_features):
    """计算FID值"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # 添加小的正则化以避免数值问题
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps
    
    # 计算平方根差异
    diff = mu_real - mu_fake
    
    # 计算协方差矩阵的平方根的乘积
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # 计算FID
    fid = np.sum(diff * diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    return fid

def main():
    # 参数配置
    folder_path = "/ext/work/4K"  # 包含图像的扁平文件夹
    checkpoint_path = "/ext/work/polar/results_balanced/vqvae_wed_jun_11_16_40_20_2025_step_85000.pth"
    output_dir = "/ext/work/eval_results"
    batch_size = 1  # 一次处理一张图像，避免内存问题
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(checkpoint_path, device)
    
    # 评估文件夹
    metrics, all_original, all_reconstructions = evaluate_folder(
        model, folder_path, output_dir, batch_size, device
    )
    
    print(f"\n评估完成! 结果保存在: {output_dir}")
    print(f"评估了 {len(metrics)} 张图像")
    
    # 尝试计算FID (如果有足够的图像)
    try:
        # 加载预训练的Inception模型
        inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inception.fc = torch.nn.Identity()  # 移除分类层
        inception.eval().to(device)
        
        # 合并所有原始和重建图像
        all_orig_tensor = torch.cat(all_original, dim=0)
        all_recon_tensor = torch.cat(all_reconstructions, dim=0)
        
        # 提取特征
        features_orig = []
        features_recon = []
        
        with torch.no_grad():
            for i in tqdm(range(all_orig_tensor.size(0)), desc="提取特征"):
                # 准备输入 (调整大小至299x299，Inception的输入大小)
                orig = F.interpolate(all_orig_tensor[i:i+1], size=(299, 299), mode='bilinear', align_corners=False).to(device)
                recon = F.interpolate(all_recon_tensor[i:i+1], size=(299, 299), mode='bilinear', align_corners=False).to(device)
                
                # 提取特征
                feat_orig = inception(orig).cpu().numpy()
                feat_recon = inception(recon).cpu().numpy()
                
                features_orig.append(feat_orig)
                features_recon.append(feat_recon)
        
        # 计算FID
        features_orig = np.concatenate(features_orig, axis=0)
        features_recon = np.concatenate(features_recon, axis=0)
        
        fid = calculate_fid(features_orig, features_recon)
        print(f"FID: {fid:.4f}")
        
        # 将FID添加到结果文件
        with open(os.path.join(output_dir, "metrics_summary.txt"), "a") as f:
            f.write(f"\nFID: {fid:.6f}\n")
            if len(metrics) < 50:
                f.write("注意: FID通常需要大量样本才能准确。此值仅供参考。\n")
                
    except Exception as e:
        print(f"FID计算失败: {e}")

if __name__ == "__main__":
    main()

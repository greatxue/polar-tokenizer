import sys
import os
sys.path.append('/ext/work')
sys.path.append('/ext/work/polar')
import torch.nn.functional as F 
import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy import linalg

# 添加安全全局函数
from torch.serialization import add_safe_globals
add_safe_globals([
    'numpy.core.multiarray._reconstruct',
    'numpy._core.multiarray._reconstruct',
    'numpy.ndarray'
])

# 导入模型定义
from polar.models.vqvae import VQVAE

# 尝试导入LPIPS (如果安装了)
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS未安装，将跳过LPIPS评估 (pip install lpips)")

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
    
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

def load_imagenet_data(batch_size):
    """加载ImageNet数据"""
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    val_data = datasets.ImageFolder(
        '/ext/imagenet/val',
        transform=test_transform
    )
    
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    return val_loader

def generate_reconstructions(model, dataloader, device, num_samples=100, save_dir="./eval_samples"):
    """生成原始图像和重建图像对"""
    os.makedirs(f"{save_dir}/original", exist_ok=True)
    os.makedirs(f"{save_dir}/reconstructed", exist_ok=True)
    
    # 收集重建误差
    mse_values = []
    original_tensors = []
    recon_tensors = []
    
    sample_count = 0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="生成样本"):
            if sample_count >= num_samples:
                break
                
            batch_size = min(images.size(0), num_samples - sample_count)
            images = images[:batch_size].to(device)
            
            # 获取重建
            _, recon_images, _, _, _ = model(images)
            
            # 计算MSE
            for i in range(batch_size):
                mse = torch.mean((images[i] - recon_images[i]) ** 2).item()
                mse_values.append(mse)
            
            # 保存用于其他指标的张量
            original_tensors.append(images.cpu())
            recon_tensors.append(recon_images.cpu())
            
            # 保存图像
            for i in range(batch_size):
                img = images[i].cpu()
                recon = recon_images[i].cpu()
                
                # 保存单独的图像
                save_image(img, f"{save_dir}/original/{sample_count+i:05d}.png", normalize=True)
                save_image(recon, f"{save_dir}/reconstructed/{sample_count+i:05d}.png", normalize=True)
                
            sample_count += batch_size
    
    # 创建并保存比较网格
    create_comparison_grid(original_tensors, recon_tensors, f"{save_dir}/comparisons.png")
    
    return np.array(mse_values), original_tensors, recon_tensors

def create_comparison_grid(original_tensors, recon_tensors, save_path, num_examples=8):
    """创建原始图像和重建图像的比较网格"""
    # 从所有批次中收集图像
    all_originals = torch.cat(original_tensors)
    all_recons = torch.cat(recon_tensors)
    
    # 选择一些样本
    indices = np.random.choice(len(all_originals), num_examples, replace=False)
    selected_originals = all_originals[indices]
    selected_recons = all_recons[indices]
    
    # 创建比较网格
    comparison = torch.cat([selected_originals, selected_recons])
    grid = make_grid(comparison, nrow=num_examples, normalize=True)
    
    # 保存网格图像
    plt.figure(figsize=(20, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("原始图像 (上) vs 重建图像 (下)")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 修复calculate_ssim函数
def calculate_ssim(original_dir, recon_dir):
    """计算SSIM指标，处理小图像和多通道图像"""
    print("计算SSIM...")
    ssim_values = []
    
    original_files = sorted(os.listdir(original_dir))
    recon_files = sorted(os.listdir(recon_dir))
    
    for orig_file, recon_file in tqdm(zip(original_files, recon_files), total=len(original_files)):
        try:
            # 加载图像
            orig_img = np.array(Image.open(os.path.join(original_dir, orig_file)).convert('RGB'))
            recon_img = np.array(Image.open(os.path.join(recon_dir, recon_file)).convert('RGB'))
            
            # 确保图像尺寸足够大
            min_dim = min(orig_img.shape[0], orig_img.shape[1])
            
            # 选择合适的窗口大小 (必须是奇数且小于最小维度)
            win_size = min(7, min_dim - (min_dim % 2 == 0))  # 确保是奇数
            if win_size < 3:
                # 图像太小，需要调整大小
                orig_img = np.array(Image.open(os.path.join(original_dir, orig_file)).convert('RGB').resize((24, 24)))
                recon_img = np.array(Image.open(os.path.join(recon_dir, recon_file)).convert('RGB').resize((24, 24)))
                win_size = 7
            
            # 计算SSIM，明确指定channel_axis参数
            ssim_value = ssim(
                orig_img, 
                recon_img, 
                win_size=win_size,
                channel_axis=2,  # RGB图像通道在最后一个维度
                data_range=255
            )
            ssim_values.append(ssim_value)
        except Exception as e:
            print(f"处理图像 {orig_file} 时出错: {e}")
            # 添加一个默认值以保持数组大小一致
            ssim_values.append(0.0)
    
    return np.array(ssim_values)

def calculate_lpips(original_tensors, recon_tensors, device):
    """计算LPIPS感知相似度"""
    if not LPIPS_AVAILABLE:
        return None
    
    print("计算LPIPS...")
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    lpips_values = []
    with torch.no_grad():
        for orig_batch, recon_batch in tqdm(zip(original_tensors, recon_tensors), total=len(original_tensors)):
            # 规范化到LPIPS期望的范围 [-1,1]
            orig_batch = orig_batch.to(device)
            recon_batch = recon_batch.to(device)
            
            batch_size = orig_batch.size(0)
            for i in range(batch_size):
                lpips_value = loss_fn(orig_batch[i:i+1], recon_batch[i:i+1]).item()
                lpips_values.append(lpips_value)
    
    return np.array(lpips_values)

def calculate_fid(features1, features2):
    """计算FID值"""
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    
    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)
    
    # 添加正则化
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    
    diff = mu1 - mu2
    
    # 计算协方差几何平均
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum(diff * diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid

def extract_features(model, dataloader, device, num_samples=100):
    """使用InceptionV3提取特征"""
    from torchvision.models import inception_v3
    
    # 加载InceptionV3
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    inception.fc = torch.nn.Identity()  # 移除分类层
    
    # 准备图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    real_features = []
    recon_features = []
    
    sample_count = 0
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="提取特征"):
            if sample_count >= num_samples:
                break
                
            batch_size = min(images.size(0), num_samples - sample_count)
            images = images[:batch_size].to(device)
            
            # 获取重建
            _, recon_images, _, _, _ = model(images)
            
            # 提取特征
            for i in range(batch_size):
                # 调整大小并预处理用于Inception
                real_img = F.interpolate(images[i:i+1], size=(299, 299), mode='bilinear', align_corners=False)
                recon_img = F.interpolate(recon_images[i:i+1], size=(299, 299), mode='bilinear', align_corners=False)
                
                # 提取特征
                real_feat = inception(real_img)
                recon_feat = inception(recon_img)
                
                real_features.append(real_feat.cpu().numpy())
                recon_features.append(recon_feat.cpu().numpy())
            
            sample_count += batch_size
    
    return np.concatenate(real_features), np.concatenate(recon_features)

def main():
    # 配置参数
    checkpoint_path = "/ext/work/polar/results_balanced/vqvae_wed_jun_11_16_40_20_2025_step_15000.pth"
    batch_size = 32
    num_samples = 100  # 用于评估的样本数
    output_dir = "./comprehensive_eval"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和数据
    model = load_model(checkpoint_path, device)
    dataloader = load_imagenet_data(batch_size)
    
    # 生成重建图像
    print(f"为 {num_samples} 个样本生成重建...")
    mse_values, original_tensors, recon_tensors = generate_reconstructions(
        model, dataloader, device, num_samples, save_dir=output_dir
    )
    
    # 计算SSIM
    ssim_values = calculate_ssim(f"{output_dir}/original", f"{output_dir}/reconstructed")
    
    # 计算LPIPS (如果可用)
    if LPIPS_AVAILABLE:
        lpips_values = calculate_lpips(original_tensors, recon_tensors, device)
    else:
        lpips_values = None
    
    # 提取特征和计算FID
    print("提取特征和计算FID...")
    real_features, recon_features = extract_features(model, dataloader, device, num_samples)
    fid_value = calculate_fid(real_features, recon_features)
    
    # 打印和保存结果
    print("\n" + "="*50)
    print("VQVAE评估结果:")
    print(f"MSE (重建误差): {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}")
    print(f"SSIM: {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}")
    if lpips_values is not None:
        print(f"LPIPS: {np.mean(lpips_values):.6f} ± {np.std(lpips_values):.6f}")
    print(f"FID: {fid_value:.6f}")
    print("="*50)
    
    # 创建结果可视化
    plt.figure(figsize=(15, 10))
    
    # MSE分布
    plt.subplot(2, 2, 1)
    plt.hist(mse_values, bins=30, alpha=0.7)
    plt.axvline(np.mean(mse_values), color='r', linestyle='--')
    plt.title(f'MSE分布 (均值: {np.mean(mse_values):.4f})')
    plt.xlabel('MSE')
    plt.ylabel('频率')
    
    # SSIM分布
    plt.subplot(2, 2, 2)
    plt.hist(ssim_values, bins=30, alpha=0.7)
    plt.axvline(np.mean(ssim_values), color='r', linestyle='--')
    plt.title(f'SSIM分布 (均值: {np.mean(ssim_values):.4f})')
    plt.xlabel('SSIM')
    plt.ylabel('频率')
    
    # LPIPS分布 (如果可用)
    if lpips_values is not None:
        plt.subplot(2, 2, 3)
        plt.hist(lpips_values, bins=30, alpha=0.7)
        plt.axvline(np.mean(lpips_values), color='r', linestyle='--')
        plt.title(f'LPIPS分布 (均值: {np.mean(lpips_values):.4f})')
        plt.xlabel('LPIPS')
        plt.ylabel('频率')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_metrics.png")
    plt.close()
    
    # 保存数值结果到文件
    with open(f"{output_dir}/evaluation_results.txt", "w") as f:
        f.write(f"评估样本数: {num_samples}\n")
        f.write(f"MSE: {np.mean(mse_values):.6f} ± {np.std(mse_values):.6f}\n")
        f.write(f"SSIM: {np.mean(ssim_values):.6f} ± {np.std(ssim_values):.6f}\n")
        if lpips_values is not None:
            f.write(f"LPIPS: {np.mean(lpips_values):.6f} ± {np.std(lpips_values):.6f}\n")
        f.write(f"FID: {fid_value:.6f}\n")

if __name__ == "__main__":
    main()

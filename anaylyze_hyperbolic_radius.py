import sys
sys.path.append('/ext/work/polar')

import torch
import matplotlib.pyplot as plt
import numpy as np
from models.vqvae import VQVAE
from models.quantizer import poincare_to_hyperboloid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import os
from models.quantizer import exp_map_to_lorentz

def analyze_radius(checkpoint_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = VQVAE(
        h_dim=128,
        res_h_dim=32,
        n_res_layers=2,
        n_embeddings=8192,
        embedding_dim=64,
        beta=0.25,
        save_img_embedding_map=False,
        use_ema=True,      # 添加这行
        ema_decay=0.99 

    )
    
    # 加载检查点，但正确处理嵌套字典结构
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"检查点结构: {list(checkpoint.keys())}")
    
    if "model" in checkpoint:
        print("从复合检查点中提取模型权重...")
        model.load_state_dict(checkpoint["model"])
    else:
        print("尝试直接加载检查点...")
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # 准备数据 - 关键修改：添加固定大小的Resize变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 固定图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    # 检查数据集路径
    train_dir = os.path.join(data_path, "train")
    print(f"加载数据目录: {train_dir}")
    if not os.path.exists(train_dir):
        raise RuntimeError(f"目录不存在: {train_dir}")
    
    # 加载单个样本来测试
    print("尝试加载单个样本进行测试...")
    sample_folders = os.listdir(train_dir)[:3]
    for folder in sample_folders:
        folder_path = os.path.join(train_dir, folder)
        if os.path.isdir(folder_path):
            sample_files = os.listdir(folder_path)[:2]
            for img_file in sample_files:
                img_path = os.path.join(folder_path, img_file)
                print(f"测试加载: {img_path}")
                try:
                    from PIL import Image
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0)
                    print(f"成功加载，形状: {img_tensor.shape}")
                    break
                except Exception as e:
                    print(f"加载出错: {e}")
    
    # 加载整个数据集，使用较小的batch_size和num_workers
    try:
        dataset = datasets.ImageFolder(train_dir, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=8,  # 减小批次大小
            shuffle=True, 
            num_workers=1  # 减少工作进程数
        )
        print(f"成功创建DataLoader，数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"创建DataLoader失败: {e}")
        raise
    
    # 分析半径分布
    radii = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 20:  # 限制处理的批次数
                break
            
            try:
                x, _ = batch
                print(f"处理批次 {i+1}/20，形状: {x.shape}...")
                x = x.to(device)
                
                # 获取编码器输出
                z_e = model.encoder(x)
                z_e = model.pre_quantization_conv(z_e)
                
                # 转换为扁平形式
                B, C, H, W = z_e.shape
                z_flat = z_e.permute(0,2,3,1).reshape(-1, C)
                
                
                # 转换到超曲面并计算半径
                u_hyp = exp_map_to_lorentz(z_flat)
                u0 = u_hyp[..., 0]
                r = torch.acosh(torch.clamp(u0, min=1.0 + 1e-5))
                radii.append(r.cpu())
            except Exception as e:
                print(f"处理批次 {i+1} 时出错: {e}")
                continue
    
    # 合并所有半径数据
    if not radii:
        raise RuntimeError("没有收集到任何半径数据，请检查前面的错误")
        
    all_radii = torch.cat(radii, dim=0).numpy()
    
    # 计算统计数据
    mean_radius = float(all_radii.mean())
    median_radius = float(np.median(all_radii))
    max_radius = float(all_radii.max())
    min_radius = float(all_radii.min())
    p95 = float(np.percentile(all_radii, 95))
    p99 = float(np.percentile(all_radii, 99))
    
    # 输出结果
    print("\n===== 双曲空间半径分布统计 =====")
    print(f"最小半径: {min_radius:.4f}")
    print(f"平均半径: {mean_radius:.4f}")
    print(f"中位数半径: {median_radius:.4f}")
    print(f"95百分位半径: {p95:.4f}")
    print(f"99百分位半径: {p99:.4f}")
    print(f"最大半径: {max_radius:.4f}")
    print("\n推荐的max_radius设置值:")
    print(f"保守设置 (99百分位的1.1倍): {p99*1.1:.4f}")
    print(f"中等设置 (99百分位的1.25倍): {p99*1.25:.4f}")
    print(f"宽松设置 (最大值的1.1倍): {max_radius*1.1:.4f}")
    print(f"z_e统计: min={z_e.min().item():.4f}, max={z_e.max().item():.4f}")
    print(f"z_flat范数: min={torch.norm(z_flat, dim=-1).min().item():.4f}, max={torch.norm(z_flat, dim=-1).max().item():.4f}")

    # 在计算半径时添加更详细的检查
    print(f"u_hyp[0] 第一个元素: {u_hyp[0, 0].item():.6f}")
    print(f"当前批次半径: min={r.min().item():.4f}, max={r.max().item():.4f}, std={r.std().item():.4f}")

    # 在合并半径数据前打印每个批次的样本数
    radii_count = sum(tensor.size(0) for tensor in radii)
    print(f"收集了{radii_count}个样本的半径")
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_radii, bins=100, alpha=0.7)
    plt.axvline(mean_radius, color='r', linestyle='--', label=f'平均值 = {mean_radius:.2f}')
    plt.axvline(p99, color='g', linestyle='--', label=f'99百分位 = {p99:.2f}')
    plt.axvline(max_radius, color='b', linestyle='--', label=f'最大值 = {max_radius:.2f}')
    plt.xlabel('双曲半径')
    plt.ylabel('频率')
    plt.title('编码器输出的双曲半径分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hyperbolic_radius_distribution.png')
    print("半径分布直方图已保存为 'hyperbolic_radius_distribution.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    parser.add_argument("--data_path", required=True, help="数据集路径")
    args = parser.parse_args()
    
    try:
        analyze_radius(args.checkpoint, args.data_path)
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

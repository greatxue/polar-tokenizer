import numpy as np
import torch
import sys
import os
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from PIL import ImageFile, Image
import time
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from models.vqvae import VQVAE
import json
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置随机种子以获得可重现的结果
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 参数解析
def parse_args():
    timestamp = utils.readable_timestamp()
    parser = argparse.ArgumentParser(description="增强版VQVAE训练")
    
    # 基本训练参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=4000080)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=1000, help="验证间隔")
    parser.add_argument("--dataset", type=str, default='IMAGENET')
    
    # 模型参数
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=8192)
    
    # 正则化与稳定性参数
    parser.add_argument("--beta", type=float, default=0.25, help="承诺损失权重")
    parser.add_argument("--entropy_weight", type=float, default=0.4, help="编码熵正则化权重")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="码本EMA更新率")
    parser.add_argument("--use_ema", action="store_true", help="使用EMA更新码本")
    parser.add_argument("--use_reset", action="store_true", help="使用码本重置机制")
    parser.add_argument("--reset_interval", type=int, default=1000, help="码本重置检查间隔")
    parser.add_argument("--jitter_prob", type=float, default=0.12, help="颜色抖动概率")
    
    # 学习率调度
    parser.add_argument("--use_lr_scheduler", action="store_true", help="使用学习率调度器")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "plateau"], help="学习率调度器类型")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5, help="学习率衰减因子")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="最小学习率")
    
    # 混合精度训练
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    
    # 恢复训练
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")
    parser.add_argument("--checkpoint", type=str, default=None, help="恢复检查点路径")
    
    # 保存和可视化
    parser.add_argument("--save", action="store_true", help="是否保存模型")
    parser.add_argument("--save_interval", type=int, default=5000, help="保存检查点间隔步数")
    parser.add_argument("--filename", type=str, default=timestamp)
    parser.add_argument("--output_dir", type=str, default="./new_results")
    
    # 码本使用率改进策略
    parser.add_argument("--usage_threshold", type=float, default=0.1, help="低使用率阈值(相对于平均值)")
    
    args = parser.parse_args()
    return args

# 可视化工具函数
def visualize_reconstructions(model, validation_loader, device, step, output_dir):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(validation_loader))
        images = images.to(device)[:8]  # 取前8张图片
        
        # 获取重建结果
        _, reconstructions, _, _, _ = model(images)
        
        # 将输入和重建结果转换为网格图像
        comparison = torch.cat([images, reconstructions])
        
        # 修改这一行 - 移除range参数
        comparison = make_grid(comparison, nrow=8, normalize=True)  # 删除range=(-1, 1)参数
        
        plt.figure(figsize=(12, 6))
        plt.imshow(comparison.cpu().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"Step {step}: Original (top) vs Reconstructed (bottom)")
        
        # 保存图像
        os.makedirs(f"{output_dir}/reconstructions", exist_ok=True)
        plt.savefig(f"{output_dir}/reconstructions/recon_step_{step}.png", bbox_inches='tight')
        plt.close()
    
    model.train()

# 可视化码本使用情况
def visualize_codebook_usage(usage_history, step, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(usage_history['steps'], usage_history['usage'], label='使用率')
    
    # 获取最近的趋势
    if len(usage_history['steps']) > 20:
        window = 20
        usage_trend = [usage_history['usage'][i+window] - usage_history['usage'][i] 
                      for i in range(len(usage_history['steps']) - window)]
        trend_steps = usage_history['steps'][window:]
        plt.plot(trend_steps, usage_trend, 'r--', label='趋势')
    
    plt.xlabel("训练步数")
    plt.ylabel("码本使用率")
    plt.title(f"码本使用率历史 (步数 {step})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(f"{output_dir}/codebook", exist_ok=True)
    plt.savefig(f"{output_dir}/codebook/usage_step_{step}.png", bbox_inches='tight')
    plt.close()

# 训练循环
def train(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    with open(f"{args.output_dir}/config_{args.filename}.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据集...")
    training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)
    print(f"数据集大小: {len(training_data)} 训练样本, {len(validation_data)} 验证样本")
    print(f"批次大小: {args.batch_size}, 每轮 {len(training_loader)} 批次")
    
    # 初始化模型
    print("创建模型...")
    model = VQVAE(
        args.n_hiddens, 
        args.n_residual_hiddens,
        args.n_residual_layers, 
        args.n_embeddings, 
        args.embedding_dim, 
        args.beta,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay
    ).to(device)
    
    # 优化器设置
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    
    # 混合精度训练设置
    scaler = GradScaler() if args.fp16 else None
    
    # 学习率调度器
    scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=args.n_updates,
                eta_min=args.min_lr
            )
        elif args.lr_scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.lr_decay_factor,
                patience=3000,
                verbose=True,
                min_lr=args.min_lr
            )
    
    # 训练状态变量
    start_step = 0
    best_val_loss = float('inf')
    
    # 结果跟踪
    results = {
        'n_updates': 0,
        'recon_errors': [],
        'loss_vals': [],
        'perplexities': [],
        'codebook_usage': []
    }
    
    # 码本使用率历史
    usage_history = {'steps': [], 'usage': []}
    
    # 从检查点恢复
    if args.resume and args.checkpoint:
        print(f"从检查点恢复: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            results = checkpoint.get('results', results)
            start_step = results.get('n_updates', 0) + 1
            if 'usage_history' in checkpoint:
                usage_history = checkpoint['usage_history']
            if scheduler and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if args.fp16 and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            
            print(f"成功恢复到步骤 {start_step}")
        except Exception as e:
            print(f"恢复失败: {e}")
            return
    
    # 训练设置
    model.train()
    print(f"开始训练，共 {args.n_updates} 步...")
    
    # 计算每个epoch的迭代次数
    iterations_per_epoch = len(training_loader)
    current_epoch = start_step // iterations_per_epoch
    
    # 创建训练迭代器
    train_iterator = iter(training_loader)
    
    # 主训练循环
    for step in range(start_step, args.n_updates):
        # 获取下一批数据
        try:
            x, _ = next(train_iterator)
        except StopIteration:
            # 完成一个epoch
            current_epoch += 1
            print(f"\n完成Epoch {current_epoch}")
            train_iterator = iter(training_loader)
            x, _ = next(train_iterator)
        
        x = x.to(device)
        optimizer.zero_grad()
        
        # 混合精度训练
        if args.fp16:
            with autocast():
                embedding_loss, x_hat, perplexity, codebook_usage, e_mean = model(x)
                recon_loss = torch.mean((x_hat - x)**2) / x_train_var
                codebook_entropy = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
                loss = recon_loss + args.beta * embedding_loss + args.entropy_weight * codebook_entropy
            
            # 缩放损失、反向传播并更新
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            embedding_loss, x_hat, perplexity, codebook_usage, e_mean = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            codebook_entropy = -torch.sum(e_mean * torch.log(e_mean + 1e-10))
            loss = recon_loss + args.beta * embedding_loss + args.entropy_weight * codebook_entropy
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 更新结果
        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["codebook_usage"].append(codebook_usage.item())
        results["n_updates"] = step
        
        # 更新码本使用率历史
        usage_history['steps'].append(step)
        usage_history['usage'].append(codebook_usage.item())
        if len(usage_history['steps']) > 5000:  # 限制历史记录长度
            usage_history['steps'].pop(0)
            usage_history['usage'].pop(0)
        
        # 学习率调度
        if args.use_lr_scheduler:
            if args.lr_scheduler == "cosine":
                scheduler.step()
            elif args.lr_scheduler == "plateau" and step % 500 == 0 and step > 0:
                # 每500步使用最近的损失平均值更新scheduler
                recent_loss = np.mean(results["loss_vals"][-500:])
                scheduler.step(recent_loss)
        
        # 码本重置机制
        if args.use_reset and step % args.reset_interval == 0 and step > 0:
            print("检查码本使用率并重置低使用率编码...")
            with torch.no_grad():
                # 统计编码使用情况
                usage_count = torch.zeros(args.n_embeddings).to(device)
                
                # 在少量批次上评估使用情况
                model.eval()
                for eval_step, (eval_x, _) in enumerate(training_loader):
                    if eval_step >= 20:  # 只用20个批次评估
                        break
                    eval_x = eval_x.to(device)
                    _, _, _, _, batch_e_mean = model(eval_x)
                    # 累计使用率
                    usage_count += batch_e_mean.mean(0) * args.batch_size
                
                # 找出使用率低的编码
                usage_prob = usage_count / usage_count.sum()
                low_usage = usage_prob < (args.usage_threshold / args.n_embeddings)  # 阈值：平均使用率的X%
                high_usage = usage_prob > (2.0 / args.n_embeddings)  # 使用率高的
                
                if low_usage.any():
                    num_reset = low_usage.sum().item()
                    print(f"重置 {num_reset} 个低使用率的码本向量")
                    
                    # 重置低使用率的编码向量 - 使用高使用率向量作为基础
                    high_indices = torch.where(high_usage)[0]
                    low_indices = torch.where(low_usage)[0]
                    
                    if len(high_indices) > 0:
                        for idx in low_indices:
                            # 随机选择一个高使用率向量
                            high_idx = high_indices[torch.randint(0, len(high_indices), (1,))]
                            # 获取该向量并添加噪声
                            vector = model.vector_quantization.embedding.weight[high_idx].clone()
                            noise = torch.randn_like(vector) * 0.1  # 10%的噪声
                            model.vector_quantization.embedding.weight.data[idx] = vector + noise
                
                model.train()
        
        # 验证步骤
        if step % args.val_interval == 0 and step > 0:
            model.eval()
            val_losses = []
            val_recon_losses = []
            val_perplexities = []
            val_usages = []
            
            with torch.no_grad():
                # 只使用部分验证集以加速验证
                for i, (val_x, _) in enumerate(validation_loader):
                    if i >= 10:  # 限制验证批次数
                        break
                    
                    val_x = val_x.to(device)
                    val_embedding_loss, val_x_hat, val_perplexity, val_codebook_usage, val_e_mean = model(val_x)
                    
                    val_recon_loss = torch.mean((val_x_hat - val_x)**2) / x_train_var
                    val_codebook_entropy = -torch.sum(val_e_mean * torch.log(val_e_mean + 1e-10))
                    val_loss = val_recon_loss + args.beta * val_embedding_loss + args.entropy_weight * val_codebook_entropy
                    
                    val_losses.append(val_loss.item())
                    val_recon_losses.append(val_recon_loss.item())
                    val_perplexities.append(val_perplexity.item())
                    val_usages.append(val_codebook_usage.item())
            
            avg_val_loss = np.mean(val_losses)
            avg_val_recon = np.mean(val_recon_losses)
            avg_val_perplexity = np.mean(val_perplexities)
            avg_val_usage = np.mean(val_usages)
            
            print(f"\n验证 (步骤 {step}):")
            print(f"    损失: {avg_val_loss:.6f}")
            print(f"    重建误差: {avg_val_recon:.6f}")
            print(f"    困惑度: {avg_val_perplexity:.2f}")
            print(f"    码本使用率: {avg_val_usage:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if args.save:
                    best_path = f"{args.output_dir}/vqvae_{args.filename}_best.pth"
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'results': results,
                        'usage_history': usage_history,
                        'hyperparameters': args.__dict__,
                        'best_val_loss': best_val_loss,
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'scaler': scaler.state_dict() if scaler else None
                    }, best_path)
                    print(f"保存最佳模型至: {best_path}")
            
            # 生成重建可视化
            if args.save:
                visualize_reconstructions(model, validation_loader, device, step, args.output_dir)
            
            model.train()
        
        # 定期打印日志
        if step % args.log_interval == 0:
            # 计算最近的统计数据
            recent_recon = np.mean(results["recon_errors"][-args.log_interval:])
            recent_loss = np.mean(results["loss_vals"][-args.log_interval:])
            recent_perplexity = np.mean(results["perplexities"][-args.log_interval:])
            recent_usage = codebook_usage.item()
            
            # 计算使用率趋势
            if len(usage_history['usage']) >= 20:
                recent_trend = np.mean(usage_history['usage'][-10:]) - np.mean(usage_history['usage'][-20:-10])
                trend_symbol = "↑" if recent_trend > 0 else "↓"
                trend_info = f"码本趋势: {trend_symbol} {abs(recent_trend):.4f}"
            else:
                trend_info = ""
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"步骤 {step}/{args.n_updates} (Epoch {current_epoch}) | "
                  f"重建误差: {recent_recon:.6f} | "
                  f"损失: {recent_loss:.6f} | "
                  f"困惑度: {recent_perplexity:.2f} | "
                  f"码本使用率: {recent_usage:.4f} | "
                  f"学习率: {current_lr:.6f} | "
                  f"{trend_info}")
            
            # 可视化码本使用率
            if args.save and step > 0:
                visualize_codebook_usage(usage_history, step, args.output_dir)
        
        # 定期保存检查点
        if args.save and step % args.save_interval == 0 and step > 0:
            checkpoint_path = f"{args.output_dir}/vqvae_{args.filename}_step_{step}.pth"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'results': results,
                'usage_history': usage_history,
                'hyperparameters': args.__dict__,
                'best_val_loss': best_val_loss,
                'scheduler': scheduler.state_dict() if scheduler else None,
                'scaler': scaler.state_dict() if scaler else None
            }, checkpoint_path)
            print(f"保存检查点至: {checkpoint_path}")
    
    # 保存最终模型
    if args.save:
        final_path = f"{args.output_dir}/vqvae_{args.filename}_final.pth"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'results': results,
            'usage_history': usage_history,
            'hyperparameters': args.__dict__,
            'best_val_loss': best_val_loss,
            'scheduler': scheduler.state_dict() if scheduler else None,
            'scaler': scaler.state_dict() if scaler else None
        }, final_path)
        print(f"保存最终模型至: {final_path}")
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")

if __name__ == "__main__":
    args = parse_args()
    train(args)

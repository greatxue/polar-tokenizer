import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.serialization import safe_globals, add_safe_globals

# Add numpy globals to safe list
add_safe_globals([
    'numpy._core.multiarray._reconstruct',
    'numpy.core.multiarray._reconstruct',
    'numpy.ndarray'
])

# ==== 1. 导入模型结构 ====
from vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VQVAE(
    h_dim=128,
    res_h_dim=32,
    n_res_layers=2,
    n_embeddings=512,
    embedding_dim=64,
    beta=0.25
).to(device)

# ==== 2. 加载模型权重 ====
checkpoint = torch.load(
    '/home/zhongkai/project/polar-tokenizer/results/vqvae_data_mon_jun_9_07_27_42_2025.pth',
    map_location=device,
    weights_only=False
)

if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()

# ==== 3. 图像预处理：不resize ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ==== 4. 加载原图 ====
# ==== 4. 加载原图 ====
img_path = '/home/zhongkai/project/polar-tokenizer/data/imagenet/train/n01443537/ILSVRC2012_val_00000002.jpeg'
img = Image.open(img_path).convert('RGB')
x = transform(img).unsqueeze(0).to(device)  # 保留原始尺寸

# ==== 5. 推理 ====
with torch.no_grad():
    loss, z_q, perplexity, codebook_usage = model(x)
    print(f"Codebook usage: {codebook_usage.item():.2%}")
    print(f"Perplexity: {perplexity.item():.2f}")

# ==== 6. 反归一化并直接转成图像 ====
z_q = z_q.squeeze(0).cpu()
z_q = z_q * 0.5 + 0.5
z_q = torch.clamp(z_q, 0, 1)
recon_img = transforms.ToPILImage()(z_q)   # 不 resize！

# ==== 7. 尺寸确认 ====
print(f"原图尺寸: {img.size}")
print(f"重构图尺寸: {recon_img.size}")   

# 强制重构图与原图尺寸一致（防止细微差异）
recon_img = recon_img.resize(img.size, Image.BILINEAR)

# ==== 8. 可视化 + 保存 ====
fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

axes[0].imshow(img)
axes[0].set_title('Origin')
axes[0].axis('off')
axes[0].set_aspect('equal')

axes[1].imshow(recon_img)
axes[1].set_title('Recon')
axes[1].axis('off')
axes[1].set_aspect('equal')

plt.savefig('reconstruction.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

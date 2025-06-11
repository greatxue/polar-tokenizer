import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torch.serialization import safe_globals, add_safe_globals
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add numpy globals to safe list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
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
    n_embeddings=8192,  # 关键参数，确保与训练时相同
    embedding_dim=64,   # 关键参数，确保与训练时相同
    beta=0.25,
    save_img_embedding_map=False
)


# ==== 2. 加载模型权重 ====
checkpoint = torch.load(
    '/ext/work/results/vqvae_data_wed_jun_11_12_31_45_2025.pth',
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint['model'])
model = model.to(device)

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
img_path = '/ext/work/ILSVRC2012_img_val/train/n03642806/ILSVRC2012_val_00000660.JPEG'
#img_path = '/ext/work/ILSVRC2012_img_val/val/n04335435/ILSVRC2012_val_00033110.JPEG'
#img_path = '/ext/work/ILSVRC2012_img_val/val/n04428191/ILSVRC2012_val_00035701.JPEG'
#img_path = '/ext/work/ILSVRC2012_img_val/train/n04371774/ILSVRC2012_val_00005319.JPEG'
#img_path = '/ext/work/ILSVRC2012_img_val/val/n04019541/ILSVRC2012_val_00019105.JPEG'
img = Image.open(img_path).convert('RGB')
x = transform(img).unsqueeze(0).to(device)  # 保留原始尺寸
x = x.to(device) 
# ==== 5. 推理 ====
with torch.no_grad():
    loss, recon_img_tensor, perplexity, codebook_usage , _= model(x)
    print(f"Codebook usage: {codebook_usage.item():.2%}")
    print(f"Perplexity: {perplexity.item():.2f}")

# ==== 6. 反归一化并直接转成图像 ====
recon_img_tensor = recon_img_tensor.squeeze(0).cpu()
recon_img_tensor = recon_img_tensor * 0.5 + 0.5
recon_img_tensor = torch.clamp(recon_img_tensor, 0, 1)
recon_img = transforms.ToPILImage()(recon_img_tensor)  # 不 resize！

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

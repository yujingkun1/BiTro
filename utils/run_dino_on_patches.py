#!/usr/bin/env python3
"""
独立加载 DINOv3 模型并对指定目录下的 patch 图像逐个提取特征，检查是否退化。
"""
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def load_dinov3_model(dinov3_repo_dir, dinov3_weights_path, device):
    print("Loading DINOv3 model...")
    model = torch.hub.load(dinov3_repo_dir, 'dinov3_vitl16', source='local',
                           weights=dinov3_weights_path, trust_repo=True)
    model.to(device)
    model.eval()
    print("Loaded DINOv3.")
    return model

def main(patch_dir, dinov3_repo_dir="/data/yujk/hovernet2feature/dinov3", dinov3_weights="/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", device='cpu', max_patches=200):
    files = sorted([os.path.join(patch_dir,f) for f in os.listdir(patch_dir) if f.endswith('.png')])[:max_patches]
    if len(files)==0:
        print("No patch images found in", patch_dir)
        return

    model = load_dinov3_model(dinov3_repo_dir, dinov3_weights, device)
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    tensors=[]
    for f in files:
        img = Image.open(f).convert('RGB')
        t = preprocess(img)
        tensors.append(t)
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(batch)
        # handle different output shapes
        if isinstance(outputs, tuple):
            out = outputs[0]
        else:
            out = outputs
        if len(out.shape)==4:
            feats = out.mean(dim=[2,3]).cpu().numpy()
        elif len(out.shape)==3:
            feats = out.mean(dim=1).cpu().numpy()
        else:
            feats = out.cpu().numpy()

    print("feats shape:", feats.shape)
    print("unique rows (rounded6):", np.unique(np.round(feats,6), axis=0).shape[0])
    col_std = feats.std(axis=0)
    print("col std percentiles:", np.percentile(col_std, [0,1,5,25,50,75,95,99,100]))
    print("total var:", np.var(feats))

if __name__ == "__main__":
    # 默认对我们保存的 preview48 目录运行
    patch_dir = "/data/yujk/hovernet2feature/xenium_xenium_dinov3_features/XENIUM_SAMPLE_patches_preview48"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(patch_dir, device=device, max_patches=100)




#!/usr/bin/env python3
"""
小规模验证：对 Xenium 数据提取前 N 个细胞的 DINO 特征并检查是否存在退化（所有行相同或方差接近0）。
"""
import os
import sys
import numpy as np
import pandas as pd

# 确保可以在脚本直接从 Cell2Gene 目录运行时导入本地模块
HERE = os.path.dirname(os.path.abspath(__file__))
CELL2GENE_ROOT = os.path.dirname(HERE)
if CELL2GENE_ROOT not in sys.path:
    sys.path.insert(0, CELL2GENE_ROOT)

# 直接导入本地模块（避免依赖 package 安装）
from utils.extract_xenium_features_dinov3 import XeniumCellFeatureExtractor
from utils.extract_spatial_features_dinov3 import HESTCellFeatureExtractor


def main(n_cells=200, dinov3_model_path="/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"):
    out_dir = "/data/yujk/hovernet2feature/xenium_xenium_dinov3_features"
    xenium_root = "/data/yujk/hovernet2feature/xenium"

    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

    extractor = XeniumCellFeatureExtractor(
        xenium_root_dir=xenium_root,
        output_dir=out_dir,
        dinov3_model_path=dinov3_model_path,
        device=device,
        cell_patch_size=48,
        dino_batch_size=64,
        cell_batch_size=1000,
        num_workers=1,   # 单线程以确保确定性
    )

    sample_data = extractor.load_sample_data(sample_id=None)
    cellvit_df = sample_data['cellvit_df']
    wsi_path = sample_data['wsi_path']

    # 打开WSI（与主流程一致）
    try:
        import openslide
        try:
            wsi = openslide.OpenSlide(wsi_path)
        except Exception:
            # 回退到 SimpleWSI，本地导入已在顶部处理
            wsi = HESTCellFeatureExtractor._SimpleWSI(wsi_path)
    except Exception:
        wsi = HESTCellFeatureExtractor._SimpleWSI(wsi_path)

    # 选择级别（复用主流程逻辑）
    if getattr(wsi, 'level_count', 1) > 1:
        level = 1
    else:
        level = 0

    subset_df = cellvit_df.iloc[:min(n_cells, len(cellvit_df))]
    print(f"测试将提取 {len(subset_df)} 个细胞的 patch 并计算 DINO 特征（device={device}）")

    # 并行提取但 num_workers=1 保证顺序与可重复性
    patches, positions = extractor._extract_patches_parallel(subset_df, wsi, level, 0)
    print(f"已提取 patches: {len(patches)}")

    dino_feats = extractor.extract_dino_features(patches)
    print("DINO 特征形状:", dino_feats.shape)

    # 诊断
    nan_count = np.isnan(dino_feats).sum()
    print("NaN count:", nan_count)
    col_std = dino_feats.std(axis=0)
    print("列 std 分位数:", np.percentile(col_std, [0,1,5,25,50,75,95,99,100]))
    total_var = np.var(dino_feats)
    print("总体方差:", total_var)
    unique_rows = np.unique(np.round(dino_feats, 6), axis=0)
    print("唯一行数 (round6):", unique_rows.shape[0])

    # PCA quick check
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, dino_feats.shape[1]))
        proj = pca.fit_transform(dino_feats)
        print("PCA explained variance ratio (top10):", pca.explained_variance_ratio_[:10])
    except Exception as e:
        print("PCA failed:", e)

    # 保存诊断文件
    np.save(os.path.join(out_dir, "test_small_dino_feats.npy"), dino_feats)
    print("诊断特征已保存到:", os.path.join(out_dir, "test_small_dino_feats.npy"))


if __name__ == "__main__":
    main(n_cells=200)



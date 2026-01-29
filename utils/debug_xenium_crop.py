#!/usr/bin/env python3
"""
小工具：直接使用 PIL 从 Xenium WSI 上按 cells.csv 的中心坐标裁剪前 N 个 patch 并保存，便于排查坐标与图像的匹配问题。
用法示例：
  python Cell2Gene/utils/debug_xenium_crop.py --wsi /data/yujk/hovernet2feature/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif \
       --cells /data/yujk/hovernet2feature/xenium/outs/cells.csv --n 10 --patch 36 --out /tmp/xenium_debug
"""
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image, ImageFile

# 允许处理超大TIFF，避免 DecompressionBombError
Image.MAX_IMAGE_PIXELS = None
# 允许加载截断图像（防御性设置）
ImageFile.LOAD_TRUNCATED_IMAGES = True


def crop_and_save(wsi_path, cells_csv, out_dir, n=10, patch_size=36, mode='first', grid_size=(4,4), rng_seed=42):
    os.makedirs(out_dir, exist_ok=True)

    # 读取细胞中心
    df = pd.read_csv(cells_csv)
    if 'x_centroid' not in df.columns or 'y_centroid' not in df.columns:
        raise RuntimeError("cells.csv 缺少 'x_centroid' 或 'y_centroid' 列")

    # 选择采样索引
    total = len(df)
    selected_indices = []
    if mode == 'first':
        selected_indices = list(range(min(n, total)))
    elif mode == 'spaced':
        step = max(1, total // n)
        selected_indices = list(range(0, total, step))[:n]
    elif mode == 'random':
        rng = np.random.default_rng(rng_seed)
        selected_indices = rng.choice(total, size=min(n, total), replace=False).tolist()
    elif mode == 'grid':
        gx, gy = grid_size
        # determine bounding box
        xs = df['x_centroid'].astype(float).values
        ys = df['y_centroid'].astype(float).values
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        # build grid and pick one centroid per cell (closest to cell center)
        for ix in range(gx):
            for iy in range(gy):
                cx = minx + (ix + 0.5) * (maxx - minx) / gx
                cy = miny + (iy + 0.5) * (maxy - miny) / gy
                # compute distances to cell centers
                d2 = (xs - cx)**2 + (ys - cy)**2
                # pick nearest if within the global set
                idx = int(np.argmin(d2))
                selected_indices.append(idx)
        # deduplicate and limit to n
        selected_indices = list(dict.fromkeys(selected_indices))[:min(n, len(selected_indices))]
    else:
        raise ValueError(f"未知采样模式: {mode}")

    # If user requested fewer than grid picks, truncate
    if len(selected_indices) > n:
        selected_indices = selected_indices[:n]

    # 打开图像（PIL）
    img = Image.open(wsi_path).convert('RGB')
    W, H = img.size
    print(f"WSI 打开: {wsi_path} 尺寸: {W}x{H}")

    half = patch_size // 2
    saved = 0
    for rank, i in enumerate(selected_indices):
        row = df.iloc[i]
        cx = float(row['x_centroid'])
        cy = float(row['y_centroid'])
        # 计算裁剪框，注意 PIL 的坐标是 (left, upper, right, lower)
        left = int(round(cx - half))
        upper = int(round(cy - half))
        right = int(round(cx + half))
        lower = int(round(cy + half))

        # 记录越界信息
        oob = False
        if left < 0 or upper < 0 or right > W or lower > H:
            oob = True

        # 创建黑色背景并粘贴有效区域（与 SimpleWSI 行为一致）
        canvas = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
        # 计算 crop 区间与放置偏移
        crop_left = max(0, left)
        crop_upper = max(0, upper)
        crop_right = min(W, right)
        crop_lower = min(H, lower)

        if crop_right > crop_left and crop_lower > crop_upper:
            crop = img.crop((crop_left, crop_upper, crop_right, crop_lower))
            offset_x = max(0, crop_left - left)
            offset_y = max(0, crop_upper - upper)
            canvas.paste(crop, (offset_x, offset_y))

        out_path = os.path.join(out_dir, f"debug_patch_{rank:03d}_idx{i:06d}.png")
        canvas.save(out_path)
        stats = np.array(canvas).astype(np.uint8)
        print(f"saved {out_path} | centroid=({cx:.2f},{cy:.2f}) oob={oob} mean={stats.mean():.2f} std={stats.std():.2f}")
        saved += 1

    print(f"已保存 {saved} 个 patch 到 {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wsi", required=True)
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--patch", type=int, default=36)
    p.add_argument("--mode", choices=['first','spaced','random','grid'], default='first',
                   help="采样模式：'first' 前N个, 'spaced' 均匀间隔, 'random' 随机, 'grid' 网格覆盖")
    p.add_argument("--grid_x", type=int, default=4, help="grid 模式下的 x 方向网格数")
    p.add_argument("--grid_y", type=int, default=4, help="grid 模式下的 y 方向网格数")
    p.add_argument("--rng_seed", type=int, default=42, help="随机种子")
    args = p.parse_args()

    crop_and_save(args.wsi, args.cells, args.out, n=args.n, patch_size=args.patch,
                  mode=args.mode, grid_size=(args.grid_x, args.grid_y), rng_seed=args.rng_seed)


if __name__ == "__main__":
    main()



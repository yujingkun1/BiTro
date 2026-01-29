#!/usr/bin/env python3
"""
Xenium 单细胞特征提取（复用 DINOv3 提取器）

此脚本基于 `extract_spatial_features_dinov3.py` 的 DINOv3 提取逻辑，
仅替换样本数据加载方法以适配 Xenium 导出的 `cells.csv` 与 `cell_boundaries.csv`。
"""
import os
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import shapely.wkb as wkb

# 从原脚本中复用 HESTCellFeatureExtractor
# 优先尝试在同一目录下本地导入（便于直接以脚本方式运行），
# 若失败再尝试作为包导入（便于以模块方式运行）。
try:
    from extract_spatial_features_dinov3 import HESTCellFeatureExtractor
except Exception:
    from Cell2Gene.utils.extract_spatial_features_dinov3 import HESTCellFeatureExtractor


class XeniumCellFeatureExtractor(HESTCellFeatureExtractor):
    """
    继承自 HESTCellFeatureExtractor，仅重写 load_sample_data 用于读取 Xenium 导出文件。
    假定 Xenium 的文件位于：
      - 图像: /data/yujk/hovernet2feature/xenium/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif
      - cells.csv: /data/yujk/hovernet2feature/xenium/outs/cells.csv
      - cell_boundaries.csv: /data/yujk/hovernet2feature/xenium/outs/cell_boundaries.csv
      - cell_feature_matrix.h5: /data/yujk/hovernet2feature/xenium/outs/cell_feature_matrix.h5
    返回格式与原脚本期望的相同：字典包含 'wsi_path' 和 'cellvit_df'（其中 'geometry' 列为 WKB bytes）。
    """

    def __init__(self, xenium_root_dir, output_dir, *args, **kwargs):
        # xenium_root_dir 例如: /data/yujk/hovernet2feature/xenium
        super().__init__(hest_data_dir=xenium_root_dir, output_dir=output_dir, *args, **kwargs)
        self.xenium_root_dir = xenium_root_dir

    def load_sample_data(self, sample_id=None):
        """
        为 Xenium 数据构造 sample_data。
        sample_id 可忽略；此方法将使用 self.xenium_root_dir 下固定路径。
        """
        xenium_outs = os.path.join(self.xenium_root_dir, "outs")
        cells_csv = os.path.join(xenium_outs, "cells.csv")
        boundaries_csv = os.path.join(xenium_outs, "cell_boundaries.csv")

        wsi_path = os.path.join(self.xenium_root_dir, "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif")
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"Xenium 图像未找到: {wsi_path}")

        if not os.path.exists(cells_csv):
            raise FileNotFoundError(f"cells.csv 未找到: {cells_csv}")
        if not os.path.exists(boundaries_csv):
            raise FileNotFoundError(f"cell_boundaries.csv 未找到: {boundaries_csv}")

        # 读取细胞中心数据信息（Xenium cells.csv）
        cells_df = pd.read_csv(cells_csv)
        # 期望列包括 'cell_id', 'x_centroid', 'y_centroid'（如果列名不同，请调整）
        # 将 cell_id 作为索引以便后续对齐
        if 'cell_id' not in cells_df.columns or 'x_centroid' not in cells_df.columns or 'y_centroid' not in cells_df.columns:
            raise RuntimeError("cells.csv 缺少预期列: 期望包含 'cell_id','x_centroid','y_centroid'")

        # 读取边界数据并为每个 cell_id 构建 Polygon
        bnd_df = pd.read_csv(boundaries_csv)
        if 'cell_id' not in bnd_df.columns or 'vertex_x' not in bnd_df.columns or 'vertex_y' not in bnd_df.columns:
            raise RuntimeError("cell_boundaries.csv 缺少预期列: 期望包含 'cell_id','vertex_x','vertex_y'")

        polygons = {}
        # 按 cell_id 分组构建多边形（按出现顺序）
        for cid, group in bnd_df.groupby('cell_id'):
            coords = list(zip(group['vertex_x'].astype(float).tolist(), group['vertex_y'].astype(float).tolist()))
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                polygons[int(cid)] = poly
            except Exception:
                # 若构建失败，使用一个小方形以避免空值
                cx = cells_df.loc[cells_df['cell_id'] == cid, 'x_centroid'].values
                cy = cells_df.loc[cells_df['cell_id'] == cid, 'y_centroid'].values
                if len(cx) and len(cy):
                    x0, y0 = float(cx[0]), float(cy[0])
                    poly = Polygon([(x0-1, y0-1), (x0+1, y0-1), (x0+1, y0+1), (x0-1, y0+1)])
                else:
                    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
                polygons[int(cid)] = poly

        # 构建与原脚本兼容的 DataFrame: 必须包含 'geometry' 列，且每行的 geometry 为 WKB bytes
        geom_list = []
        ids = []
        for _, row in cells_df.iterrows():
            cid = int(row['cell_id'])
            ids.append(cid)
            poly = polygons.get(cid, None)
            if poly is None:
                # 退化为以中心点构成的小方形
                cx = float(row['x_centroid'])
                cy = float(row['y_centroid'])
                poly = Polygon([(cx-1, cy-1), (cx+1, cy-1), (cx+1, cy+1), (cx-1, cy+1)])
            geom_list.append(poly.wkb)

        cellvit_df = pd.DataFrame({
            'cell_id': ids,
            'geometry': geom_list,
            # 保留中心坐标，若需要可在后续处理中使用
            'x_centroid': cells_df['x_centroid'].astype(float).values,
            'y_centroid': cells_df['y_centroid'].astype(float).values
        })

        sample_data = {
            'wsi_path': wsi_path,
            'cellvit_df': cellvit_df,
            'sample_id': 'XENIUM_SAMPLE'
        }

        print(f"✓ 已加载 Xenium 数据: image={wsi_path}, cells={len(cellvit_df)}")
        return sample_data


def main():
    xenium_root = "/data/yujk/hovernet2feature/xenium"
    output_dir = "/data/yujk/hovernet2feature/xenium_xenium_dinov3_features"
    os.makedirs(output_dir, exist_ok=True)

    # DINOv3模型路径可根据本地情况调整
    dinov3_model_path = "/data/yujk/hovernet2feature/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

    # 使用与测试脚本一致的小并发/批次配置，避免并发导致的 OpenSlide/IO 超时和大量默认黑色patch
    extractor = XeniumCellFeatureExtractor(
        xenium_root_dir=xenium_root,
        output_dir=output_dir,
        dinov3_model_path=dinov3_model_path,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu',
        assign_spot=False,  # Xenium 已为单细胞，可关闭 spot 分配
        num_workers=1,
        dino_batch_size=64,
        cell_batch_size=1000
    )

    # 仅处理 Xenium 单个样本（内部方法会忽略 sample_id）
    result = extractor.process_sample_with_independent_pca(sample_id='XENIUM_SAMPLE')
    print("处理完成，结果:", result)


if __name__ == "__main__":
    main()



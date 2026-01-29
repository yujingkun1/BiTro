#!/usr/bin/env python3
"""
基于Xenium（单细胞）构图脚本
本文件目标：针对单细胞定位数据，构建每个细胞与其最近的 K 个细胞的图（双向边）。
其它输入/输出格式与原 `spatial_graph_construction.py` 保持兼容，以便后续训练代码复用。
"""

import os
import pandas as pd
import numpy as np
import torch
import json
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import pickle
from tqdm import tqdm
import warnings
import scanpy as sc
import shapely.wkb as wkb
import h5py
import math

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1


class XeniumGraphBuilder:
    """针对Xenium单细胞数据的图构建器"""

    def __init__(self, data_dir, sample_ids, features_dir=None, k_neighbors=8):
        self.data_dir = data_dir
        self.sample_ids = sample_ids if isinstance(sample_ids, list) else [sample_ids]
        self.features_dir = features_dir
        self.k_neighbors = k_neighbors

        self.sample_data = {}
        self.processed_data = {}
        self.deep_features = {}

    def load_sample_data(self):
        """加载样本的AnnData与可选的细胞位置/特征文件（与原逻辑一致）"""
        print("=== 加载Xenium样本 ===")
        for sid in self.sample_ids:
            try:
                info = {}
                st_file = os.path.join(self.data_dir, "st", f"{sid}.h5ad")
                if os.path.exists(st_file):
                    adata = sc.read_h5ad(st_file)
                    info['adata'] = adata
                    print(f"加载样本 {sid}: {adata.n_obs} spots")
                else:
                    print(f"警告: 未找到 {st_file}, 跳过样本 {sid}")
                    continue

                # metadata
                metadata_file = os.path.join(self.data_dir, "metadata", f"{sid}.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        info['metadata'] = json.load(f)
                else:
                    info['metadata'] = {}

                # optional cellvit parquet kept for compatibility
                cellvit_file = os.path.join(self.data_dir, "cellvit_seg", f"{sid}_cellvit_seg.parquet")
                info['cellvit'] = pd.read_parquet(cellvit_file) if os.path.exists(cellvit_file) else None

                self.sample_data[sid] = info
            except Exception as e:
                print(f"错误: 加载样本 {sid} 失败: {e}")
                continue

        if self.features_dir:
            self.load_deep_features()

    def load_deep_features(self):
        """加载 npz 深度特征（可选）"""
        for sid in self.sample_ids:
            try:
                feat_file = os.path.join(self.features_dir, f"{sid}_combined_features.npz")
                if os.path.exists(feat_file):
                    data = np.load(feat_file, allow_pickle=True)
                    features = data['features'].astype(np.float32)
                    positions = data['positions'].astype(np.float32) if 'positions' in data else None
                    cluster_labels = data['cluster_labels'] if 'cluster_labels' in data else None
                    self.deep_features[sid] = {
                        'features': features,
                        'positions': positions,
                        'cluster_labels': cluster_labels
                    }
                    print(f"  已加载 {sid} 的深度特征: {features.shape}")
            except Exception as e:
                print(f"  警告: 加载深度特征失败 ({sid}): {e}")

    def extract_cells(self, sid):
        """严格从 Xenium 输出的 cells.csv 构建细胞表格（不使用回退逻辑）。"""
        xen = self.load_xenium_files(sid)
        cells_df = xen.get('cells_df')
        if cells_df is None or len(cells_df) == 0:
            raise RuntimeError(f"Xenium cells.csv 未找到或为空: 无法为样本 {sid} 构建细胞表")

        # 要求存在坐标列
        if 'x_centroid' not in cells_df.columns or 'y_centroid' not in cells_df.columns:
            raise RuntimeError("cells.csv 缺少 x_centroid/y_centroid 列，无法继续")

        df = pd.DataFrame({
            'cell_id': cells_df['cell_id'].astype(int).values,
            'x': cells_df['x_centroid'].astype(float).values,
            'y': cells_df['y_centroid'].astype(float).values,
            'area': cells_df['cell_area'].astype(float).values if 'cell_area' in cells_df.columns else np.full(len(cells_df), 100.0),
            'perimeter': cells_df['nucleus_area'].astype(float).values if 'nucleus_area' in cells_df.columns else np.full(len(cells_df), 35.4),
            'shape_feature': np.ones(len(cells_df), dtype=float)
        })
        return df

    def load_xenium_files(self, sid):
        """专门加载Xenium输出文件（cells.csv, cell_boundaries.csv, features h5, clusters）"""
        base = os.path.join("/data/yujk/hovernet2feature/xenium/outs")
        cells_path = os.path.join(base, "cells.csv")
        boundaries_path = os.path.join(base, "cell_boundaries.csv")
        features_path = os.path.join(base, "cell_feature_matrix.h5")
        clusters_path = os.path.join(base, "analysis", "clustering", "gene_expression_kmeans_10_clusters", "clusters.csv")

        xenium = {}
        try:
            if os.path.exists(cells_path):
                xenium['cells_df'] = pd.read_csv(cells_path)
            else:
                xenium['cells_df'] = None
        except Exception as e:
            print(f"  警告: 读取 cells.csv 失败: {e}")
            xenium['cells_df'] = None

        try:
            if os.path.exists(boundaries_path):
                xenium['boundaries_df'] = pd.read_csv(boundaries_path)
            else:
                xenium['boundaries_df'] = None
        except Exception as e:
            print(f"  警告: 读取 cell_boundaries.csv 失败: {e}")
            xenium['boundaries_df'] = None

        # 读取聚类标签（Barcode -> Cluster）
        try:
            if os.path.exists(clusters_path):
                clusters_df = pd.read_csv(clusters_path)
                # 将 Barcode 转为与 cells.csv 的 cell_id 对齐（两者似乎都是1-based）
                xenium['clusters_map'] = dict(zip(clusters_df['Barcode'].astype(int), clusters_df['Cluster'].astype(int)))
            else:
                xenium['clusters_map'] = {}
        except Exception as e:
            print(f"  警告: 读取 clusters.csv 失败: {e}")
            xenium['clusters_map'] = {}

        # 延迟读取 features h5，仅在需要时读取（可能很大）
        xenium['features_path'] = features_path if os.path.exists(features_path) else None

        return xenium

    def build_global_cell_graph(self, sid):
        """基于虚拟spot（大patch）在每个patch内构建spot内图。"""
        print(f"为样本 {sid} 构建按patch的spot内图（patch内每个细胞连接到最近 {self.k_neighbors} 个细胞）...")

        # 读取xenium专用输出
        xen = self.load_xenium_files(sid)
        cells_df = xen.get('cells_df')
        if cells_df is None or len(cells_df) == 0:
            # 回退到extract_cells（保持兼容）
            cells_df = self.extract_cells(sid)

        if cells_df is None or len(cells_df) == 0:
            print("  警告: 未检测到细胞，返回空字典")
            return {}

        # cells.csv 中可能列名不同，支持常见列
        if 'x_centroid' in cells_df.columns and 'y_centroid' in cells_df.columns:
            positions = cells_df[['x_centroid', 'y_centroid']].values
            barcode_ids = cells_df['cell_id'].astype(int).values if 'cell_id' in cells_df.columns else np.arange(len(cells_df)) + 1
        elif 'x' in cells_df.columns and 'y' in cells_df.columns:
            positions = cells_df[['x', 'y']].values
            barcode_ids = cells_df['cell_id'].astype(int).values if 'cell_id' in cells_df.columns else np.arange(len(cells_df)) + 1
        else:
            # 兼容之前的 DataFrame 格式
            positions = cells_df[['x', 'y']].values
            barcode_ids = cells_df['cell_id'].astype(int).values if 'cell_id' in cells_df.columns else np.arange(len(cells_df)) + 1

        # 读取聚类标签映射
        clusters_map = xen.get('clusters_map', {})

        # 尝试读取 features h5，如果存在且包含 'data' 或 'features' 数据集
        features_arr = None
        if xen.get('features_path'):
            try:
                with h5py.File(xen['features_path'], 'r') as hf:
                    # 尝试常见key
                    for key in ['features', 'data', 'X', 'matrix', 'expression']:
                        if key in hf:
                            features_arr = np.array(hf[key])
                            break
                    # 如果没有直接命中，尝试第一个可读的数据集
                    if features_arr is None:
                        for k in hf.keys():
                            try:
                                tmp = np.array(hf[k])
                                if tmp.ndim == 2:
                                    features_arr = tmp
                                    break
                            except Exception:
                                continue
            except Exception as e:
                print(f"  警告: 读取 features h5 失败: {e}")

        # 如果 features_arr 存在且与cells数量一致，使用深度特征
        use_deep = False
        if features_arr is not None and features_arr.shape[0] == positions.shape[0]:
            use_deep = True
            cell_features = features_arr.astype(np.float32)
        else:
            # 回退到几何特征并扩展至128维
            cell_features = np.array([
                [
                    row.get('cell_area', row.get('area', 100.0)),
                    row.get('nucleus_area', row.get('perimeter', 35.4)),
                    1.0,
                    float(row.get('x_centroid', row.get('x', 0.0))),
                    float(row.get('y_centroid', row.get('y', 0.0)))
                ]
                for _, row in cells_df.iterrows()
            ], dtype=np.float32)
            target_dim = 128
            if cell_features.shape[1] < target_dim:
                pad = np.zeros((cell_features.shape[0], target_dim - cell_features.shape[1]), dtype=np.float32)
                cell_features = np.concatenate([cell_features, pad], axis=1)
            elif cell_features.shape[1] > target_dim:
                cell_features = cell_features[:, :target_dim]

        # 计算patch划分（虚拟spot）
        xs = positions[:, 0]
        ys = positions[:, 1]
        # patch_size 默认使用较大值，可调整
        patch_size = 2000.0
        min_x, min_y = xs.min(), ys.min()
        # normalize to start at 0
        gx = ((xs - min_x) // patch_size).astype(int)
        gy = ((ys - min_y) // patch_size).astype(int)
        patch_tuples = list(zip(gy.tolist(), gx.tolist()))

        # 给每个唯一patch赋一个连续的索引（便于保存与兼容）
        unique_patches = {}
        patch_indices = []
        for t in patch_tuples:
            if t not in unique_patches:
                unique_patches[t] = len(unique_patches)
            patch_indices.append(unique_patches[t])
        patch_indices = np.array(patch_indices, dtype=int)

        intra_spot_graphs = {}

        # 针对每个patch（虚拟spot）构建局部图
        for patch_id in tqdm(sorted(unique_patches.values()), desc="构建虚拟spot内图"):
            idxs = np.where(patch_indices == patch_id)[0]
            if len(idxs) == 0:
                continue

            pos_patch = positions[idxs]
            feats_patch = cell_features[idxs]

            n_patch = pos_patch.shape[0]
            k = min(self.k_neighbors, max(0, n_patch - 1))

            if k <= 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                nbrs = NearestNeighbors(n_neighbors=k + 1).fit(pos_patch)
                _, indices = nbrs.kneighbors(pos_patch)
                edges = []
                for i_local, neigh in enumerate(indices):
                    for nb in neigh[1:]:
                        edges.extend([[i_local, nb], [nb, i_local]])
                edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)

            x_tensor = torch.tensor(feats_patch, dtype=torch.float32)
            pos_tensor = torch.tensor(pos_patch, dtype=torch.float32)
            graph = Data(x=x_tensor, edge_index=edge_index, pos=pos_tensor)

            # 附加聚类标签（使用 clusters_map，通过 barcode_ids[idxs] 映射）
            if clusters_map:
                try:
                    cls_list = []
                    for absolute_idx in idxs:
                        barcode = int(barcode_ids[absolute_idx])
                        cls_list.append(int(clusters_map.get(barcode, -1)))
                    graph.cluster_labels = torch.as_tensor(cls_list, dtype=torch.long)
                except Exception:
                    pass

            intra_spot_graphs[int(patch_id)] = graph

        print(f"  构建了 {len(intra_spot_graphs)} 个虚拟spot内图（patch_size={patch_size}）")
        return intra_spot_graphs

    def process_all_samples(self):
        for sid in self.sample_ids:
            print(f"处理样本 {sid} ...")
            cells_df = self.extract_cells(sid)
            # Xenium每个细胞即spot，标注spot_assignment为自身索引，保持与原接口兼容
            if len(cells_df) > 0:
                cells_df['spot_assignment'] = np.arange(len(cells_df))
                cells_df['distance_to_spot'] = 0.0
            self.processed_data[sid] = {'cells': cells_df, 'adata': self.sample_data[sid]['adata']}

    def save_graphs(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_graphs = {}
        all_metadata = {}

        for sid in self.processed_data.keys():
            intra = self.build_global_cell_graph(sid)
            inter = None  # Xenium场景下可选择不构建spot间图或使用原spot位置构建；这里保留None以示区分
            all_graphs[sid] = {'intra_spot_graphs': intra, 'inter_spot_graph': inter}

            adata = self.sample_data[sid]['adata']
            cells_df = self.processed_data[sid]['cells']
            meta = self.sample_data[sid].get('metadata', {})

            # 统计信息：细胞数、每个虚拟spot的节点/边统计等
            num_cells = len(cells_df) if cells_df is not None else 0
            num_assigned = num_cells  # 在Xenium我们将所有细胞视为已分配
            intra_count = len(intra) if isinstance(intra, dict) else 0

            # 汇总每个patch的节点/边
            patch_node_counts = []
            patch_edge_counts = []
            total_edges = 0
            for pid, g in (intra.items() if isinstance(intra, dict) else []):
                try:
                    n_nodes = int(g.x.shape[0]) if hasattr(g, 'x') else 0
                    n_edges = int(g.edge_index.shape[1]) if hasattr(g, 'edge_index') else 0
                except Exception:
                    n_nodes, n_edges = 0, 0
                patch_node_counts.append(n_nodes)
                patch_edge_counts.append(n_edges)
                total_edges += n_edges

            patch_nodes_min = int(min(patch_node_counts)) if patch_node_counts else 0
            patch_nodes_max = int(max(patch_node_counts)) if patch_node_counts else 0
            patch_nodes_mean = float(np.mean(patch_node_counts)) if patch_node_counts else 0.0

            # 记录到metadata
            all_metadata[sid] = {
                'num_spots': int(getattr(adata, 'n_obs', 0)),
                'num_genes': int(getattr(adata, 'n_vars', 0)),
                'num_cells': int(num_cells),
                'num_assigned_cells': int(num_assigned),
                'k_neighbors': int(self.k_neighbors),
                'intra_graph_count': int(intra_count),
                'intra_graph_total_edges': int(total_edges),
                'intra_patch_nodes_min': int(patch_nodes_min),
                'intra_patch_nodes_max': int(patch_nodes_max),
                'intra_patch_nodes_mean': float(patch_nodes_mean),
                'inter_graph_edges': 0 if inter is None else int(getattr(inter.edge_index, 'shape', [None, 0])[1]),
                'tissue': meta.get('tissue', 'unknown'),
                'pixel_size_um': meta.get('pixel_size_um_estimated', 0.5)
            }

            # 控制台打印（类似原 spatial 脚本的输出）
            print(f"\n样本 {sid}:")
            print(f"  - 总细胞数: {num_cells}")
            print(f"  - 已分配细胞数: {num_assigned}")
            print(f"  - 虚拟spot(patch)数量: {intra_count}")
            print(f"  - 每patch节点数（min/mean/max): {patch_nodes_min}/{patch_nodes_mean:.1f}/{patch_nodes_max}")
            print(f"  - 虚拟spot内总边数: {total_edges}")

        # 保存文件
        intra_graphs_path = os.path.join(output_dir, "xenium_intra_spot_graphs.pkl")
        inter_graphs_path = os.path.join(output_dir, "xenium_inter_spot_graphs.pkl")
        metadata_path = os.path.join(output_dir, "xenium_graph_metadata.json")
        processed_data_path = os.path.join(output_dir, "xenium_processed_data.pkl")

        with open(intra_graphs_path, 'wb') as f:
            pickle.dump({sid: data['intra_spot_graphs'] for sid, data in all_graphs.items()}, f)
        with open(inter_graphs_path, 'wb') as f:
            pickle.dump({sid: data['inter_spot_graph'] for sid, data in all_graphs.items()}, f)
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        with open(processed_data_path, 'wb') as f:
            pickle.dump(self.processed_data, f)

        print(f"\n图数据保存至: {output_dir}")
        print(f"- Spot内图: {intra_graphs_path}")
        print(f"- Spot间图: {inter_graphs_path}")
        print(f"- 元数据: {metadata_path}")
        print(f"- 处理数据: {processed_data_path}")
        return all_metadata


def main():
    # 使用 Xenium 数据目录和特征路径
    data_dir = "/data/yujk/hovernet2feature/xenium/outs"
    features_dir = "/data/yujk/hovernet2feature/xenium/outs"
    output_dir = "/data/yujk/hovernet2feature/xenium_graphs"

    # 检查 xenium 输出的 cells.csv 以确定样本是否存在（将全部视为单个样本处理）
    cells_path = os.path.join(data_dir, "cells.csv")
    if not os.path.exists(cells_path):
        print(f"未找到 Xenium cells.csv: {cells_path}")
        return

    sample_ids = ["xenium_sample"]

    builder = XeniumGraphBuilder(data_dir=data_dir, sample_ids=sample_ids, features_dir=features_dir, k_neighbors=8)

    # 为保持与原有接口兼容，构建一个最小 AnnData 并赋值到 builder.sample_data
    try:
        cells_df = pd.read_csv(cells_path)
        if 'x_centroid' in cells_df.columns and 'y_centroid' in cells_df.columns:
            positions = cells_df[['x_centroid', 'y_centroid']].values.astype(np.float32)
        elif 'x' in cells_df.columns and 'y' in cells_df.columns:
            positions = cells_df[['x', 'y']].values.astype(np.float32)
        else:
            positions = np.zeros((len(cells_df), 2), dtype=np.float32)

        adata = sc.AnnData(np.zeros((positions.shape[0], 1)))
        adata.obsm['spatial'] = positions
        # 简单metadata占位
        metadata = {}
        builder.sample_data[sample_ids[0]] = {'adata': adata, 'metadata': metadata}
    except Exception as e:
        print(f"构建 AnnData 失败: {e}")
        return

    # 可选加载深度特征（如果存在于 features_dir）
    builder.load_deep_features()

    # 直接使用我们的 Xenium 专用处理流程
    builder.process_all_samples()
    builder.save_graphs(output_dir)


if __name__ == "__main__":
    main()



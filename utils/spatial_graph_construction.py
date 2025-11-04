#!/usr/bin/env python3
"""
HEST数据集图构建脚本 - 直接读取HEST数据文件（无需HEST库）
基于HEST数据格式构建两层次图结构：
1. Spot内图：基于CellViT细胞分割构建细胞级图
2. Spot间图：基于空转数据的spot邻近关系建图

直接读取HEST数据文件格式：
- st/*.h5ad: AnnData格式的空转数据
- cellvit_seg/*.parquet: CellViT细胞分割数据
- metadata/*.json: 样本元数据
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
warnings.filterwarnings("ignore")

# 设置scanpy不显示过多信息
sc.settings.verbosity = 1


class HESTDirectReader:
    """直接读取HEST数据文件的图构建器"""

    def __init__(self,
                 hest_data_dir,
                 sample_ids,
                 features_dir=None,             # 深度特征文件目录
                 inter_spot_k_neighbors=6):          # spot间k近邻数量

        self.hest_data_dir = hest_data_dir
        self.sample_ids = sample_ids if isinstance(
            sample_ids, list) else [sample_ids]
        self.features_dir = features_dir  # 深度特征文件目录
        self.inter_spot_k_neighbors = inter_spot_k_neighbors

        self.sample_data = {}
        self.processed_data = {}
        self.deep_features = {}  # 深度特征缓存

    def load_sample_data(self):
        """直接加载HEST数据文件"""
        print("=== 直接加载HEST数据文件 ===")

        for sample_id in self.sample_ids:
            try:
                print(f"加载样本: {sample_id}")

                sample_info = {}

                # 1. 加载AnnData文件
                st_file = os.path.join(
                    self.hest_data_dir, "st", f"{sample_id}.h5ad")
                if os.path.exists(st_file):
                    adata = sc.read_h5ad(st_file)
                    sample_info['adata'] = adata
                    print(
                        f"  - AnnData: {adata.n_obs} spots × {adata.n_vars} genes")
                else:
                    print(f"  - 警告: 未找到AnnData文件: {st_file}")
                    continue

                # 2. 加载元数据
                metadata_file = os.path.join(
                    self.hest_data_dir, "metadata", f"{sample_id}.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    sample_info['metadata'] = metadata
                    print(f"  - 元数据: {metadata.get('tissue', 'unknown')} 组织")
                else:
                    sample_info['metadata'] = {}

                # 3. 加载细胞分割数据
                cellvit_file = os.path.join(
                    self.hest_data_dir, "cellvit_seg", f"{sample_id}_cellvit_seg.parquet")
                if os.path.exists(cellvit_file):
                    cellvit_df = pd.read_parquet(cellvit_file)
                    sample_info['cellvit'] = cellvit_df
                    print(f"  - 细胞分割: {len(cellvit_df)} 个细胞")
                else:
                    print(f"  - 警告: 未找到细胞分割文件: {cellvit_file}")
                    sample_info['cellvit'] = None

                # 4. 检查图像patches
                patches_file = os.path.join(
                    self.hest_data_dir, "patches", f"{sample_id}.h5")
                if os.path.exists(patches_file):
                    sample_info['patches_file'] = patches_file
                    print(f"  - 图像patches: 可用")
                else:
                    sample_info['patches_file'] = None
                    print(f"  - 警告: 未找到patches文件")

                self.sample_data[sample_id] = sample_info

            except Exception as e:
                print(f"  错误: 加载样本 {sample_id} 失败: {e}")
                continue

        print(f"成功加载 {len(self.sample_data)} 个样本")

        # 如果指定了深度特征目录，加载深度特征
        if self.features_dir:
            self.load_deep_features()

    def load_deep_features(self):
        """加载深度特征文件"""
        print("=== 加载深度特征文件 ===")

        for sample_id in self.sample_ids:
            try:
                # 查找样本的特征文件
                feature_file = os.path.join(
                    self.features_dir, f"{sample_id}_combined_features.npz")

                if os.path.exists(feature_file):
                    print(f"加载样本 {sample_id} 的深度特征...")

                    # 加载特征数据 - 添加allow_pickle=True来处理metadata
                    data = np.load(feature_file, allow_pickle=True)
                    features = data['features'].astype(
                        np.float32)  # [N, D] 的深度特征
                    positions = data['positions'].astype(
                        np.float32) if 'positions' in data else None
                    cell_index = data['cell_index'] if 'cell_index' in data else None
                    spot_ptr = data['spot_ptr'] if 'spot_ptr' in data else None
                    spot_index = data['spot_index'] if 'spot_index' in data else None

                    # 安全处理metadata
                    try:
                        if 'metadata' in data:
                            metadata_array = data['metadata']
                            # 处理0维numpy array的metadata
                            if metadata_array.ndim == 0:
                                metadata = metadata_array.item() if metadata_array.dtype == object else {}
                            else:
                                metadata = metadata_array if isinstance(
                                    metadata_array, dict) else {}
                        else:
                            metadata = {}
                    except Exception as meta_e:
                        print(f"    警告: metadata解析失败: {meta_e}")
                        metadata = {}

                    # 对齐检查与裁剪（如有需要）
                    n_feat = features.shape[0]
                    if positions is not None:
                        n_pos = positions.shape[0]
                        if n_pos != n_feat:
                            n_min = min(n_feat, n_pos)
                            print(
                                f"    警告: features与positions长度不一致: feats={n_feat}, pos={n_pos}，将裁剪为 {n_min}")
                            features = features[:n_min]
                            positions = positions[:n_min]
                            n_feat = n_min

                    # 读取可选的聚类标签（若已由聚类脚本写回）
                    cluster_labels = None
                    try:
                        if 'cluster_labels' in data:
                            cluster_labels = data['cluster_labels']
                            # 类型与长度检查
                            if getattr(cluster_labels, 'ndim', 1) != 1 or cluster_labels.shape[0] != features.shape[0]:
                                print("    警告: cluster_labels 维度或长度与 features 不一致，将忽略")
                                cluster_labels = None
                    except Exception as _:
                        cluster_labels = None

                    self.deep_features[sample_id] = {
                        'features': features,
                        'positions': positions,
                        'cell_index': cell_index,
                        'spot_ptr': spot_ptr,
                        'spot_index': spot_index,
                        'metadata': metadata,
                        'cluster_labels': cluster_labels
                    }

                    print(f"  - 特征形状: {features.shape}")
                    print(f"  - 特征维度: {features.shape[1]}")
                    print(f"  - 细胞数量: {features.shape[0]}")
                    if positions is not None:
                        print(f"  - 坐标可用: {positions.shape}")

                else:
                    print(f"  警告: 未找到样本 {sample_id} 的特征文件: {feature_file}")

            except Exception as e:
                print(f"  错误: 加载样本 {sample_id} 特征失败: {e}")

        print(f"成功加载 {len(self.deep_features)} 个样本的深度特征")

    def extract_cell_features_from_cellvit(self, sample_id):
        """从CellViT分割数据提取细胞特征"""
        print(f"提取样本 {sample_id} 的细胞特征...")

        sample_info = self.sample_data[sample_id]
        cellvit_df = sample_info.get('cellvit')

        # 如果深度特征中存在positions，则优先使用以确保与特征严格对齐
        deep = self.deep_features.get(sample_id) if hasattr(
            self, 'deep_features') else None
        if deep is not None and deep.get('positions') is not None:
            positions = deep['positions']
            n = positions.shape[0]
            print(f"  使用NPZ中的positions作为细胞坐标，对齐长度: {n}")
            # 若没有cellvit或长度不匹配，仅使用positions生成基础表
            cells_data = []
            for idx in range(n):
                x, y = float(positions[idx, 0]), float(positions[idx, 1])
                # 面积与周长无法从positions推断，使用默认值占位（不会影响深度特征分支）
                cells_data.append({
                    'cell_id': idx,
                    'x': x,
                    'y': y,
                    'area': 100.0,
                    'perimeter': 35.4,
                    'shape_feature': 1.0
                })
            return pd.DataFrame(cells_data)

        if cellvit_df is None or len(cellvit_df) == 0:
            print(f"  警告: 样本 {sample_id} 无细胞分割数据且无positions，使用spot中心点")
            return self.create_spot_based_features(sample_info['adata'])

        # 提取细胞特征
        cells_data = []

        print(f"  正在解析 {len(cellvit_df)} 个细胞的几何数据...")

        for idx in tqdm(range(len(cellvit_df)), desc="  解析细胞坐标"):
            try:
                row = cellvit_df.iloc[idx]

                # 解析WKB格式的geometry
                if 'geometry' in cellvit_df.columns:
                    geom_bytes = row['geometry']
                    geom = wkb.loads(geom_bytes)

                    # 获取中心点
                    centroid = geom.centroid
                    cell_x, cell_y = centroid.x, centroid.y

                    # 计算面积和周长
                    cell_area = geom.area
                    cell_perimeter = geom.length

                else:
                    # 如果没有geometry列，使用默认值
                    cell_x, cell_y = float(idx % 100), float(idx // 100)
                    cell_area = 100.0
                    cell_perimeter = 35.4

                # 计算形状特征
                cell_shape_feature = cell_perimeter**2 / \
                    (4 * np.pi * cell_area) if cell_area > 0 else 1.0

                cells_data.append({
                    'cell_id': idx,
                    'x': cell_x,
                    'y': cell_y,
                    'area': cell_area,
                    'perimeter': cell_perimeter,
                    'shape_feature': cell_shape_feature
                })

            except Exception as e:
                if idx < 10:  # 只打印前10个错误
                    print(f"    警告: 处理细胞 {idx} 时出错: {e}")
                # 使用默认值
                cells_data.append({
                    'cell_id': idx,
                    'x': float(idx % 100),
                    'y': float(idx // 100),
                    'area': 100.0,
                    'perimeter': 35.4,
                    'shape_feature': 1.0
                })

        if not cells_data:
            print(f"  警告: 未能提取到细胞数据，使用spot中心点")
            return self.create_spot_based_features(sample_info['adata'])

        cells_df = pd.DataFrame(cells_data)

        # 打印坐标范围信息
        print(f"  提取了 {len(cells_df)} 个细胞的特征")
        print(
            f"  细胞坐标范围: X[{cells_df['x'].min():.1f}, {cells_df['x'].max():.1f}], Y[{cells_df['y'].min():.1f}, {cells_df['y'].max():.1f}]")

        return cells_df

    def create_spot_based_features(self, adata):
        """当没有细胞分割时，基于spot创建伪细胞特征"""
        spots_coords = adata.obsm['spatial']

        cells_data = []
        for spot_idx, (x, y) in enumerate(spots_coords):
            cells_data.append({
                'cell_id': spot_idx,
                'x': float(x),
                'y': float(y),
                'area': 100.0,  # 假设固定面积
                'perimeter': 35.4,  # 对应圆形的周长
                'shape_feature': 1.0  # 圆形的形状特征
            })

        return pd.DataFrame(cells_data)

    def assign_cells_to_spots(self, sample_id, cells_df):
        """将细胞分配到spots"""
        print(f"为样本 {sample_id} 分配细胞到spots...")

        adata = self.sample_data[sample_id]['adata']
        metadata = self.sample_data[sample_id]['metadata']
        st_technology = metadata.get('st_technology', 'Unknown')

        # Xenium技术：每个细胞就是一个spot
        if st_technology == 'Xenium':
            print(f"  检测到Xenium技术，直接将细胞映射为spots...")
            cells_df = cells_df.copy()

            # 获取spot坐标
            spots_coords = adata.obsm['spatial']
            print(f"  细胞数量: {len(cells_df)}, Spot数量: {len(spots_coords)}")

            # 对于Xenium，直接一一对应（取较小的数量）
            min_count = min(len(cells_df), len(spots_coords))
            cells_df['spot_assignment'] = -1
            cells_df['distance_to_spot'] = 0.0

            # 前min_count个细胞直接分配到对应的spot
            cells_df.iloc[:min_count, cells_df.columns.get_loc(
                'spot_assignment')] = range(min_count)

            assigned_count = min_count
            print(f"  成功分配: {assigned_count}/{len(cells_df)} 细胞 (Xenium直接映射)")

            return cells_df

        # 其他技术：使用距离匹配
        # 获取spot坐标
        spots_coords = adata.obsm['spatial']

        # 打印调试信息
        print(
            f"  Spots坐标范围: X[{spots_coords[:, 0].min():.1f}, {spots_coords[:, 0].max():.1f}], Y[{spots_coords[:, 1].min():.1f}, {spots_coords[:, 1].max():.1f}]")
        print(
            f"  细胞坐标范围: X[{cells_df['x'].min():.1f}, {cells_df['x'].max():.1f}], Y[{cells_df['y'].min():.1f}, {cells_df['y'].max():.1f}]")

        # 为每个细胞分配最近的spot
        cells_df = cells_df.copy()
        cells_df['spot_assignment'] = -1
        cells_df['distance_to_spot'] = float('inf')

        # 处理所有细胞（移除采样限制）
        print(f"  正在为 {len(cells_df)} 个细胞分配到 {len(spots_coords)} 个spots...")

        assigned_count = 0
        distances_list = []

        # 分批处理以避免内存问题
        batch_size = 200000
        num_batches = (len(cells_df) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(cells_df))
            batch_cells = cells_df.iloc[start_idx:end_idx]

            print(
                f"  处理批次 {batch_idx+1}/{num_batches} ({len(batch_cells)} 个细胞)")

            for cell_idx, cell_row in tqdm(batch_cells.iterrows(), total=len(batch_cells), desc=f"  批次{batch_idx+1}"):
                cell_pos = np.array([cell_row['x'], cell_row['y']])

                # 计算到所有spots的距离
                distances = np.linalg.norm(spots_coords - cell_pos, axis=1)
                nearest_spot_idx = np.argmin(distances)
                nearest_distance = distances[nearest_spot_idx]

                distances_list.append(nearest_distance)

                # 使用像素大小转换距离
                pixel_size_um = metadata.get(
                    'pixel_size_um_estimated', 0.5)  # 默认0.5微米/像素
                distance_um = nearest_distance * pixel_size_um

                # 从元数据获取spot半径（像素 -> 微米）
                spot_diameter_px = metadata.get('spot_diameter', None)

                if pd.isna(spot_diameter_px) or spot_diameter_px is None:
                    # 其他未知技术使用默认值
                    spot_radius_um = 25.0  # 默认25微米半径
                else:
                    # Visium等基于spot的技术
                    base_radius_um = (spot_diameter_px / 2.0) * \
                        pixel_size_um  # 基础半径
                    spot_radius_um = base_radius_um * 1.5  # 使用1.5倍半径增加匹配范围

                if distance_um <= spot_radius_um:
                    cells_df.at[cell_idx, 'spot_assignment'] = nearest_spot_idx
                    cells_df.at[cell_idx, 'distance_to_spot'] = distance_um
                    assigned_count += 1

        # 统计分配结果
        assigned_cells = cells_df[cells_df['spot_assignment'] >= 0]

        print(f"  成功分配: {len(assigned_cells)}/{len(cells_df)} 细胞")
        pixel_size_um = metadata.get('pixel_size_um_estimated', 0.5)
        print(f"  像素大小: {pixel_size_um} μm/pixel")

        if len(distances_list) > 0:
            distances_array = np.array(distances_list)
            print(
                f"  距离统计: 最小={distances_array.min():.1f}px, 最大={distances_array.max():.1f}px, 平均={distances_array.mean():.1f}px")
            print(
                f"  距离统计(μm): 最小={distances_array.min()*pixel_size_um:.1f}μm, 最大={distances_array.max()*pixel_size_um:.1f}μm, 平均={distances_array.mean()*pixel_size_um:.1f}μm")

        if len(assigned_cells) > 0:
            spot_counts = assigned_cells['spot_assignment'].value_counts()
            print(
                f"  每个spot的细胞数: 平均={spot_counts.mean():.1f}, 范围=[{spot_counts.min()}-{spot_counts.max()}]")
        else:
            print(f"  警告: 没有细胞被分配到spots，可能需要调整距离阈值")

        return cells_df

    def build_intra_spot_graphs(self, sample_id, intra_spot_k_neighbors=8):
        """构建spot内的图结构（基于细胞）- spot内所有细胞可连接，使用k近邻"""
        print(f"构建样本 {sample_id} 的spot内图...")

        cells_df = self.processed_data[sample_id]['cells']
        # 如果有spot分组信息（来自深度特征spot_ptr或spot_index），尝试优先使用
        deep = self.deep_features.get(sample_id) if hasattr(
            self, 'deep_features') else None
        group_by_spot = None
        if deep is not None:
            spot_ptr = deep.get('spot_ptr')
            spot_index = deep.get('spot_index')
            if spot_ptr is not None and isinstance(spot_ptr, (np.ndarray, list)) and len(spot_ptr) >= 2:
                # 通过ptr映射构造字典：spot_idx -> 索引范围
                group_by_spot = {int(si): (int(spot_ptr[si]), int(
                    spot_ptr[si+1])) for si in range(len(spot_ptr)-1)}
            elif spot_index is not None and isinstance(spot_index, np.ndarray):
                # 通过index分组
                group_by_spot = {}
                for i, si in enumerate(spot_index.tolist()):
                    si = int(si)
                    group_by_spot.setdefault(si, []).append(i)
        assigned_cells = cells_df[cells_df['spot_assignment'] >= 0]

        intra_spot_graphs = {}

        # 按spot分组处理
        for spot_idx in tqdm(assigned_cells['spot_assignment'].unique(), desc="构建spot内图"):
            spot_cells = assigned_cells[assigned_cells['spot_assignment'] == spot_idx].copy(
            )

            if len(spot_cells) < 2:
                # 单个或无细胞的spot
                if len(spot_cells) == 1:
                    # 创建单节点图
                    cell_row = spot_cells.iloc[0]
                    cell_features = self.extract_cell_feature_vector(
                        cell_row, sample_id, cell_row.name)
                    x = torch.tensor([cell_features], dtype=torch.float32)
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    pos = torch.tensor(
                        [[cell_row['x'], cell_row['y']]], dtype=torch.float32)
                    graph = Data(x=x, edge_index=edge_index, pos=pos)
                    # 如果存在聚类标签，附加到图
                    if deep is not None and deep.get('cluster_labels') is not None:
                        cls = deep['cluster_labels']
                        if cell_row.name < len(cls):
                            graph.cluster_labels = torch.tensor([
                                int(cls[cell_row.name])
                            ], dtype=torch.long)
                    intra_spot_graphs[int(spot_idx)] = graph
                continue

            # 提取位置和特征
            # 默认使用cells_df计算位置与特征
            positions = spot_cells[['x', 'y']].values
            cell_features = np.array([
                self.extract_cell_feature_vector(row, sample_id, row.name)
                for _, row in spot_cells.iterrows()
            ])
            cluster_slice = None

            # 若深度特征提供了严格按spot划分的索引，且长度匹配，优先使用以避免对齐偏差
            if deep is not None and deep.get('features') is not None and deep.get('positions') is not None:
                feats_np = deep['features']
                pos_np = deep['positions']
                cls_np = deep.get('cluster_labels')
                if isinstance(group_by_spot, dict) and spot_idx in group_by_spot:
                    idxs = group_by_spot[spot_idx]
                    if isinstance(idxs, tuple):
                        s, e = idxs
                        if 0 <= s < e <= feats_np.shape[0] and e <= pos_np.shape[0]:
                            positions = pos_np[s:e]
                            cell_features = feats_np[s:e]
                            if cls_np is not None and e <= cls_np.shape[0]:
                                cluster_slice = cls_np[s:e]
                    elif isinstance(idxs, list) and len(idxs) > 0:
                        idxs = [i for i in idxs if 0 <= i <
                                feats_np.shape[0] and i < pos_np.shape[0]]
                        if idxs:
                            positions = pos_np[idxs]
                            cell_features = feats_np[idxs]
                            if cls_np is not None:
                                valid_cls = [i for i in idxs if i < cls_np.shape[0]]
                                if len(valid_cls) == len(idxs):
                                    cluster_slice = cls_np[idxs]

            # spot内所有细胞都可以连接，使用k近邻控制连接密度
            k = min(intra_spot_k_neighbors, len(spot_cells) - 1)

            if k > 0:
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(positions)
                _, indices = nbrs.kneighbors(positions)

                edges = []
                for i, neighbors in enumerate(indices):
                    for neighbor in neighbors[1:]:  # 跳过自己
                        edges.extend([[i, neighbor], [neighbor, i]])

                if edges:
                    edge_index = torch.tensor(
                        np.array(edges).T, dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # 创建图数据
            x = torch.tensor(cell_features, dtype=torch.float32)
            pos = torch.tensor(positions, dtype=torch.float32)

            graph = Data(x=x, edge_index=edge_index, pos=pos)
            # 附加聚类标签（若可用且长度匹配）
            if cluster_slice is not None:
                try:
                    cls_t = torch.as_tensor(cluster_slice, dtype=torch.long)
                    if cls_t.dim() == 1 and cls_t.numel() == x.shape[0]:
                        graph.cluster_labels = cls_t
                except Exception:
                    pass
            intra_spot_graphs[int(spot_idx)] = graph

        print(
            f"  构建了 {len(intra_spot_graphs)} 个spot内图，每个spot内使用 {intra_spot_k_neighbors} 近邻连接")
        return intra_spot_graphs

    def extract_cell_feature_vector(self, cell_row, sample_id=None, cell_idx=None):
        """提取细胞特征向量（支持深度特征）"""

        # 优先使用深度特征
        if sample_id and sample_id in self.deep_features and cell_idx is not None:
            deep_features = self.deep_features[sample_id]['features']
            if cell_idx < len(deep_features):
                return deep_features[cell_idx].astype(np.float32)  # 返回128维深度特征

        # 回退到简单几何特征
        features = [
            cell_row['area'],
            cell_row['perimeter'],
            cell_row['shape_feature'],
            cell_row['x'],
            cell_row['y']
        ]

        # 归一化处理
        features = np.array(features, dtype=np.float32)

        # 扩展到目标维度
        target_dim = 128  # 改为128以匹配深度特征维度
        if len(features) < target_dim:
            # 零填充
            features = np.pad(
                features, (0, target_dim - len(features)), mode='constant')
        elif len(features) > target_dim:
            # 截断
            features = features[:target_dim]

        return features

    def build_inter_spot_graph(self, sample_id):
        """构建spot间的图结构"""
        print(f"构建样本 {sample_id} 的spot间图...")

        adata = self.sample_data[sample_id]['adata']

        # 获取spot位置
        spot_positions = adata.obsm['spatial']

        # 使用k近邻构建spot间连接
        nbrs = NearestNeighbors(
            n_neighbors=self.inter_spot_k_neighbors+1).fit(spot_positions)
        _, indices = nbrs.kneighbors(spot_positions)

        # 构建边列表
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # 跳过自己
                edges.extend([[i, neighbor], [neighbor, i]])

        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)

        # spot特征：使用位置特征
        spot_features = torch.tensor(spot_positions, dtype=torch.float32)
        pos = torch.tensor(spot_positions, dtype=torch.float32)

        inter_spot_graph = Data(
            x=spot_features, edge_index=edge_index, pos=pos)

        print(
            f"  构建了spot间图: {len(spot_positions)} spots, {edge_index.shape[1]} 条边")
        return inter_spot_graph

    def process_all_samples(self):
        """处理所有样本"""
        print("=== 处理所有样本 ===")

        for sample_id in self.sample_data.keys():
            print(f"\n处理样本: {sample_id}")

            # 提取细胞特征
            cells_df = self.extract_cell_features_from_cellvit(sample_id)

            # 分配细胞到spots
            cells_with_spots = self.assign_cells_to_spots(sample_id, cells_df)

            # 存储处理结果
            self.processed_data[sample_id] = {
                'cells': cells_with_spots,
                'adata': self.sample_data[sample_id]['adata']
            }

            print(f"  样本 {sample_id} 处理完成")

    def save_graphs(self, output_dir):
        """保存构建的图"""
        print("=== 保存图结构 ===")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_graphs = {}
        all_metadata = {}

        for sample_id in self.processed_data.keys():
            print(f"\n保存样本 {sample_id} 的图...")

            # 构建图
            intra_spot_graphs = self.build_intra_spot_graphs(
                sample_id, intra_spot_k_neighbors=8)
            inter_spot_graph = self.build_inter_spot_graph(sample_id)

            # 保存到总的字典中
            all_graphs[sample_id] = {
                'intra_spot_graphs': intra_spot_graphs,
                'inter_spot_graph': inter_spot_graph
            }

            # 元数据
            adata = self.sample_data[sample_id]['adata']
            cells_df = self.processed_data[sample_id]['cells']
            metadata = self.sample_data[sample_id]['metadata']

            all_metadata[sample_id] = {
                'num_spots': adata.n_obs,
                'num_genes': adata.n_vars,
                'num_cells': len(cells_df),
                'num_assigned_cells': len(cells_df[cells_df['spot_assignment'] >= 0]),
                'inter_spot_k_neighbors': self.inter_spot_k_neighbors,
                'intra_graph_count': len(intra_spot_graphs),
                'inter_graph_edges': inter_spot_graph.edge_index.shape[1],
                'tissue': metadata.get('tissue', 'unknown'),
                'pixel_size_um': metadata.get('pixel_size_um_estimated', 0.5)
            }

        # 保存文件
        intra_graphs_path = os.path.join(
            output_dir, "hest_intra_spot_graphs.pkl")
        inter_graphs_path = os.path.join(
            output_dir, "hest_inter_spot_graphs.pkl")
        metadata_path = os.path.join(output_dir, "hest_graph_metadata.json")
        processed_data_path = os.path.join(
            output_dir, "hest_processed_data.pkl")

        # 保存spot内图
        with open(intra_graphs_path, 'wb') as f:
            pickle.dump({sid: data['intra_spot_graphs']
                        for sid, data in all_graphs.items()}, f)

        # 保存spot间图
        with open(inter_graphs_path, 'wb') as f:
            pickle.dump({sid: data['inter_spot_graph']
                        for sid, data in all_graphs.items()}, f)

        # 保存元数据
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        # 保存处理后的数据
        with open(processed_data_path, 'wb') as f:
            pickle.dump(self.processed_data, f)

        print(f"\n图数据保存至: {output_dir}")
        print(f"- Spot内图: {intra_graphs_path}")
        print(f"- Spot间图: {inter_graphs_path}")
        print(f"- 元数据: {metadata_path}")
        print(f"- 处理数据: {processed_data_path}")

        return all_metadata


def main():
    """主函数"""

    # 配置参数
    hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
    output_dir = "/data/yujk/hovernet2feature/hest_graphs_normalized_dinov3"

    # 深度特征目录（新增）
    features_dir = "/data/yujk/hovernet2feature/hest_normalized_dinov3"

    # 设置使用的样本范围
    USE_ALL_SAMPLES = True  # 使用所有样本
    MAX_SAMPLES = None  # 不限制样本数（修改：确保使用全部数据集）

    # 首先检查可用的样本
    print("=== 检查可用样本 ===")
    available_samples = []

    # 检查st目录中的样本文件
    st_dir = os.path.join(hest_data_dir, "st")
    if os.path.exists(st_dir):
        for file in os.listdir(st_dir):
            if file.endswith('.h5ad'):
                sample_id = file.replace('.h5ad', '')
                available_samples.append(sample_id)

    available_samples.sort()  # 排序以确保一致性
    print(f"发现可用样本总数: {len(available_samples)}")
    print(f"样本列表: {available_samples}")

    if USE_ALL_SAMPLES:
        # 使用所有可用样本
        sample_ids = available_samples
        if MAX_SAMPLES is not None:
            sample_ids = sample_ids[:MAX_SAMPLES]
        print(f"\n选择模式: 使用所有样本")
        print(f"最大样本数限制: {MAX_SAMPLES if MAX_SAMPLES else '无限制'}")
    else:
        # 只使用结直肠癌相关样本（TENX前缀的样本通常是结直肠癌）
        preferred_samples = ['TENX128', 'TENX139',
                             'TENX147', 'TENX148', 'TENX149']
        sample_ids = [
            sid for sid in preferred_samples if sid in available_samples]

        if not sample_ids:
            # 如果没有找到首选样本，使用TENX开头的样本（修改：使用全部而不是前5个）
            tenx_samples = [
                sid for sid in available_samples if sid.startswith('TENX')]
            sample_ids = tenx_samples if tenx_samples else available_samples

        print(f"\n选择模式: 仅结直肠癌样本")

    print(f"将处理的样本数: {len(sample_ids)}")
    print(f"样本列表: {sample_ids}")

    if not sample_ids:
        print("错误: 未找到可用的样本数据")
        return

    # 图构建参数
    inter_spot_k_neighbors = 6           # spot间k近邻数量

    print("\n=== HEST数据集图构建（直接文件读取+深度特征）===")
    print(f"HEST数据目录: {hest_data_dir}")
    print(f"深度特征目录: {features_dir}")
    print(f"输出目录: {output_dir}")
    print(f"样本列表: {sample_ids}")
    print(f"配置参数:")
    print(f"  - Spot间k近邻: {inter_spot_k_neighbors}")
    print(
        f"  - 使用深度特征: {os.path.exists(features_dir) if features_dir else False}")
    print(f"  - 特征维度: 128 (深度特征) 或自动扩展至128")

    # 检查输入目录
    if not os.path.exists(hest_data_dir):
        print(f"错误: HEST数据目录不存在: {hest_data_dir}")
        return

    # 创建图构建器
    try:
        builder = HESTDirectReader(
            hest_data_dir=hest_data_dir,
            sample_ids=sample_ids,
            features_dir=features_dir,  # 传递深度特征目录
            inter_spot_k_neighbors=inter_spot_k_neighbors
        )

        # 加载HEST数据
        builder.load_sample_data()

        if not builder.sample_data:
            print("错误: 未能加载任何HEST数据")
            return

        # 处理所有样本
        builder.process_all_samples()

        # 构建并保存图
        metadata = builder.save_graphs(output_dir)

        print("\n=== 图构建完成 ===")
        for sample_id, meta in metadata.items():
            print(f"样本 {sample_id}:")
            print(f"  - {meta['intra_graph_count']} 个spot内图")
            print(f"  - 1 个spot间图（{meta['inter_graph_edges']} 条边）")
            print(
                f"  - 总计 {meta['num_assigned_cells']}/{meta['num_cells']} 个分配细胞")
            print(f"  - 组织类型: {meta['tissue']}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    # test

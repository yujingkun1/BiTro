#!/usr/bin/env python3
"""
Bulk数据集静态图构建脚本 - 使用预分割patch - 完整新逻辑版本
基于bulk特征数据和预分割patch构建图结构，支持后续的迁移学习

特征文件格式：
- train/*.parquet: 训练集特征文件
- test/*.parquet: 测试集特征文件
- 每个文件包含128维DINO特征和细胞位置信息

Patch文件格式：
- patches_dir/{patient_id}/*.png: 每个患者的预分割patch文件
- 文件名格式: {patient_id}_patch_tile_{tile_id}_level0_{x1}-{y1}-{x2}-{y2}.png
"""

import os
import pandas as pd
import numpy as np
import torch
import json
import pickle
import re
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class BulkStaticGraphBuilder:
    """Bulk数据集静态图构建器 - 使用预分割patch"""
    
    def __init__(self, 
                 train_features_dir,
                 test_features_dir,
                 bulk_csv_path,
                 patches_dir,                        # 预分割的patch目录
                 wsi_input_dir,                      # 原始WSI文件目录
                 intra_patch_distance_threshold=250,  # patch内细胞连接距离阈值（像素）
                 inter_patch_k_neighbors=6,          # patch间k近邻数量
                 use_deep_features=True,             # 使用深度特征
                 feature_dim=128,                    # 特征维度
                 max_cells_per_patch=None,           # 每个patch的最大细胞数
                 max_train_slides=None,              # 训练集最多处理的特征文件数
                 max_test_slides=None,               # 测试集最多处理的特征文件数
                 checkpoint_dir=None):               # 检查点目录，用于断点续传
        
        self.train_features_dir = train_features_dir
        self.test_features_dir = test_features_dir
        self.bulk_csv_path = bulk_csv_path
        self.patches_dir = patches_dir
        self.wsi_input_dir = wsi_input_dir
        self.intra_patch_distance_threshold = intra_patch_distance_threshold
        self.inter_patch_k_neighbors = inter_patch_k_neighbors
        self.use_deep_features = use_deep_features
        self.feature_dim = feature_dim
        self.max_cells_per_patch = max_cells_per_patch
        self.max_train_slides = max_train_slides
        self.max_test_slides = max_test_slides
        self.checkpoint_dir = checkpoint_dir
        
        self.processed_data = {}
        self.bulk_data = None
        self.valid_patient_ids = []
        self.case_to_bulk_cols = {}
        self.selected_feature_files = {'train': [], 'test': []}
        
        # 如果指定了检查点目录，创建它
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(self.checkpoint_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoint_dir, 'test'), exist_ok=True)
        
    def load_bulk_data(self):
        """加载bulk RNA-seq数据"""
        print("=== 加载bulk RNA-seq数据 ===")
        
        bulk_df = pd.read_csv(self.bulk_csv_path)
        bulk_df["gene_name"] = bulk_df['Unnamed: 0'].str[:15]
        bulk_df = bulk_df.drop(columns=['Unnamed: 0'])
        bulk_df = bulk_df.set_index('gene_name')
        
        original_ids = list(bulk_df.columns)
        case_ids = [pid[:12] for pid in original_ids]  # TCGA病例ID（例如 TCGA-3C-AALI）
        case_id_series = pd.Series(case_ids)
        
        # 建立病例ID到所有bulk列的映射，后续取平均
        case_to_cols = {}
        for case_id, original_id in zip(case_ids, original_ids):
            case_to_cols.setdefault(case_id, []).append(original_id)
        
        multi_col_cases = sum(1 for cols in case_to_cols.values() if len(cols) > 1)
        print(f"病例总数: {len(case_to_cols)}, 其中 {multi_col_cases} 个病例拥有多列bulk数据（将取平均）")
        
        self.bulk_data = bulk_df
        self.case_to_bulk_cols = case_to_cols
        self.valid_patient_ids = list(case_to_cols.keys())
        
        print(f"Bulk数据形状: {bulk_df.shape}")
        print(f"有效病例ID数: {len(self.valid_patient_ids)}")
        
    def extract_slide_id(self, file_path):
        """从文件路径或文件名提取切片ID（包括UUID）"""
        basename = os.path.basename(file_path)
        # 从文件名中提取完整的切片标识符
        # 支持多种格式：
        # 1. TCGA-AA-3872-01A-01-TS1.4f7d5598-e36a-4e30-9b7b-ab55cc6fc3a0_tile36_features.parquet
        # 2. TCGA-A2-A0YI-01A-03-TSC.315f5bb4-4ef4-471e-b5b4-ae73a6038c20_features.parquet
        # 3. TCGA-AA-3872-01A-01-BS1.e29045b5-113d-4dba-b03b-ba2e0d82a388_patch_tile_542_level0_5540-10952-5796-11208.png
        if '_tile36_features.parquet' in basename:
            return basename.replace('_tile36_features.parquet', '')
        elif '_features.parquet' in basename:
            # 新格式：TCGA-A2-A0YI-01A-03-TSC.315f5bb4-4ef4-471e-b5b4-ae73a6038c20_features.parquet
            return basename.replace('_features.parquet', '')
        elif '_patch_tile_' in basename:
            # patch文件格式：TCGA-AA-3872-01A-01-BS1.e29045b5-113d-4dba-b03b-ba2e0d82a388_patch_tile_542_level0_5540-10952-5796-11208.png
            # 提取：TCGA-AA-3872-01A-01-BS1.e29045b5-113d-4dba-b03b-ba2e0d82a388
            parts = basename.split('_patch_tile_')
            if len(parts) >= 2:
                return parts[0]
        return basename
    
    def extract_patient_id_from_slide(self, slide_id):
        """从切片ID提取患者ID"""
        # 从 TCGA-AA-3872-01A-01-TS1.4f7d5598-e36a-4e30-9b7b-ab55cc6fc3a0 
        # 提取病例ID TCGA-AA-3872
        parts = slide_id.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
        return slide_id[:12]
        
    def find_patch_files_by_slide(self, slide_id):
        """根据切片ID查找对应的patch文件"""
        # 直接在patches目录下搜索包含相同slide_id的patch文件
        matching_patch_files = []
        
        # 搜索所有子目录
        for root, dirs, files in os.walk(self.patches_dir):
            for file in files:
                if file.endswith(".png") and "_patch_tile_" in file:
                    file_slide_id = self.extract_slide_id(file)
                    if slide_id == file_slide_id:
                        matching_patch_files.append(os.path.join(root, file))
        
        print(f"  - 切片 {slide_id} 找到 {len(matching_patch_files)} 个匹配的patch文件")
        return matching_patch_files
    
    def parse_patch_coordinates(self, patch_filename):
        """从patch文件名解析坐标信息"""
        # 文件名格式: {patient_id}_patch_tile_{tile_id}_level0_{x1}-{y1}-{x2}-{y2}.png
        basename = os.path.basename(patch_filename)
        
        # 使用正则表达式提取坐标
        pattern = r'_patch_tile_(\d+)_level0_(\d+)-(\d+)-(\d+)-(\d+)\.png'
        match = re.search(pattern, basename)
        
        if match:
            tile_id = int(match.group(1))
            x1, y1, x2, y2 = map(int, match.groups()[1:])
            return {
                'tile_id': tile_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1
            }
        else:
            print(f"警告: 无法解析patch文件名: {basename}")
            return None
    
    def convert_to_absolute_coordinates(self, df):
        """将细胞相对坐标转换为WSI绝对坐标"""
        def parse_tile_coordinates(image_name):
            """从image_name解析tile的绝对坐标"""
            # 示例：TCGA-AA-3844-01A-01-BS1.xxx_patch_tile_1435_level0_12800-8727-13056-8983
            pattern = r'_patch_tile_\d+_level0_(\d+)-(\d+)-(\d+)-(\d+)$'
            match = re.search(pattern, image_name)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return x1, y1, x2, y2
            return None, None, None, None
        
        # 解析每个细胞所属tile的绝对坐标
        tile_coords = df['image_name'].apply(parse_tile_coordinates)
        
        # 将结果转换为单独的列
        df[['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2']] = pd.DataFrame(tile_coords.tolist(), index=df.index)
        
        # 计算绝对坐标：tile起始坐标 + 细胞相对坐标
        df['abs_x'] = df['tile_x1'] + df['x']
        df['abs_y'] = df['tile_y1'] + df['y']
        
        # 用绝对坐标替换相对坐标
        df['x'] = df['abs_x']
        df['y'] = df['abs_y']
        
        # 清理临时列
        df = df.drop(columns=['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2', 'abs_x', 'abs_y', 'image_name'])
        
        return df
    
    def assign_cells_to_patches(self, cells_df, patch_files):
        """将细胞分配到相应的patch"""
        print(f"  - 将 {len(cells_df)} 个细胞分配到 {len(patch_files)} 个patch")
        
        patches = []
        cells_df = cells_df.copy()
        cells_df['patch_id'] = -1
        
        for patch_file in patch_files:
            patch_coords = self.parse_patch_coordinates(patch_file)
            if patch_coords is None:
                continue
            
            # 找到在当前patch范围内的细胞
            patch_mask = ((cells_df['x'] >= patch_coords['x1']) & 
                         (cells_df['x'] < patch_coords['x2']) & 
                         (cells_df['y'] >= patch_coords['y1']) & 
                         (cells_df['y'] < patch_coords['y2']))
            
            patch_cells = cells_df[patch_mask].copy()
            
            if len(patch_cells) > 0:
                patch_id = patch_coords['tile_id']
                cells_df.loc[patch_mask, 'patch_id'] = patch_id
                
                # 调整细胞坐标到patch内的相对位置
                patch_cells_relative = patch_cells.copy()
                patch_cells_relative['x'] = patch_cells['x'] - patch_coords['x1']
                patch_cells_relative['y'] = patch_cells['y'] - patch_coords['y1']
                
                patches.append({
                    'patch_id': patch_id,
                    'cells': patch_cells_relative,
                    'center': [patch_coords['center_x'], patch_coords['center_y']],
                    'bounds': [patch_coords['x1'], patch_coords['x2'], 
                              patch_coords['y1'], patch_coords['y2']],
                    'size': [patch_coords['width'], patch_coords['height']]
                })
        
        assigned_count = len(cells_df[cells_df['patch_id'] >= 0])
        print(f"  - 成功分配 {assigned_count}/{len(cells_df)} 个细胞到 {len(patches)} 个有效patch")
        
        return patches
    
    def build_single_patch_graph(self, patch_cells, patch_id):
        """构建单个patch的图结构（流式处理，立即释放资源）"""
        if len(patch_cells) < 1:
            return None
        
        if len(patch_cells) == 1:
            # 单个细胞的patch
            cell_row = patch_cells.iloc[0]
            cell_features = self.extract_cell_feature_vector(cell_row)
            x = torch.tensor([cell_features], dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            pos = torch.tensor([[cell_row['x'], cell_row['y']]], dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, pos=pos)
        
        # 提取位置和特征（使用patch内相对坐标）
        positions = patch_cells[['x', 'y']].values
        cell_features = np.array([
            self.extract_cell_feature_vector(row) 
            for _, row in patch_cells.iterrows()
        ])
        
        # 计算距离矩阵
        distances = squareform(pdist(positions))
        
        # 基于距离阈值构建邻接矩阵
        adj_matrix = (distances <= self.intra_patch_distance_threshold) & (distances > 0)
        
        # 转换为边列表
        edge_indices = np.where(adj_matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # 如果没有边，使用k近邻连接
        if edge_index.shape[1] == 0:
            k = min(3, len(patch_cells) - 1)
            if k > 0:
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(positions)
                _, indices = nbrs.kneighbors(positions)
                
                edges = []
                for i, neighbors in enumerate(indices):
                    for neighbor in neighbors[1:]:  # 跳过自己
                        edges.extend([[i, neighbor], [neighbor, i]])
                
                if edges:
                    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 创建图数据
        x = torch.tensor(cell_features, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, pos=pos)
    
    def build_intra_patch_graphs(self, patches):
        """构建patch内的图结构（基于细胞）- 保留用于兼容性"""
        intra_patch_graphs = {}
        
        for patch_info in tqdm(patches, desc="构建patch内图"):
            patch_id = patch_info['patch_id']
            patch_cells = patch_info['cells']
            
            graph = self.build_single_patch_graph(patch_cells, patch_id)
            if graph is not None:
                intra_patch_graphs[patch_id] = graph
        
        return intra_patch_graphs
    
    def build_inter_patch_graph(self, patches):
        """构建patch间的图结构"""
        if len(patches) < 2:
            # 只有一个patch的情况
            if len(patches) == 1:
                patch_center = patches[0]['center']
                patch_features = torch.tensor([patch_center], dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                pos = torch.tensor([patch_center], dtype=torch.float32)
                return Data(x=patch_features, edge_index=edge_index, pos=pos)
            else:
                # 没有patch的情况
                return Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        
        # 获取patch中心点（使用原始WSI坐标）
        patch_positions = np.array([patch['center'] for patch in patches])
        
        # 使用k近邻构建patch间连接
        k = min(self.inter_patch_k_neighbors, len(patches) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(patch_positions)
        _, indices = nbrs.kneighbors(patch_positions)
        
        # 构建边列表
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # 跳过自己
                edges.extend([[i, neighbor], [neighbor, i]])
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
        
        # patch特征：使用位置特征
        patch_features = torch.tensor(patch_positions, dtype=torch.float32)
        pos = torch.tensor(patch_positions, dtype=torch.float32)
        
        inter_patch_graph = Data(x=patch_features, edge_index=edge_index, pos=pos)
        
        return inter_patch_graph
    
    def build_inter_patch_graph_from_centers(self, patch_centers):
        """从patch中心点构建patch间的图结构（流式处理版本）"""
        if len(patch_centers) < 2:
            # 只有一个patch的情况
            if len(patch_centers) == 1:
                patch_center = patch_centers[0]
                patch_features = torch.tensor([patch_center], dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                pos = torch.tensor([patch_center], dtype=torch.float32)
                return Data(x=patch_features, edge_index=edge_index, pos=pos)
            else:
                # 没有patch的情况
                return Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        
        # 使用patch中心点
        patch_positions = np.array(patch_centers)
        
        # 使用k近邻构建patch间连接
        k = min(self.inter_patch_k_neighbors, len(patch_centers) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(patch_positions)
        _, indices = nbrs.kneighbors(patch_positions)
        
        # 构建边列表
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # 跳过自己
                edges.extend([[i, neighbor], [neighbor, i]])
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
        
        # patch特征：使用位置特征
        patch_features = torch.tensor(patch_positions, dtype=torch.float32)
        pos = torch.tensor(patch_positions, dtype=torch.float32)
        
        inter_patch_graph = Data(x=patch_features, edge_index=edge_index, pos=pos)
        
        return inter_patch_graph
    
    def extract_cell_feature_vector(self, cell_row):
        """提取细胞特征向量"""
        if self.use_deep_features:
            # 使用深度特征
            features = [cell_row[f'feature_{i}'] for i in range(self.feature_dim)]
            return np.array(features, dtype=np.float32)
        else:
            # 使用几何特征（如果需要的话）
            features = [
                cell_row['x'],
                cell_row['y'],
                cell_row.get('area', 100.0),
                cell_row.get('perimeter', 35.4),
            ]
            # 扩展到目标维度
            features = np.array(features, dtype=np.float32)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant')
            return features[:self.feature_dim]
    
    def get_feature_file_list(self, split='train'):
        """获取特征文件列表（不加载数据）"""
        # 根据split选择目录
        if split == 'train':
            features_dir = self.train_features_dir
        else:
            features_dir = self.test_features_dir

        # 找到所有parquet文件
        feature_files = []
        for root, _, files in os.walk(features_dir):
            for file in files:
                if file.endswith(".parquet"):
                    full_path = os.path.join(root, file)
                    feature_files.append(full_path)

        feature_files = sorted(feature_files)
        limit = self.max_train_slides if split == 'train' else self.max_test_slides
        if limit is not None:
            original_count = len(feature_files)
            feature_files = feature_files[:limit]
            print(f"找到 {original_count} 个{split}集特征文件，选取前 {len(feature_files)} 个进行构建")
        else:
            print(f"找到 {len(feature_files)} 个{split}集特征文件（全部使用）")

        # 记录被选取的特征文件名
        self.selected_feature_files[split] = [os.path.basename(p) for p in feature_files]

        return feature_files

    def get_checkpoint_file_list(self, split='train'):
        """获取检查点文件列表（仅返回文件名，不加载数据）"""
        if not self.checkpoint_dir:
            return []

        checkpoint_dir = os.path.join(self.checkpoint_dir, split)
        if not os.path.exists(checkpoint_dir):
            return []

        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        return checkpoint_files
    
    def save_slide_checkpoint(self, slide_id, slide_data, split='train'):
        """保存单个切片的检查点"""
        if not self.checkpoint_dir:
            return
        
        # 使用安全的文件名（替换特殊字符）
        safe_slide_id = slide_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        checkpoint_path = os.path.join(self.checkpoint_dir, split, f"{safe_slide_id}.pkl")
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(slide_data, f)
        except Exception as e:
            print(f"警告: 无法保存检查点 {checkpoint_path}: {e}")
    
    def load_slide_checkpoint(self, slide_id, split='train'):
        """加载单个切片的检查点"""
        if not self.checkpoint_dir:
            return None
        
        safe_slide_id = slide_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        checkpoint_path = os.path.join(self.checkpoint_dir, split, f"{safe_slide_id}.pkl")
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"警告: 无法加载检查点 {checkpoint_path}: {e}")
                return None
        return None
    
    def load_all_checkpoints(self, split='train'):
        """加载所有已保存的检查点"""
        if not self.checkpoint_dir:
            return {}
        
        checkpoint_dir = os.path.join(self.checkpoint_dir, split)
        if not os.path.exists(checkpoint_dir):
            return {}
        
        checkpoints = {}
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        
        print(f"  发现 {len(checkpoint_files)} 个已保存的{split}集检查点")
        
        for checkpoint_file in tqdm(checkpoint_files, desc=f"加载{split}集检查点"):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            try:
                with open(checkpoint_path, 'rb') as f:
                    slide_data = pickle.load(f)
                    slide_id = slide_data.get('slide_id', checkpoint_file.replace('.pkl', ''))
                    checkpoints[slide_id] = slide_data
            except Exception as e:
                print(f"警告: 无法加载检查点 {checkpoint_path}: {e}")
        
        return checkpoints
    
    def is_slide_processed(self, slide_id, split='train'):
        """检查切片是否已处理"""
        if not self.checkpoint_dir:
            return False
        
        safe_slide_id = slide_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        checkpoint_path = os.path.join(self.checkpoint_dir, split, f"{safe_slide_id}.pkl")
        return os.path.exists(checkpoint_path)
    
    def process_all_slides_new_logic(self):
        """使用新逻辑处理所有切片数据：按切片级别匹配patch，逐个加载和处理切片以减少内存占用，支持断点续传"""
        print("=== 使用切片级别匹配逻辑处理所有数据（逐个切片处理，支持断点续传）===")
        
        # 如果启用了检查点，初始化已处理的数据记录（不预加载，节省内存）
        if self.checkpoint_dir:
            print("\n=== 初始化检查点状态（按需加载，节省内存）===")
            # 只记录已处理的切片ID，不预加载数据
            self.processed_data['train'] = {}
            self.processed_data['test'] = {}

            # 统计已处理的切片数量
            train_checkpoints = self.get_checkpoint_file_list('train')
            test_checkpoints = self.get_checkpoint_file_list('test')
            print(f"  发现训练集已处理切片: {len(train_checkpoints)}")
            print(f"  发现测试集已处理切片: {len(test_checkpoints)}")
        
        # 获取特征文件列表（不加载数据）
        train_feature_files = self.get_feature_file_list('train')
        test_feature_files = self.get_feature_file_list('test')
        
        # 处理训练集：逐个加载和处理切片
        print("\n处理训练集...")
        if 'train' not in self.processed_data:
            self.processed_data['train'] = {}
        
        processed_count = 0
        skipped_count = 0
        
        for feature_file in tqdm(train_feature_files, desc="处理训练集切片"):
            # 提取切片ID（包括UUID）
            slide_id = self.extract_slide_id(feature_file)
            
            # 检查是否已处理（断点续传）
            if slide_id in self.processed_data['train'] or self.is_slide_processed(slide_id, 'train'):
                skipped_count += 1
                continue
            
            # 提取患者ID，检查是否在有效患者列表中
            patient_id = self.extract_patient_id_from_slide(slide_id)
            if patient_id not in self.valid_patient_ids:
                continue
            
            # 加载并处理单个切片
            try:
                df = pd.read_parquet(feature_file)
                
                # 检查必要的列
                required_columns = [f'feature_{i}' for i in range(128)] + ['x', 'y', 'image_name', 'cluster_label']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"警告: 文件 {feature_file} 缺少列: {missing_cols}")
                    continue
                
                # 转换坐标为绝对坐标
                df_processed = self.convert_to_absolute_coordinates(df.copy())
                
                # 立即处理这个切片
                result = self.process_single_slide_new_logic(df_processed, slide_id, patient_id)
                if result:
                    self.processed_data['train'][slide_id] = result
                    # 立即保存检查点
                    self.save_slide_checkpoint(slide_id, result, 'train')
                    processed_count += 1
                
                # 释放内存
                del df, df_processed, result
                
            except Exception as e:
                print(f"错误: 无法处理 {feature_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if skipped_count > 0:
            print(f"  跳过已处理的切片: {skipped_count} 个")
        print(f"  新处理切片: {processed_count} 个")
        
        # 处理测试集：逐个加载和处理切片
        print("\n处理测试集...")
        if 'test' not in self.processed_data:
            self.processed_data['test'] = {}
        
        processed_count = 0
        skipped_count = 0
        
        for feature_file in tqdm(test_feature_files, desc="处理测试集切片"):
            # 提取切片ID（包括UUID）
            slide_id = self.extract_slide_id(feature_file)
            
            # 检查是否已处理（断点续传）
            if slide_id in self.processed_data['test'] or self.is_slide_processed(slide_id, 'test'):
                skipped_count += 1
                continue
            
            # 提取患者ID，检查是否在有效患者列表中
            patient_id = self.extract_patient_id_from_slide(slide_id)
            if patient_id not in self.valid_patient_ids:
                continue
            
            # 加载并处理单个切片
            try:
                df = pd.read_parquet(feature_file)
                
                # 检查必要的列
                required_columns = [f'feature_{i}' for i in range(128)] + ['x', 'y', 'image_name', 'cluster_label']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"警告: 文件 {feature_file} 缺少列: {missing_cols}")
                    continue
                
                # 转换坐标为绝对坐标
                df_processed = self.convert_to_absolute_coordinates(df.copy())
                
                # 立即处理这个切片
                result = self.process_single_slide_new_logic(df_processed, slide_id, patient_id)
                if result:
                    self.processed_data['test'][slide_id] = result
                    # 立即保存检查点
                    self.save_slide_checkpoint(slide_id, result, 'test')
                    processed_count += 1
                
                # 释放内存
                del df, df_processed, result
                
            except Exception as e:
                print(f"错误: 无法处理 {feature_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if skipped_count > 0:
            print(f"  跳过已处理的切片: {skipped_count} 个")
        print(f"  新处理切片: {processed_count} 个")
        
        print(f"\n处理完成:")
        print(f"  - 训练集切片: {len(self.processed_data['train'])}")
        print(f"  - 测试集切片: {len(self.processed_data['test'])}")
        
        # 统计建图情况
        total_slides = len(self.processed_data['train']) + len(self.processed_data['test'])
        slides_with_graphs = 0
        slides_without_graphs = 0
        
        for split_data in [self.processed_data['train'], self.processed_data['test']]:
            for slide_data in split_data.values():
                if slide_data.get('has_graphs', False):
                    slides_with_graphs += 1
                else:
                    slides_without_graphs += 1
        
        print(f"\n建图统计:")
        print(f"  - 总切片数: {total_slides}")
        print(f"  - 成功建图切片: {slides_with_graphs}")
        print(f"  - 仅保留原始特征切片: {slides_without_graphs}")
        
    def process_single_slide_new_logic(self, cells_df, slide_id, patient_id):
        """处理单个切片数据 - 新逻辑：保证所有细胞数据都被保留"""
        print(f"处理切片: {slide_id} (患者: {patient_id})")
        
        if cells_df is None or len(cells_df) == 0:
            print(f"  - 警告: 切片 {slide_id} 没有细胞数据")
            return None
            
        print(f"  - 细胞数量: {len(cells_df)}")
        
        # 提取所有细胞的特征、坐标和聚类标签
        all_cell_features = self.extract_all_cell_features_with_clusters(cells_df)
        all_cell_positions = cells_df[['x', 'y']].values.astype(np.float32)
        cluster_labels = cells_df['cluster_label'].values
        
        # 获取patch文件列表
        patch_files = self.find_patch_files_by_slide(slide_id)
        print(f"  - 匹配的Patch文件数量: {len(patch_files)}")
        
        has_graphs = False
        intra_patch_graphs = {}
        patches = []
        
        if len(patch_files) > 0:
            # 将细胞分配到patch（assign_cells_to_patches内部会修改cells_df的copy，我们需要获取patch_id信息）
            # 先复制一份用于分配，保留原始index
            cells_df_with_patch = cells_df.copy()
            patches = self.assign_cells_to_patches(cells_df_with_patch, patch_files)
            
            # 将patch_id信息从cells_df_with_patch复制回原始cells_df
            if 'patch_id' in cells_df_with_patch.columns:
                cells_df['patch_id'] = cells_df_with_patch['patch_id'].values
            else:
                cells_df['patch_id'] = -1
            
            if len(patches) > 0:
                # 构建patch内图
                intra_patch_graphs = self.build_intra_patch_graphs(patches)
                
                # 构建patch间图
                inter_patch_graph = self.build_inter_patch_graph(patches)
                has_graphs = True
                
                print(f"  - 成功构建图: Patch内图 {len(intra_patch_graphs)} 个")
                print(f"  - Patch间图: {inter_patch_graph.edge_index.shape[1]} 条边")
            else:
                print(f"  - 未能成功分配细胞到patch，将保留原始特征")
                inter_patch_graph = Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        else:
            print(f"  - 未找到匹配的patch文件，将保留原始特征")
            cells_df['patch_id'] = -1
            inter_patch_graph = Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        
        return {
            'slide_id': slide_id,
            'patient_id': patient_id,
            'cells_df': cells_df,
            'patches': patches,
            'intra_patch_graphs': intra_patch_graphs,
            'inter_patch_graph': inter_patch_graph,
            'bulk_expr': self.get_bulk_expression(patient_id),  # 仍然使用患者ID获取bulk表达
            'has_graphs': has_graphs,
            'all_cell_features': all_cell_features,          # 所有细胞的DINO特征
            'all_cell_positions': torch.tensor(all_cell_positions),  # 所有细胞的空间坐标
            'cluster_labels': torch.tensor(cluster_labels),         # 所有细胞的聚类标签
            'cell_to_graph_mapping': self.build_cell_to_graph_mapping(cells_df, patches) if has_graphs else None
        }
    
    def extract_all_cell_features_with_clusters(self, cells_df):
        """提取所有细胞的DINO特征"""
        if cells_df is None or len(cells_df) == 0:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)
        
        # 提取DINO特征 (feature_0 到 feature_127)
        feature_columns = [f'feature_{i}' for i in range(128)]
        features_matrix = cells_df[feature_columns].values.astype(np.float32)
        
        return torch.tensor(features_matrix, dtype=torch.float32)
    
    def build_cell_to_graph_mapping(self, cells_df, patches):
        """构建细胞到图的映射关系"""
        if not patches:
            return None
            
        cell_to_graph = {}
        
        for patch_info in patches:
            patch_id = patch_info['patch_id']
            patch_cells = patch_info['cells']
            
            # 为这个patch中的每个细胞建立映射
            for cell_idx in patch_cells.index:
                if cell_idx in cells_df.index:
                    cell_to_graph[cell_idx] = {
                        'patch_id': patch_id,
                        'has_graph': True
                    }
        
        return cell_to_graph
    
    def get_bulk_expression(self, patient_id):
        """获取患者的bulk表达数据"""
        if self.bulk_data is None:
            return None
            
        # 找到匹配的列（病例级）
        bulk_cols = self.case_to_bulk_cols.get(patient_id, [])
        if not bulk_cols:
            print(f"警告: 未找到病例 {patient_id} 的bulk数据")
            return None
        
        bulk_values = self.bulk_data[bulk_cols].values.astype(np.float32)
        if bulk_values.ndim == 2 and bulk_values.shape[1] > 1:
            # 多列：取平均值
            print(f"信息: 病例 {patient_id} 有 {bulk_values.shape[1]} 列bulk数据，使用平均值")
            print(f"  - 可用列: {bulk_cols}")
            return np.mean(bulk_values, axis=1)
        
        # 单列：直接返回（values 可能是(N, 1)，需要压缩为(N,))
        return bulk_values.reshape(-1)
    
    def save_selected_feature_filenames(self, output_dir):
        """保存被选中的训练/测试特征文件名"""
        print("=== 保存特征文件名列表 ===")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for split in ['train', 'test']:
            filenames = self.selected_feature_files.get(split, [])
            txt_path = os.path.join(output_dir, f"{split}_selected_feature_files.txt")
            with open(txt_path, 'w') as f:
                f.write('\n'.join(filenames))
            print(f"  - {split}集文件列表: {txt_path} (共 {len(filenames)} 个)")
    
    def save_graphs_slide_logic(self, output_dir):
        """保存构建的图数据 - 切片逻辑：从检查点文件加载并保存，节省内存"""
        print("=== 保存图结构和完整细胞数据（切片级别，从检查点加载）===")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 分别保存训练集和测试集的图数据
        for split in ['train', 'test']:
            # 获取所有检查点文件列表
            checkpoint_files = self.get_checkpoint_file_list(split)
            if not checkpoint_files:
                print(f"{split}集没有找到检查点文件，跳过")
                continue

            print(f"{split}集发现 {len(checkpoint_files)} 个检查点文件，开始分批加载并保存")

            # 分批处理，避免内存溢出（每批处理5个文件）
            batch_size = 5
            total_batches = (len(checkpoint_files) + batch_size - 1) // batch_size

            # 准备保存的数据（分批累积）
            intra_graphs = {}
            inter_graphs = {}
            bulk_expressions = {}
            all_cell_features = {}
            all_cell_positions = {}
            cluster_labels = {}
            graph_status = {}
            cell_to_graph_mappings = {}
            slide_to_patient_mapping = {}
            metadata = {}

            checkpoint_dir = os.path.join(self.checkpoint_dir, split)

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(checkpoint_files))
                batch_files = checkpoint_files[start_idx:end_idx]

                print(f"  处理{split}集第 {batch_idx + 1}/{total_batches} 批 ({len(batch_files)} 个文件)...")

                # 分批加载检查点
                for checkpoint_file in batch_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                    try:
                        with open(checkpoint_path, 'rb') as f:
                            slide_data = pickle.load(f)
                            slide_id = slide_data.get('slide_id', checkpoint_file.replace('.pkl', ''))

                            # 累积数据
                            intra_graphs[slide_id] = slide_data['intra_patch_graphs']
                            inter_graphs[slide_id] = slide_data['inter_patch_graph']
                            bulk_expressions[slide_id] = slide_data['bulk_expr']
                            all_cell_features[slide_id] = slide_data['all_cell_features']
                            all_cell_positions[slide_id] = slide_data['all_cell_positions']
                            cluster_labels[slide_id] = slide_data['cluster_labels']
                            graph_status[slide_id] = slide_data.get('has_graphs', False)
                            cell_to_graph_mappings[slide_id] = slide_data.get('cell_to_graph_mapping', None)
                            slide_to_patient_mapping[slide_id] = slide_data['patient_id']

                            metadata[slide_id] = {
                                'slide_id': slide_id,
                                'patient_id': slide_data['patient_id'],
                                'num_cells': len(slide_data['cells_df']),
                                'num_patches': len(slide_data['patches']),
                                'intra_graph_count': len(slide_data['intra_patch_graphs']),
                                'inter_graph_edges': slide_data['inter_patch_graph'].edge_index.shape[1],
                                'has_bulk_expr': slide_data['bulk_expr'] is not None,
                                'has_graphs': slide_data.get('has_graphs', False),
                                'total_cell_features': slide_data['all_cell_features'].shape[0],
                                'cell_feature_dim': slide_data['all_cell_features'].shape[1]
                            }
                    except Exception as e:
                        print(f"警告: 无法加载检查点 {checkpoint_path}: {e}")
                        continue

                # 每批处理完后，立即保存中间结果，避免内存累积过多
                if batch_idx < total_batches - 1:  # 不是最后一批时进行中间保存
                    print(f"    第 {batch_idx + 1} 批处理完成，保存中间结果...")
                    self._save_split_data_partial(split, intra_graphs, inter_graphs, bulk_expressions,
                                                all_cell_features, all_cell_positions, cluster_labels,
                                                graph_status, cell_to_graph_mappings, slide_to_patient_mapping,
                                                metadata, output_dir, batch_idx + 1)

                    # 清理已保存的数据，释放内存
                    intra_graphs.clear()
                    inter_graphs.clear()
                    bulk_expressions.clear()
                    all_cell_features.clear()
                    all_cell_positions.clear()
                    cluster_labels.clear()
                    graph_status.clear()
                    cell_to_graph_mappings.clear()
                    slide_to_patient_mapping.clear()
                    metadata.clear()
            
            # 保存最终结果
            self._save_split_data_final(split, intra_graphs, inter_graphs, bulk_expressions,
                                      all_cell_features, all_cell_positions, cluster_labels,
                                      graph_status, cell_to_graph_mappings, slide_to_patient_mapping,
                                      metadata, output_dir)
            
            # 统计信息
            total_slides = len(metadata)
            slides_with_graphs = sum([status for status in graph_status.values()])
            slides_without_graphs = total_slides - slides_with_graphs
            unique_patients = len(set(slide_to_patient_mapping.values()))
            
            print(f"{split}集保存完成:")
            print(f"  - 总切片数: {total_slides}")
            print(f"  - 覆盖患者数: {unique_patients}")
            print(f"  - 有图数据切片: {slides_with_graphs}")
            print(f"  - 无图数据切片: {slides_without_graphs} (保留完整DINO特征)")
            print(f"  - Patch内图: {intra_path}")
            print(f"  - Patch间图: {inter_path}")
            print(f"  - Bulk表达: {bulk_path}")
            print(f"  - 细胞特征: {features_path}")
            print(f"  - 细胞坐标: {positions_path}")
            print(f"  - 聚类标签: {clusters_path}")
            print(f"  - 图状态: {status_path}")
            print(f"  - 细胞映射: {mappings_path}")
            print(f"  - 切片映射: {slide_mappings_path}")  # 新增
            print(f"  - 元数据: {metadata_path}")
        
        # 保存全局配置
        config = {
            'feature_dim': self.feature_dim,
            'intra_patch_distance_threshold': self.intra_patch_distance_threshold,
            'inter_patch_k_neighbors': self.inter_patch_k_neighbors,
            'use_deep_features': self.use_deep_features,
            'max_cells_per_patch': self.max_cells_per_patch,
            'num_genes': len(self.bulk_data.index) if self.bulk_data is not None else 0,
            'gene_names': self.bulk_data.index.tolist() if self.bulk_data is not None else [],
            'patches_dir': self.patches_dir,
            'wsi_input_dir': self.wsi_input_dir,
            'supports_no_graph_patients': True,
            'uses_dino_files_directly': True,
            'preserves_cluster_labels': True,
            'uses_slide_level_matching': True,  # 新增：标记使用切片级别匹配
            'allows_multiple_slides_per_patient': True  # 新增：允许一个患者多个切片
        }
        
        config_path = os.path.join(output_dir, "bulk_graph_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n配置文件: {config_path}")
        return metadata
        """保存构建的图数据 - 新逻辑：保存完整的细胞特征数据"""
        print("=== 保存图结构和完整细胞数据 ===")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分别保存训练集和测试集的图数据
        for split in ['train', 'test']:
            if split not in self.processed_data:
                continue
                
            split_data = self.processed_data[split]
            
            # 准备保存的数据
            intra_graphs = {}
            inter_graphs = {}
            bulk_expressions = {}
            all_cell_features = {}       # 所有细胞的DINO特征
            all_cell_positions = {}      # 所有细胞的空间坐标  
            cluster_labels = {}          # 所有细胞的聚类标签
            graph_status = {}            # 每个患者是否有图数据的状态
            cell_to_graph_mappings = {}  # 细胞到图的映射关系
            metadata = {}
            
            for patient_id, patient_data in split_data.items():
                intra_graphs[patient_id] = patient_data['intra_patch_graphs']
                inter_graphs[patient_id] = patient_data['inter_patch_graph']
                bulk_expressions[patient_id] = patient_data['bulk_expr']
                all_cell_features[patient_id] = patient_data['all_cell_features']
                all_cell_positions[patient_id] = patient_data['all_cell_positions']
                cluster_labels[patient_id] = patient_data['cluster_labels']
                graph_status[patient_id] = patient_data.get('has_graphs', False)
                cell_to_graph_mappings[patient_id] = patient_data.get('cell_to_graph_mapping', None)
                
                metadata[patient_id] = {
                    'num_cells': len(patient_data['cells_df']),
                    'num_patches': len(patient_data['patches']),
                    'intra_graph_count': len(patient_data['intra_patch_graphs']),
                    'inter_graph_edges': patient_data['inter_patch_graph'].edge_index.shape[1],
                    'has_bulk_expr': patient_data['bulk_expr'] is not None,
                    'has_graphs': patient_data.get('has_graphs', False),
                    'total_cell_features': patient_data['all_cell_features'].shape[0],
                    'cell_feature_dim': patient_data['all_cell_features'].shape[1]
                }
            
            # 保存文件
            intra_path = os.path.join(output_dir, f"bulk_{split}_intra_patch_graphs.pkl")
            inter_path = os.path.join(output_dir, f"bulk_{split}_inter_patch_graphs.pkl")
            bulk_path = os.path.join(output_dir, f"bulk_{split}_expressions.pkl")
            features_path = os.path.join(output_dir, f"bulk_{split}_all_cell_features.pkl")
            positions_path = os.path.join(output_dir, f"bulk_{split}_all_cell_positions.pkl")
            clusters_path = os.path.join(output_dir, f"bulk_{split}_cluster_labels.pkl")
            status_path = os.path.join(output_dir, f"bulk_{split}_graph_status.pkl")
            mappings_path = os.path.join(output_dir, f"bulk_{split}_cell_to_graph_mappings.pkl")
            metadata_path = os.path.join(output_dir, f"bulk_{split}_metadata.json")
            
            with open(intra_path, 'wb') as f:
                pickle.dump(intra_graphs, f)
            
            with open(inter_path, 'wb') as f:
                pickle.dump(inter_graphs, f)
            
            with open(bulk_path, 'wb') as f:
                pickle.dump(bulk_expressions, f)
                
            with open(features_path, 'wb') as f:
                pickle.dump(all_cell_features, f)
                
            with open(positions_path, 'wb') as f:
                pickle.dump(all_cell_positions, f)
                
            with open(clusters_path, 'wb') as f:
                pickle.dump(cluster_labels, f)
                
            with open(status_path, 'wb') as f:
                pickle.dump(graph_status, f)
                
            with open(mappings_path, 'wb') as f:
                pickle.dump(cell_to_graph_mappings, f)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 统计信息
            total_patients = len(split_data)
            patients_with_graphs = sum([status for status in graph_status.values()])
            patients_without_graphs = total_patients - patients_with_graphs
            
            print(f"{split}集保存完成:")
            print(f"  - 总患者数: {total_patients}")
            print(f"  - 有图数据患者: {patients_with_graphs}")
            print(f"  - 无图数据患者: {patients_without_graphs} (保留完整DINO特征)")
            print(f"  - Patch内图: {intra_path}")
            print(f"  - Patch间图: {inter_path}")
            print(f"  - Bulk表达: {bulk_path}")
            print(f"  - 细胞特征: {features_path}")
            print(f"  - 细胞坐标: {positions_path}")
            print(f"  - 聚类标签: {clusters_path}")
            print(f"  - 图状态: {status_path}")
            print(f"  - 细胞映射: {mappings_path}")
            print(f"  - 元数据: {metadata_path}")
        
        # 保存全局配置
        config = {
            'feature_dim': self.feature_dim,
            'intra_patch_distance_threshold': self.intra_patch_distance_threshold,
            'inter_patch_k_neighbors': self.inter_patch_k_neighbors,
            'use_deep_features': self.use_deep_features,
            'max_cells_per_patch': self.max_cells_per_patch,
            'num_genes': len(self.bulk_data.index) if self.bulk_data is not None else 0,
            'gene_names': self.bulk_data.index.tolist() if self.bulk_data is not None else [],
            'patches_dir': self.patches_dir,
            'wsi_input_dir': self.wsi_input_dir,
            'supports_no_graph_patients': True,
            'uses_dino_files_directly': True,  # 新增：标记直接使用DINO文件
            'preserves_cluster_labels': True   # 新增：标记保留聚类标签
        }
        
        config_path = os.path.join(output_dir, "bulk_graph_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n配置文件: {config_path}")
        return metadata

    def _save_split_data_partial(self, split, intra_graphs, inter_graphs, bulk_expressions,
                                all_cell_features, all_cell_positions, cluster_labels,
                                graph_status, cell_to_graph_mappings, slide_to_patient_mapping,
                                metadata, output_dir, batch_idx):
        """分批保存中间结果（用于内存优化）"""
        temp_dir = os.path.join(output_dir, f"temp_{split}_batch_{batch_idx}")
        os.makedirs(temp_dir, exist_ok=True)

        # 保存到临时目录
        paths = {
            'intra': os.path.join(temp_dir, f"bulk_{split}_intra_patch_graphs_batch_{batch_idx}.pkl"),
            'inter': os.path.join(temp_dir, f"bulk_{split}_inter_patch_graphs_batch_{batch_idx}.pkl"),
            'bulk': os.path.join(temp_dir, f"bulk_{split}_expressions_batch_{batch_idx}.pkl"),
            'features': os.path.join(temp_dir, f"bulk_{split}_all_cell_features_batch_{batch_idx}.pkl"),
            'positions': os.path.join(temp_dir, f"bulk_{split}_all_cell_positions_batch_{batch_idx}.pkl"),
            'clusters': os.path.join(temp_dir, f"bulk_{split}_cluster_labels_batch_{batch_idx}.pkl"),
            'status': os.path.join(temp_dir, f"bulk_{split}_graph_status_batch_{batch_idx}.pkl"),
            'mappings': os.path.join(temp_dir, f"bulk_{split}_cell_to_graph_mappings_batch_{batch_idx}.pkl"),
            'slide_mappings': os.path.join(temp_dir, f"bulk_{split}_slide_to_patient_mapping_batch_{batch_idx}.pkl"),
            'metadata': os.path.join(temp_dir, f"bulk_{split}_metadata_batch_{batch_idx}.json")
        }

        for name, path in paths.items():
            if name.endswith('.pkl'):
                with open(path, 'wb') as f:
                    if name == 'intra':
                        pickle.dump(intra_graphs, f)
                    elif name == 'inter':
                        pickle.dump(inter_graphs, f)
                    elif name == 'bulk':
                        pickle.dump(bulk_expressions, f)
                    elif name == 'features':
                        pickle.dump(all_cell_features, f)
                    elif name == 'positions':
                        pickle.dump(all_cell_positions, f)
                    elif name == 'clusters':
                        pickle.dump(cluster_labels, f)
                    elif name == 'status':
                        pickle.dump(graph_status, f)
                    elif name == 'mappings':
                        pickle.dump(cell_to_graph_mappings, f)
                    elif name == 'slide_mappings':
                        pickle.dump(slide_to_patient_mapping, f)
            else:  # metadata json
                with open(path, 'w') as f:
                    json.dump(metadata, f, indent=2)

    def _save_split_data_final(self, split, intra_graphs, inter_graphs, bulk_expressions,
                              all_cell_features, all_cell_positions, cluster_labels,
                              graph_status, cell_to_graph_mappings, slide_to_patient_mapping,
                              metadata, output_dir):
        """最终保存所有数据（合并所有批次的结果）"""
        print(f"  合并并保存{split}集最终结果...")

        # 检查是否有临时批次文件需要合并
        temp_pattern = f"temp_{split}_batch_*"
        import glob
        temp_dirs = glob.glob(os.path.join(output_dir, temp_pattern))

        if temp_dirs:
            print(f"  发现 {len(temp_dirs)} 个临时批次，需要合并")
            # 重新从所有检查点文件加载完整数据
            print(f"  重新从检查点加载{split}集的完整数据...")
            checkpoint_dir_path = os.path.join(self.checkpoint_dir, split)
            all_checkpoints = self.load_all_checkpoints(split)

            # 从加载的检查点重新构建数据
            for slide_id, slide_data in all_checkpoints.items():
                intra_graphs[slide_id] = slide_data['intra_patch_graphs']
                inter_graphs[slide_id] = slide_data['inter_patch_graph']
                bulk_expressions[slide_id] = slide_data['bulk_expr']
                all_cell_features[slide_id] = slide_data['all_cell_features']
                all_cell_positions[slide_id] = slide_data['all_cell_positions']
                cluster_labels[slide_id] = slide_data['cluster_labels']
                graph_status[slide_id] = slide_data.get('has_graphs', False)
                cell_to_graph_mappings[slide_id] = slide_data.get('cell_to_graph_mapping', None)
                slide_to_patient_mapping[slide_id] = slide_data['patient_id']

                metadata[slide_id] = {
                    'slide_id': slide_id,
                    'patient_id': slide_data['patient_id'],
                    'num_cells': len(slide_data['cells_df']),
                    'num_patches': len(slide_data['patches']),
                    'intra_graph_count': len(slide_data['intra_patch_graphs']),
                    'inter_graph_edges': slide_data['inter_patch_graph'].edge_index.shape[1],
                    'has_bulk_expr': slide_data['bulk_expr'] is not None,
                    'has_graphs': slide_data.get('has_graphs', False),
                    'total_cell_features': slide_data['all_cell_features'].shape[0],
                    'cell_feature_dim': slide_data['all_cell_features'].shape[1]
                }

        # 最终保存路径
        intra_path = os.path.join(output_dir, f"bulk_{split}_intra_patch_graphs.pkl")
        inter_path = os.path.join(output_dir, f"bulk_{split}_inter_patch_graphs.pkl")
        bulk_path = os.path.join(output_dir, f"bulk_{split}_expressions.pkl")
        features_path = os.path.join(output_dir, f"bulk_{split}_all_cell_features.pkl")
        positions_path = os.path.join(output_dir, f"bulk_{split}_all_cell_positions.pkl")
        clusters_path = os.path.join(output_dir, f"bulk_{split}_cluster_labels.pkl")
        status_path = os.path.join(output_dir, f"bulk_{split}_graph_status.pkl")
        mappings_path = os.path.join(output_dir, f"bulk_{split}_cell_to_graph_mappings.pkl")
        slide_mappings_path = os.path.join(output_dir, f"bulk_{split}_slide_to_patient_mapping.pkl")
        metadata_path = os.path.join(output_dir, f"bulk_{split}_metadata.json")

        with open(intra_path, 'wb') as f:
            pickle.dump(intra_graphs, f)

        with open(inter_path, 'wb') as f:
            pickle.dump(inter_graphs, f)

        with open(bulk_path, 'wb') as f:
            pickle.dump(bulk_expressions, f)

        with open(features_path, 'wb') as f:
            pickle.dump(all_cell_features, f)

        with open(positions_path, 'wb') as f:
            pickle.dump(all_cell_positions, f)

        with open(clusters_path, 'wb') as f:
            pickle.dump(cluster_labels, f)

        with open(status_path, 'wb') as f:
            pickle.dump(graph_status, f)

        with open(mappings_path, 'wb') as f:
            pickle.dump(cell_to_graph_mappings, f)

        with open(slide_mappings_path, 'wb') as f:
            pickle.dump(slide_to_patient_mapping, f)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 统计信息
        total_slides = len(metadata)
        slides_with_graphs = sum([status for status in graph_status.values()])
        slides_without_graphs = total_slides - slides_with_graphs
        unique_patients = len(set(slide_to_patient_mapping.values()))

        print(f"{split}集保存完成:")
        print(f"  - 总切片数: {total_slides}")
        print(f"  - 覆盖患者数: {unique_patients}")
        print(f"  - 有图数据切片: {slides_with_graphs}")
        print(f"  - 无图数据切片: {slides_without_graphs} (保留完整DINO特征)")
        print(f"  - Patch内图: {intra_path}")
        print(f"  - Patch间图: {inter_path}")
        print(f"  - Bulk表达: {bulk_path}")
        print(f"  - 细胞特征: {features_path}")
        print(f"  - 细胞坐标: {positions_path}")
        print(f"  - 聚类标签: {clusters_path}")
        print(f"  - 图状态: {status_path}")
        print(f"  - 细胞映射: {mappings_path}")
        print(f"  - 切片映射: {slide_mappings_path}")
        print(f"  - 元数据: {metadata_path}")


def main():
    """主函数"""
    
    # 配置参数
    train_features_dir = "/media/yujk/Elements/ouput_features/PRAD_train"
    test_features_dir = "/media/yujk/Elements/ouput_features/PRAD_test"
    bulk_csv_path = "/data/hdd2/yujk/TPM/tpm-TCGA-PRAD.csv"
    patches_dir = "/data/hdd2/yujk/TCGA_patches/PRAD"
    wsi_input_dir = "/data/hdd2/yujk/TCGA/PRAD"
    output_dir = "/data/hdd1/yujk/bulk_PRAD_graphs_new_all_graph"
    checkpoint_dir = "/data/hdd1/yujk/bulk_PRAD_graphs_checkpoints"  # 检查点目录，用于断点续传
    
    # 图构建参数（方案3：提升GPU利用率的新参数）
    intra_patch_distance_threshold = 256   # patch内细胞连接距离阈值（像素）- 从250增加到256
    inter_patch_k_neighbors = 8           # patch间k近邻数量 - 从6增加到8
    use_deep_features = True              # 使用深度特征
    feature_dim = 128                     # 特征维度
    max_cells_per_patch = None           # 每个patch的最大细胞数 - 不限制
    max_train_slides = 200              # 仅处理训练集前N个特征文件，None表示全部
    max_test_slides = 50               # 仅处理测试集前N个特征文件，None表示全部
    
    print("=== Bulk数据集静态图构建（使用预分割patch）- 新逻辑版本（支持断点续传）===")
    print(f"训练特征目录: {train_features_dir}")
    print(f"测试特征目录: {test_features_dir}")
    print(f"Bulk数据文件: {bulk_csv_path}")
    print(f"Patch目录: {patches_dir}")
    print(f"WSI输入目录: {wsi_input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"检查点目录: {checkpoint_dir}")
    print(f"配置参数:")
    print(f"  - Patch内距离阈值: {intra_patch_distance_threshold}px")
    print(f"  - Patch间k近邻: {inter_patch_k_neighbors}")
    train_limit_text = max_train_slides if max_train_slides is not None else '全部'
    test_limit_text = max_test_slides if max_test_slides is not None else '全部'
    print(f"  - 使用深度特征: {use_deep_features}")
    print(f"  - 特征维度: {feature_dim}")
    print(f"  - 每patch最大细胞数: {max_cells_per_patch}")
    print(f"  - 训练集特征文件上限: {train_limit_text}")
    print(f"  - 测试集特征文件上限: {test_limit_text}")
    print(f"  - 断点续传: {'启用' if checkpoint_dir else '禁用'}")
    
    # 检查输入目录
    for path, name in [(train_features_dir, "训练特征目录"), (test_features_dir, "测试特征目录"), 
                       (bulk_csv_path, "Bulk数据文件"), (patches_dir, "Patch目录"), (wsi_input_dir, "WSI输入目录")]:
        if not os.path.exists(path):
            print(f"错误: {name}不存在: {path}")
            return
    
    # 创建图构建器
    try:
        builder = BulkStaticGraphBuilder(
            train_features_dir=train_features_dir,
            test_features_dir=test_features_dir,
            bulk_csv_path=bulk_csv_path,
            patches_dir=patches_dir,
            wsi_input_dir=wsi_input_dir,
            intra_patch_distance_threshold=intra_patch_distance_threshold,
            inter_patch_k_neighbors=inter_patch_k_neighbors,
            use_deep_features=use_deep_features,
            feature_dim=feature_dim,
            max_cells_per_patch=max_cells_per_patch,
            max_train_slides=max_train_slides,
            max_test_slides=max_test_slides,
            checkpoint_dir=checkpoint_dir
        )
        
        # 加载bulk数据
        builder.load_bulk_data()
        
        # 处理所有切片 - 使用切片级别匹配逻辑
        builder.process_all_slides_new_logic()
        
        # 构建并保存图 - 使用切片级别保存逻辑
        metadata = builder.save_graphs_slide_logic(output_dir)
        builder.save_selected_feature_filenames(output_dir)
        
        print("\n=== 图构建完成（切片级别匹配，0%数据丢失版本）===")
        for split in ['train', 'test']:
            total_slides = len(builder.processed_data.get(split, {}))
            print(f"{split}集:")
            print(f"  - 切片数: {total_slides}")
            if total_slides > 0:
                # 从processed_data中计算统计信息
                split_slides = builder.processed_data.get(split, {})
                if split_slides:
                    avg_cells = np.mean([len(s['cells_df']) for s in split_slides.values()])
                    avg_patches = np.mean([len(s['patches']) for s in split_slides.values()])
                    has_graphs_count = sum([1 for s in split_slides.values() if s.get('has_graphs', False)])
                    no_graphs_count = total_slides - has_graphs_count
                    unique_patients = len(set([s['patient_id'] for s in split_slides.values()]))
                    print(f"  - 覆盖患者数: {unique_patients}")
                    print(f"  - 平均细胞数/切片: {avg_cells:.0f}")
                    print(f"  - 平均patch数/切片: {avg_patches:.1f}")
                    print(f"  - 有图切片: {has_graphs_count}")
                    print(f"  - 无图切片: {no_graphs_count} (保留完整DINO特征)")
        
        print("\n✅ 完成：实现切片级别精确匹配，支持混合处理（有图增强 + 无图原始特征）")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
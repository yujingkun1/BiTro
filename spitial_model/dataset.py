#!/usr/bin/env python3
"""
Dataset Module for Cell2Gene HEST Spatial Gene Expression Prediction

author: Jingkun Yu
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import pickle
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

try:
    from torch_geometric.data import Data
    GEOMETRIC_AVAILABLE = True
except ImportError:
    GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available")


class HESTSpatialDataset(Dataset):
    """
    HEST Spatial Transcriptomics Dataset for Cell2Gene prediction
    """

    def __init__(self,
                 hest_data_dir,
                 graph_dir,
                 sample_ids,
                 features_dir=None,
                 feature_dim=128,
                 mode='train',
                 seed=42,
                 gene_file=None,
                 apply_gene_normalization=True,
                 normalization_stats=None,
                 normalization_eps=1e-6):

        self.feature_dim = feature_dim
        self.mode = mode
        self.graph_dir = graph_dir
        self.hest_data_dir = hest_data_dir
        self.sample_ids = sample_ids if isinstance(
            sample_ids, list) else [sample_ids]
        self.features_dir = features_dir
        self.seed = seed
        self.gene_file = gene_file
        self.apply_gene_normalization = apply_gene_normalization
        self._normalization_eps = normalization_eps
        self._provided_normalization_stats = normalization_stats
        self.gene_mean_tensor = None
        self.gene_std_tensor = None
        self.gene_mean_np = None
        self.gene_std_np = None
        self.normalization_stats = None

        print(f"=== Initializing HEST Dataset (mode: {mode}) ===")
        print(f"Sample count: {len(self.sample_ids)}")
        print(f"Sample list: {self.sample_ids}")

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 加载基因映射
        self.load_gene_mapping()

        # 加载HEST数据
        self.load_hest_data()

        # 加载细胞特征数据（用于无图时直接使用特征）
        self.load_feature_data()

        # 加载图数据
        self.load_graph_data()

        # Setup per-gene normalization if required
        self.setup_gene_normalization()

        print(f"=== Dataset initialization complete ===")
        print(f"Total spots: {len(self.all_spots_data)}")
        print(f"Graph files available: {self.graphs_available}")
        print(f"Genes count: {self.num_genes}")

    def load_gene_mapping(self):
        """Load intersection gene list"""
        print("=== Loading intersection gene list ===")

        # Load intersection gene list (preserve order from file!)
        if self.gene_file is None:
            self.gene_file = "/data/yujk/hovernet2feature/HisToGene/data/her_hvg_cut_1000.txt"

        print(f"Loading intersection gene list from: {self.gene_file}")
        selected_genes_ordered = []
        with open(self.gene_file, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith('Efficiently') and not gene.startswith('Total') and not gene.startswith('Detection') and not gene.startswith('Samples'):
                    selected_genes_ordered.append(gene)

        # Remove duplicates while preserving order
        seen = set()
        self.selected_genes = [
            g for g in selected_genes_ordered if not (g in seen or seen.add(g))]
        self.selected_genes_set = set(self.selected_genes)

        print(f"Final genes count: {len(self.selected_genes)}")

    def load_hest_data(self):
        """Load HEST data files directly"""
        print("=== Loading HEST data files ===")

        self.hest_data = {}
        self.all_spots_data = []

        # Per-sample cached gene indexers for fast aligned extraction
        self.sample_gene_indexer = {}
        self.sample_gene_present_mask = {}

        for sample_id in self.sample_ids:
            try:
                print(f"Loading sample: {sample_id}")

                sample_info = {}

                # 1. Load AnnData file
                st_file = os.path.join(
                    self.hest_data_dir, "st", f"{sample_id}.h5ad")
                if os.path.exists(st_file):
                    adata = sc.read_h5ad(st_file)
                    sample_info['adata'] = adata
                    print(
                        f"  - AnnData: {adata.n_obs} spots × {adata.n_vars} genes")
                else:
                    print(f"  - Warning: AnnData file not found: {st_file}")
                    continue

                # 2. Load metadata
                metadata_file = os.path.join(
                    self.hest_data_dir, "metadata", f"{sample_id}.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    sample_info['metadata'] = metadata
                    print(
                        f"  - Metadata: {metadata.get('tissue', 'unknown')} tissue")
                else:
                    sample_info['metadata'] = {}
                    print(
                        f"  - Warning: Metadata file not found: {metadata_file}")

                # Store sample info
                self.hest_data[sample_id] = sample_info

                # Build gene indexer aligned to selected_genes order
                adata_gene_index = pd.Index(adata.var.index)
                gene_indexer = adata_gene_index.get_indexer(
                    self.selected_genes)
                present_mask = gene_indexer >= 0
                self.sample_gene_indexer[sample_id] = gene_indexer
                self.sample_gene_present_mask[sample_id] = present_mask

                # Extract spots data
                adata = sample_info['adata']
                for spot_idx in range(adata.n_obs):
                    spot_id = adata.obs.index[spot_idx]
                    # Initialize with zero expression (will be updated later)
                    gene_expression = np.zeros(
                        len(self.selected_genes), dtype=np.float32)

                    self.all_spots_data.append({
                        'sample_id': sample_id,
                        'spot_idx': spot_idx,
                        'spot_id': spot_id,
                        'gene_expression': gene_expression
                    })

            except Exception as e:
                print(f"  Error loading {sample_id}: {e}")
                continue

        print(f"Successfully loaded {len(self.hest_data)} samples")
        print(f"Total spots loaded: {len(self.all_spots_data)}")

        # Update gene expression data with intersection genes
        if self.hest_data:
            print(
                f"Using specified {len(self.selected_genes)} intersection genes")

            for spot_data in self.all_spots_data:
                sample_id = spot_data['sample_id']
                adata = self.hest_data[sample_id]['adata']
                spot_idx = spot_data['spot_idx']

                # Precomputed mapping from selected_genes -> adata columns
                gene_indexer = self.sample_gene_indexer[sample_id]
                present_mask = self.sample_gene_present_mask[sample_id]

                if present_mask.any():
                    # Pull values for present genes in one go, aligned to selected_genes order
                    adata_cols = gene_indexer[present_mask]
                    # Slice adata row for present genes
                    values = adata.X[spot_idx, adata_cols]
                    if hasattr(values, 'toarray'):
                        values = values.toarray().flatten()
                    else:
                        values = np.asarray(values).flatten()

                    gene_expression = np.zeros(
                        len(self.selected_genes), dtype=np.float32)
                    gene_expression[present_mask] = values.astype(np.float32)

                    # Apply log1p transformation to gene expression data
                    gene_expression = np.log1p(
                        gene_expression).astype(np.float32)

                    spot_data['gene_expression'] = gene_expression
                    spot_data['available_genes'] = [self.selected_genes[i]
                                                    for i, m in enumerate(present_mask) if m]
                else:
                    print(
                        f"Warning: spot {spot_idx} has no intersection genes")
                    spot_data['gene_expression'] = np.log1p(
                        np.zeros(len(self.selected_genes), dtype=np.float32))
                    spot_data['available_genes'] = []

            # Calculate final gene count and verify log transformation
            if self.all_spots_data:
                self.num_genes = len(self.selected_genes)
                self.common_genes = self.selected_genes
                print(f"Final gene count: {self.num_genes}")
                print(f"Expected gene count: {len(self.selected_genes)}")

                # Verify log transformation effect
                all_expressions = np.concatenate(
                    [spot['gene_expression'] for spot in self.all_spots_data])
                print(f"Gene expression after log1p transformation:")
                print(
                    f"  Range: [{all_expressions.min():.4f}, {all_expressions.max():.4f}]")
                print(
                    f"  Mean: {all_expressions.mean():.4f}, Std: {all_expressions.std():.4f}")
            else:
                self.num_genes = len(self.selected_genes)
                self.common_genes = self.selected_genes
        else:
            print("Error: Failed to load any sample data")
            self.num_genes = 0
            self.common_genes = []

    def load_graph_data(self):
        """Load graph data (plus_mlp style: use graph_dir, tolerate missing)."""
        print("Loading graph data (intra-spot only, tolerant to missing)...")

        if not GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required for intra-spot graph training but was not found.")

        if self.graph_dir is None:
            raise ValueError(
                "graph_dir must be provided for loading intra-spot graphs.")

        aggregated_intra_path = os.path.join(
            self.graph_dir, "hest_intra_spot_graphs.pkl")

        if not os.path.exists(aggregated_intra_path):
            raise FileNotFoundError(
                f"Intra-spot graphs file not found: {aggregated_intra_path}")

        try:
            with open(aggregated_intra_path, 'rb') as f:
                # {sample_id: {spot_idx: Data or dict}}
                aggregated_intra = pickle.load(f)
            if not isinstance(aggregated_intra, dict):
                raise ValueError(
                    "Loaded intra-spot graphs object must be a dict")
            print(
                f"Found aggregated intra-spot graphs: {aggregated_intra_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load aggregated intra graphs: {e}")

        # Store for lookup at __getitem__ time; do not enforce completeness
        self.intra_spot_graphs = aggregated_intra

        total_spots = len(self.all_spots_data)
        graphs_available = 0
        for spot_data in self.all_spots_data:
            sample_id = spot_data['sample_id']
            spot_idx = spot_data['spot_idx']
            if sample_id in self.intra_spot_graphs and isinstance(self.intra_spot_graphs[sample_id], dict) \
               and spot_idx in self.intra_spot_graphs[sample_id]:
                graphs_available += 1
        self.graphs_available = graphs_available
        print(
            f"Graph data loaded (intra only): {self.graphs_available}/{total_spots} spots available")

    def load_feature_data(self):
        """Load per-spot cell features/positions from combined_features.npz per sample.

        约定的数据格式（简化后）：
        - 每个样本一个NPZ：{sample_id}_combined_features.npz
        - 支持两种组织方式（二选一）：
          1) per_spot: dict[int spot_idx -> { 'x': ndarray[num_cells,D], 'pos': ndarray[num_cells,2] }]
          2) 扁平：features(ndarray[N,D]), positions(ndarray[N,2])，并配合 spot_ptr(M+1) 或 spot_index(N) 来分组至spot
        
        目标：当某个spot没有图时，使用该spot的细胞特征与坐标构建“无边图”（edge_index为空），
        在模型中自动跳过GNN，直接进入Transformer。
        """
        import numpy as np
        self.spot_features_map = {}

        if self.features_dir is None:
            print("[feature] features_dir not provided, skip loading features")
            return

        print("=== Loading per-spot cell features for missing-graph fallback (simplified) ===")
        for sample_id in self.sample_ids:
            npz_path = os.path.join(self.features_dir, f"{sample_id}_combined_features.npz")
            if not os.path.exists(npz_path):
                print(f"  - Warning: features file not found for sample {sample_id}: {npz_path}")
                continue
            try:
                npz = np.load(npz_path, allow_pickle=True)
                keys = set(npz.keys())

                # 1) per_spot 组织
                if 'per_spot' in keys:
                    per_spot = None
                    try:
                        per_spot = npz['per_spot'].item()
                    except Exception:
                        per_spot = None
                    if isinstance(per_spot, dict):
                        count = 0
                        for spot_idx, v in per_spot.items():
                            if isinstance(v, dict) and 'x' in v and 'pos' in v:
                                x = np.asarray(v['x'], dtype=np.float32)
                                pos = np.asarray(v['pos'], dtype=np.float32)
                                if len(x) == len(pos) and x.ndim == 2 and pos.ndim == 2 and pos.shape[1] == 2:
                                    self.spot_features_map[(sample_id, int(spot_idx))] = {'x': x, 'pos': pos}
                                    count += 1
                        print(f"  - {sample_id}: loaded per_spot entries: {count}")
                        continue

                # 2) 扁平组织：features + positions + spot_ptr 或 spot_index
                if 'features' in keys and 'positions' in keys:
                    feats = np.asarray(npz['features'], dtype=np.float32)
                    poss = np.asarray(npz['positions'], dtype=np.float32)
                    if feats.shape[0] != poss.shape[0]:
                        n_min = min(feats.shape[0], poss.shape[0])
                        print(f"  - Warning: feats({feats.shape[0]}) != positions({poss.shape[0]}), trunc -> {n_min}")
                        feats = feats[:n_min]
                        poss = poss[:n_min]

                    spot_ptr = npz['spot_ptr'] if 'spot_ptr' in keys else None
                    spot_index = npz['spot_index'] if 'spot_index' in keys else None

                    if spot_ptr is not None:
                        for si in range(len(spot_ptr) - 1):
                            s, e = int(spot_ptr[si]), int(spot_ptr[si+1])
                            if e > s:
                                self.spot_features_map[(sample_id, si)] = {
                                    'x': feats[s:e],
                                    'pos': poss[s:e]
                                }
                        print(f"  - {sample_id}: built spot features via spot_ptr")
                        continue
                    if spot_index is not None:
                        from collections import defaultdict
                        buf_x, buf_p = defaultdict(list), defaultdict(list)
                        for i in range(feats.shape[0]):
                            si = int(spot_index[i])
                            buf_x[si].append(feats[i])
                            buf_p[si].append(poss[i])
                        for si in buf_x:
                            x = np.stack(buf_x[si], axis=0).astype(np.float32)
                            p = np.stack(buf_p[si], axis=0).astype(np.float32)
                            self.spot_features_map[(sample_id, si)] = {'x': x, 'pos': p}
                        print(f"  - {sample_id}: built spot features via spot_index")
                        continue

                    print(f"  - Warning: {sample_id} lacks spot_ptr/spot_index; cannot group features")
                else:
                    print(f"  - Warning: {sample_id} missing 'features' or 'positions'")
            except Exception as e:
                print(f"  - Error loading features for {sample_id}: {e}")

    def setup_gene_normalization(self):
        """
        Prepare per-gene statistics for z-score normalization.
        """
        if not self.apply_gene_normalization:
            print("[normalization] Gene normalization disabled")
            return

        if self._provided_normalization_stats is not None:
            print("[normalization] Using provided gene normalization statistics")
            if not isinstance(self._provided_normalization_stats, dict):
                raise TypeError(
                    "normalization_stats must be a dict containing 'mean' and 'std'")
            mean = self._provided_normalization_stats.get('mean')
            std = self._provided_normalization_stats.get('std')
            if mean is None or std is None:
                raise ValueError(
                    "normalization_stats must include both 'mean' and 'std'")
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
        else:
            if self.mode != 'train':
                raise ValueError(
                    "Normalization stats must be provided when mode is not 'train'")
            if not self.all_spots_data:
                raise RuntimeError(
                    "Cannot compute normalization stats: dataset contains no spots")
            print("[normalization] Computing gene-wise mean/std from training data")
            stacked = np.stack(
                [spot['gene_expression'] for spot in self.all_spots_data], axis=0)
            mean = stacked.mean(axis=0).astype(np.float32)
            std = stacked.std(axis=0).astype(np.float32)

        std[std < self._normalization_eps] = 1.0

        self.gene_mean_np = mean
        self.gene_std_np = std
        self.gene_mean_tensor = torch.from_numpy(mean.copy())
        self.gene_std_tensor = torch.from_numpy(std.copy())
        self.normalization_stats = {
            'mean': mean.copy(),
            'std': std.copy()
        }
        print("[normalization] Gene normalization ready")

    def __len__(self):
        return len(self.all_spots_data)

    def __getitem__(self, idx):
        spot_data = self.all_spots_data[idx]

        # Get gene expression (target)
        spot_expressions = torch.FloatTensor(spot_data['gene_expression'])
        if self.apply_gene_normalization and self.gene_mean_tensor is not None and self.gene_std_tensor is not None:
            spot_expressions = (spot_expressions - self.gene_mean_tensor) / self.gene_std_tensor

        # Get graph data (lookup from aggregated dict; if missing, try build from raw features)
        graph_data = None
        if hasattr(self, 'intra_spot_graphs'):
            sample_graphs = self.intra_spot_graphs.get(spot_data['sample_id'])
            if isinstance(sample_graphs, dict):
                graph_data = sample_graphs.get(spot_data['spot_idx'])

        if not GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required but not available.")

        if graph_data is None:
            # 优先使用特征构造“无边图”，否则退化为旧的占位图
            built = False
            feat_key = (spot_data['sample_id'], spot_data['spot_idx'])
            feat_entry = getattr(self, 'spot_features_map', {}).get(feat_key)
            if feat_entry is not None:
                x_np = feat_entry.get('x')
                pos_np = feat_entry.get('pos')
                if x_np is not None and pos_np is not None and len(x_np) == len(pos_np) and len(x_np) > 0:
                    spot_graph = Data(
                        x=torch.as_tensor(x_np, dtype=torch.float32),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        pos=torch.as_tensor(pos_np, dtype=torch.float32)
                    )
                    built = True
            if not built:
                # 保底：单节点占位，避免训练崩溃
                spot_graph = Data(
                    x=torch.zeros(1, self.feature_dim, dtype=torch.float32),
                    edge_index=torch.empty((2, 0), dtype=torch.long)
                )
        elif isinstance(graph_data, Data):
            spot_graph = graph_data
        elif isinstance(graph_data, dict):
            x_key = 'x' if 'x' in graph_data else 'node_features'
            pos_key = 'pos' if 'pos' in graph_data else (
                'positions' if 'positions' in graph_data else None)
            if 'edge_index' not in graph_data or x_key not in graph_data:
                raise KeyError(
                    "Graph dict must contain 'edge_index' and 'x' or 'node_features'.")
            x_tensor = torch.as_tensor(graph_data[x_key], dtype=torch.float32)
            edge_index_tensor = torch.as_tensor(
                graph_data['edge_index'], dtype=torch.long)
            pos_tensor = None
            if pos_key is not None:
                pos_tensor = torch.as_tensor(
                    graph_data[pos_key], dtype=torch.float32)
            spot_graph = Data(
                x=x_tensor, edge_index=edge_index_tensor, pos=pos_tensor)
        else:
            raise TypeError(
                "Unsupported graph data type; expected PyG Data or dict.")

        return {
            'spot_expressions': spot_expressions,
            'spot_graphs': spot_graph,
            'sample_id': spot_data['sample_id'],
            'spot_id': spot_data['spot_id']
        }

    def get_normalization_stats(self):
        """
        Return copies of gene-wise normalization statistics.
        """
        if not self.apply_gene_normalization or self.gene_mean_np is None or self.gene_std_np is None:
            return None
        return {
            'mean': self.gene_mean_np.copy(),
            'std': self.gene_std_np.copy()
        }


def collate_fn_hest_graph(batch):
    """
    Custom collate function for HEST graph data
    """
    spot_expressions = torch.stack(
        [item['spot_expressions'] for item in batch])
    spot_graphs = [item['spot_graphs'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    spot_ids = [item['spot_id'] for item in batch]

    return {
        'spot_expressions': spot_expressions,
        'spot_graphs': spot_graphs,
        'sample_ids': sample_ids,
        'spot_ids': spot_ids
    }

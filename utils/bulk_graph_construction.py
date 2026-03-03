#!/usr/bin/env python3
"""
Bulk static graph construction for BiTro (patch-based).

This script builds static graph data structures from:
- Per-cell features stored in Parquet files (train/test splits)
- Pre-segmented patch tiles for each slide/patient

Feature file layout:
- ``train/*.parquet``: training feature files
- ``test/*.parquet``: test feature files
- Each file contains 128-D DINO features and per-cell spatial coordinates

Patch file layout:
- ``patches_dir/{patient_id}/*.png``: pre-segmented patch tiles
- Filename pattern:
  ``{slide_id}_patch_tile_{tile_id}_level0_{x1}-{y1}-{x2}-{y2}.png``
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
    """Build patch-level intra-graphs and an inter-patch kNN graph."""
    
    def __init__(self, 
                 train_features_dir,
                 test_features_dir,
                 bulk_csv_path,
                 patches_dir,
                 wsi_input_dir,
                 intra_patch_distance_threshold=250,
                 inter_patch_k_neighbors=6,
                 use_deep_features=True,
                 feature_dim=128,
                 max_cells_per_patch=None,
                 max_train_slides=None,
                 max_test_slides=None,
                 checkpoint_dir=None):
        
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
        
        # Optional checkpoint directory for resuming long runs.
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(self.checkpoint_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.checkpoint_dir, 'test'), exist_ok=True)
        
    def load_bulk_data(self):
        """Load bulk RNA-seq expression table and build case-to-columns mapping."""
        print("=== Loading bulk RNA-seq table ===")
        
        bulk_df = pd.read_csv(self.bulk_csv_path)
        bulk_df["gene_name"] = bulk_df['Unnamed: 0'].str[:15]
        bulk_df = bulk_df.drop(columns=['Unnamed: 0'])
        bulk_df = bulk_df.set_index('gene_name')
        
        original_ids = list(bulk_df.columns)
        case_ids = [pid[:12] for pid in original_ids]  # TCGA case id (e.g., TCGA-3C-AALI)
        case_id_series = pd.Series(case_ids)
        
        # Map case id -> all original columns (later averaged if multiple).
        case_to_cols = {}
        for case_id, original_id in zip(case_ids, original_ids):
            case_to_cols.setdefault(case_id, []).append(original_id)
        
        multi_col_cases = sum(1 for cols in case_to_cols.values() if len(cols) > 1)
        print(f"Cases: {len(case_to_cols)} (multi-column cases: {multi_col_cases}; will average columns)")
        
        self.bulk_data = bulk_df
        self.case_to_bulk_cols = case_to_cols
        self.valid_patient_ids = list(case_to_cols.keys())
        
        print(f"Bulk table shape: {bulk_df.shape}")
        print(f"Valid case IDs: {len(self.valid_patient_ids)}")
        
    def extract_slide_id(self, file_path):
        """Extract a slide identifier from a path or filename.

        This supports multiple naming conventions used across feature and patch
        files and preserves the UUID component when present.
        """
        basename = os.path.basename(file_path)
        # Supported examples:
        # 1. TCGA-AA-3872-01A-01-TS1.4f7d5598-e36a-4e30-9b7b-ab55cc6fc3a0_tile36_features.parquet
        # 2. TCGA-A2-A0YI-01A-03-TSC.315f5bb4-4ef4-471e-b5b4-ae73a6038c20_features.parquet
        # 3. TCGA-AA-3872-01A-01-BS1.e29045b5-113d-4dba-b03b-ba2e0d82a388_patch_tile_542_level0_5540-10952-5796-11208.png
        if '_tile36_features.parquet' in basename:
            return basename.replace('_tile36_features.parquet', '')
        elif '_features.parquet' in basename:
            # Newer format: ``..._features.parquet``.
            return basename.replace('_features.parquet', '')
        elif '_patch_tile_' in basename:
            # Patch filename format: extract the prefix before ``_patch_tile_``.
            parts = basename.split('_patch_tile_')
            if len(parts) >= 2:
                return parts[0]
        return basename
    
    def extract_patient_id_from_slide(self, slide_id):
        """Extract a TCGA case identifier from a slide identifier."""
        # Example:
        #   slide_id: TCGA-AA-3872-01A-01-TS1.<uuid>
        #   case_id:  TCGA-AA-3872
        parts = slide_id.split('-')
        if len(parts) >= 3:
            return '-'.join(parts[:3])
        return slide_id[:12]
        
    def find_patch_files_by_slide(self, slide_id):
        """Find patch PNG files that correspond to a given slide_id."""
        matching_patch_files = []
        
        # Search recursively under the patches directory.
        for root, dirs, files in os.walk(self.patches_dir):
            for file in files:
                if file.endswith(".png") and "_patch_tile_" in file:
                    file_slide_id = self.extract_slide_id(file)
                    if slide_id == file_slide_id:
                        matching_patch_files.append(os.path.join(root, file))
        
        print(f"  - Slide {slide_id}: found {len(matching_patch_files)} matching patch files")
        return matching_patch_files
    
    def parse_patch_coordinates(self, patch_filename):
        """Parse tile id and coordinates from a patch filename.

        Expected pattern:
            ``*_patch_tile_{tile_id}_level0_{x1}-{y1}-{x2}-{y2}.png``
        """
        basename = os.path.basename(patch_filename)
        
        # Use regex to extract coordinates.
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
            print(f"Warning: unable to parse patch filename: {basename}")
            return None
    
    def convert_to_absolute_coordinates(self, df):
        """Convert per-tile relative cell coordinates into WSI absolute coordinates."""
        def parse_tile_coordinates(image_name):
            """Extract absolute tile coordinates from an image_name string."""
            pattern = r'_patch_tile_\d+_level0_(\d+)-(\d+)-(\d+)-(\d+)$'
            match = re.search(pattern, image_name)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                return x1, y1, x2, y2
            return None, None, None, None
        
        # Parse the absolute origin of each tile.
        tile_coords = df['image_name'].apply(parse_tile_coordinates)
        
        # Expand into separate columns.
        df[['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2']] = pd.DataFrame(tile_coords.tolist(), index=df.index)
        
        # Absolute coordinates = tile origin + relative coordinates.
        df['abs_x'] = df['tile_x1'] + df['x']
        df['abs_y'] = df['tile_y1'] + df['y']
        
        # Overwrite x/y with absolute coordinates.
        df['x'] = df['abs_x']
        df['y'] = df['abs_y']
        
        # Drop temporary columns.
        df = df.drop(columns=['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2', 'abs_x', 'abs_y', 'image_name'])
        
        return df
    
    def assign_cells_to_patches(self, cells_df, patch_files):
        """Assign cells to patch tiles based on spatial bounding boxes."""
        print(f"  - Assigning {len(cells_df)} cells to {len(patch_files)} patches")
        
        patches = []
        cells_df = cells_df.copy()
        cells_df['patch_id'] = -1
        
        for patch_file in patch_files:
            patch_coords = self.parse_patch_coordinates(patch_file)
            if patch_coords is None:
                continue
            
            # Select cells within the patch bounds.
            patch_mask = ((cells_df['x'] >= patch_coords['x1']) & 
                         (cells_df['x'] < patch_coords['x2']) & 
                         (cells_df['y'] >= patch_coords['y1']) & 
                         (cells_df['y'] < patch_coords['y2']))
            
            patch_cells = cells_df[patch_mask].copy()
            
            if len(patch_cells) > 0:
                patch_id = patch_coords['tile_id']
                cells_df.loc[patch_mask, 'patch_id'] = patch_id
                
                # Convert absolute coordinates into patch-relative coordinates.
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
        print(f"  - Assigned {assigned_count}/{len(cells_df)} cells to {len(patches)} valid patches")
        
        return patches
    
    def build_single_patch_graph(self, patch_cells, patch_id):
        """Build a single patch-level cell graph (streaming-friendly)."""
        if len(patch_cells) < 1:
            return None
        
        if len(patch_cells) == 1:
            # Patch containing a single cell.
            cell_row = patch_cells.iloc[0]
            cell_features = self.extract_cell_feature_vector(cell_row)
            x = torch.tensor([cell_features], dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            pos = torch.tensor([[cell_row['x'], cell_row['y']]], dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, pos=pos)
        
        # Extract positions and features (patch-relative coordinates).
        positions = patch_cells[['x', 'y']].values
        cell_features = np.array([
            self.extract_cell_feature_vector(row) 
            for _, row in patch_cells.iterrows()
        ])
        
        # Pairwise distance matrix.
        distances = squareform(pdist(positions))
        
        # Adjacency by distance threshold.
        adj_matrix = (distances <= self.intra_patch_distance_threshold) & (distances > 0)
        
        # Convert to edge list.
        edge_indices = np.where(adj_matrix)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        # If no edges are found, fall back to a small kNN graph.
        if edge_index.shape[1] == 0:
            k = min(3, len(patch_cells) - 1)
            if k > 0:
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(positions)
                _, indices = nbrs.kneighbors(positions)
                
                edges = []
                for i, neighbors in enumerate(indices):
                    for neighbor in neighbors[1:]:
                        edges.extend([[i, neighbor], [neighbor, i]])
                
                if edges:
                    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Build graph object.
        x = torch.tensor(cell_features, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, pos=pos)
    
    def build_intra_patch_graphs(self, patches):
        """Build intra-patch graphs (kept for compatibility)."""
        intra_patch_graphs = {}
        
        for patch_info in tqdm(patches, desc="Building intra-patch graphs"):
            patch_id = patch_info['patch_id']
            patch_cells = patch_info['cells']
            
            graph = self.build_single_patch_graph(patch_cells, patch_id)
            if graph is not None:
                intra_patch_graphs[patch_id] = graph
        
        return intra_patch_graphs
    
    def build_inter_patch_graph(self, patches):
        """Build an inter-patch kNN graph using patch centers."""
        if len(patches) < 2:
            # One patch only.
            if len(patches) == 1:
                patch_center = patches[0]['center']
                patch_features = torch.tensor([patch_center], dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                pos = torch.tensor([patch_center], dtype=torch.float32)
                return Data(x=patch_features, edge_index=edge_index, pos=pos)
            else:
                # No patches.
                return Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        
        # Patch centers in WSI coordinates.
        patch_positions = np.array([patch['center'] for patch in patches])
        
        # kNN connectivity between patches.
        k = min(self.inter_patch_k_neighbors, len(patches) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(patch_positions)
        _, indices = nbrs.kneighbors(patch_positions)
        
        # Build undirected edge list.
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:
                edges.extend([[i, neighbor], [neighbor, i]])
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
        
        # Use positions as patch node features.
        patch_features = torch.tensor(patch_positions, dtype=torch.float32)
        pos = torch.tensor(patch_positions, dtype=torch.float32)
        
        inter_patch_graph = Data(x=patch_features, edge_index=edge_index, pos=pos)
        
        return inter_patch_graph
    
    def build_inter_patch_graph_from_centers(self, patch_centers):
        """Build an inter-patch kNN graph from patch centers (streaming version)."""
        if len(patch_centers) < 2:
            # One patch only.
            if len(patch_centers) == 1:
                patch_center = patch_centers[0]
                patch_features = torch.tensor([patch_center], dtype=torch.float32)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                pos = torch.tensor([patch_center], dtype=torch.float32)
                return Data(x=patch_features, edge_index=edge_index, pos=pos)
            else:
                # No patches.
                return Data(x=torch.empty((0, 2)), edge_index=torch.empty((2, 0)), pos=torch.empty((0, 2)))
        
        # Patch centers.
        patch_positions = np.array(patch_centers)
        
        # kNN connectivity.
        k = min(self.inter_patch_k_neighbors, len(patch_centers) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(patch_positions)
        _, indices = nbrs.kneighbors(patch_positions)
        
        # Build undirected edge list.
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:
                edges.extend([[i, neighbor], [neighbor, i]])
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
        
        # Use positions as patch node features.
        patch_features = torch.tensor(patch_positions, dtype=torch.float32)
        pos = torch.tensor(patch_positions, dtype=torch.float32)
        
        inter_patch_graph = Data(x=patch_features, edge_index=edge_index, pos=pos)
        
        return inter_patch_graph
    
    def extract_cell_feature_vector(self, cell_row):
        """Extract a cell feature vector."""
        if self.use_deep_features:
            # Use deep features.
            features = [cell_row[f'feature_{i}'] for i in range(self.feature_dim)]
            return np.array(features, dtype=np.float32)
        else:
            # Fallback: simple geometric features.
            features = [
                cell_row['x'],
                cell_row['y'],
                cell_row.get('area', 100.0),
                cell_row.get('perimeter', 35.4),
            ]
            # Pad/truncate to feature_dim.
            features = np.array(features, dtype=np.float32)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant')
            return features[:self.feature_dim]
    
    def get_feature_file_list(self, split='train'):
        """List feature Parquet files for a split (without loading data)."""
        # Select directory by split.
        if split == 'train':
            features_dir = self.train_features_dir
        else:
            features_dir = self.test_features_dir

        # Discover parquet files.
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
            print(f"Found {original_count} feature files for split '{split}'; using first {len(feature_files)}")
        else:
            print(f"Found {len(feature_files)} feature files for split '{split}' (using all)")

        # Record selected filenames for reproducibility.
        self.selected_feature_files[split] = [os.path.basename(p) for p in feature_files]

        return feature_files

    def get_checkpoint_file_list(self, split='train'):
        """List checkpoint files (filenames only, without loading)."""
        if not self.checkpoint_dir:
            return []

        checkpoint_dir = os.path.join(self.checkpoint_dir, split)
        if not os.path.exists(checkpoint_dir):
            return []

        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        return checkpoint_files
    
    def save_slide_checkpoint(self, slide_id, slide_data, split='train'):
        """Save a per-slide checkpoint."""
        if not self.checkpoint_dir:
            return
        
        # Use a filesystem-safe filename.
        safe_slide_id = slide_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        checkpoint_path = os.path.join(self.checkpoint_dir, split, f"{safe_slide_id}.pkl")
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(slide_data, f)
        except Exception as e:
            print(f"Warning: unable to save checkpoint {checkpoint_path}: {e}")
    
    def load_slide_checkpoint(self, slide_id, split='train'):
        """Load a per-slide checkpoint."""
        if not self.checkpoint_dir:
            return None
        
        safe_slide_id = slide_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        checkpoint_path = os.path.join(self.checkpoint_dir, split, f"{safe_slide_id}.pkl")
        
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: unable to load checkpoint {checkpoint_path}: {e}")
                return None
        return None
    
    def load_all_checkpoints(self, split='train'):
        """Load all saved checkpoints (split-level)."""
        if not self.checkpoint_dir:
            return {}
        
        checkpoint_dir = os.path.join(self.checkpoint_dir, split)
        if not os.path.exists(checkpoint_dir):
            return {}
        
        checkpoints = {}
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
        
        print(f"  Found {len(checkpoint_files)} saved checkpoints for split '{split}'")
        
        for checkpoint_file in tqdm(checkpoint_files, desc=f"Loading checkpoints ({split})"):
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            try:
                with open(checkpoint_path, 'rb') as f:
                    slide_data = pickle.load(f)
                    slide_id = slide_data.get('slide_id', checkpoint_file.replace('.pkl', ''))
                    checkpoints[slide_id] = slide_data
            except Exception as e:
                print(f"Warning: unable to load checkpoint {checkpoint_path}: {e}")
        
        return checkpoints
    
    def is_slide_processed(self, slide_id, split='train'):
        """Return True if a slide checkpoint already exists."""
        if not self.checkpoint_dir:
            return False
        
        safe_slide_id = slide_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        checkpoint_path = os.path.join(self.checkpoint_dir, split, f"{safe_slide_id}.pkl")
        return os.path.exists(checkpoint_path)
    
    def process_all_slides_new_logic(self):
        """Process all slides using slide-level patch matching (resume-friendly).

        Slides are processed one-by-one to reduce peak memory usage. When a
        checkpoint directory is configured, per-slide results are saved
        immediately to enable resuming.
        """
        print("=== Processing all data (slide-level matching, resume-friendly) ===")
        
        # When checkpointing is enabled, initialize bookkeeping without preloading data.
        if self.checkpoint_dir:
            print("\n=== 初始化检查点状态（按需加载，节省内存）===")
            # Track processed slide IDs only (do not preload checkpoints).
            self.processed_data['train'] = {}
            self.processed_data['test'] = {}

            # Count processed slides.
            train_checkpoints = self.get_checkpoint_file_list('train')
            test_checkpoints = self.get_checkpoint_file_list('test')
            print(f"  发现训练集已处理切片: {len(train_checkpoints)}")
            print(f"  发现测试集已处理切片: {len(test_checkpoints)}")
        
        # List feature files (without loading them).
        train_feature_files = self.get_feature_file_list('train')
        test_feature_files = self.get_feature_file_list('test')
        
        # Train split: load and process slides one by one.
        print("\n处理训练集...")
        if 'train' not in self.processed_data:
            self.processed_data['train'] = {}
        
        processed_count = 0
        skipped_count = 0
        
        for feature_file in tqdm(train_feature_files, desc="Processing train slides"):
            # Extract slide id (including UUID when present).
            slide_id = self.extract_slide_id(feature_file)
            
            # Skip if already processed (resume).
            if slide_id in self.processed_data['train'] or self.is_slide_processed(slide_id, 'train'):
                skipped_count += 1
                continue
            
            # Validate patient/case id.
            patient_id = self.extract_patient_id_from_slide(slide_id)
            if patient_id not in self.valid_patient_ids:
                continue
            
            # Load and process a single slide.
            try:
                df = pd.read_parquet(feature_file)
                
                # Validate required columns.
                required_columns = [f'feature_{i}' for i in range(128)] + ['x', 'y', 'image_name', 'cluster_label']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"警告: 文件 {feature_file} 缺少列: {missing_cols}")
                    continue
                
                # Convert to absolute WSI coordinates.
                df_processed = self.convert_to_absolute_coordinates(df.copy())
                
                # Process immediately.
                result = self.process_single_slide_new_logic(df_processed, slide_id, patient_id)
                if result:
                    self.processed_data['train'][slide_id] = result
                    # Save checkpoint immediately.
                    self.save_slide_checkpoint(slide_id, result, 'train')
                    processed_count += 1
                
                # Free memory.
                del df, df_processed, result
                
            except Exception as e:
                print(f"错误: 无法处理 {feature_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if skipped_count > 0:
            print(f"  跳过已处理的切片: {skipped_count} 个")
        print(f"  新处理切片: {processed_count} 个")
        
        # Test split: load and process slides one by one.
        print("\n处理测试集...")
        if 'test' not in self.processed_data:
            self.processed_data['test'] = {}
        
        processed_count = 0
        skipped_count = 0
        
        for feature_file in tqdm(test_feature_files, desc="Processing test slides"):
            # Extract slide id (including UUID when present).
            slide_id = self.extract_slide_id(feature_file)
            
            # Skip if already processed (resume).
            if slide_id in self.processed_data['test'] or self.is_slide_processed(slide_id, 'test'):
                skipped_count += 1
                continue
            
            # Validate patient/case id.
            patient_id = self.extract_patient_id_from_slide(slide_id)
            if patient_id not in self.valid_patient_ids:
                continue
            
            # Load and process a single slide.
            try:
                df = pd.read_parquet(feature_file)
                
                # Validate required columns.
                required_columns = [f'feature_{i}' for i in range(128)] + ['x', 'y', 'image_name', 'cluster_label']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"警告: 文件 {feature_file} 缺少列: {missing_cols}")
                    continue
                
                # Convert to absolute WSI coordinates.
                df_processed = self.convert_to_absolute_coordinates(df.copy())
                
                # Process immediately.
                result = self.process_single_slide_new_logic(df_processed, slide_id, patient_id)
                if result:
                    self.processed_data['test'][slide_id] = result
                    # Save checkpoint immediately.
                    self.save_slide_checkpoint(slide_id, result, 'test')
                    processed_count += 1
                
                # Free memory.
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
        
        # Graph availability summary.
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
        """Process a single slide while preserving all cells (new logic)."""
        print(f"处理切片: {slide_id} (患者: {patient_id})")
        
        if cells_df is None or len(cells_df) == 0:
            print(f"  - 警告: 切片 {slide_id} 没有细胞数据")
            return None
            
        print(f"  - 细胞数量: {len(cells_df)}")
        
        # Extract full-cell features, coordinates, and cluster labels.
        all_cell_features = self.extract_all_cell_features_with_clusters(cells_df)
        all_cell_positions = cells_df[['x', 'y']].values.astype(np.float32)
        cluster_labels = cells_df['cluster_label'].values
        
        # Find matching patch files for this slide.
        patch_files = self.find_patch_files_by_slide(slide_id)
        print(f"  - 匹配的Patch文件数量: {len(patch_files)}")
        
        has_graphs = False
        intra_patch_graphs = {}
        patches = []
        
        if len(patch_files) > 0:
            # Assign cells to patches. assign_cells_to_patches operates on a copy,
            # but we also want to retain patch_id on the returned cells_df.
            cells_df_with_patch = cells_df.copy()
            patches = self.assign_cells_to_patches(cells_df_with_patch, patch_files)
            
            # Copy patch_id back to the original table.
            if 'patch_id' in cells_df_with_patch.columns:
                cells_df['patch_id'] = cells_df_with_patch['patch_id'].values
            else:
                cells_df['patch_id'] = -1
            
            if len(patches) > 0:
                # Build intra-patch graphs.
                intra_patch_graphs = self.build_intra_patch_graphs(patches)
                
                # Build inter-patch graph.
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
            # Note: bulk expression is fetched by patient/case id.
            'bulk_expr': self.get_bulk_expression(patient_id),
            'has_graphs': has_graphs,
            'all_cell_features': all_cell_features,
            'all_cell_positions': torch.tensor(all_cell_positions),
            'cluster_labels': torch.tensor(cluster_labels),
            'cell_to_graph_mapping': self.build_cell_to_graph_mapping(cells_df, patches) if has_graphs else None
        }
    
    def extract_all_cell_features_with_clusters(self, cells_df):
        """Extract per-cell DINO features for all cells in a slide."""
        if cells_df is None or len(cells_df) == 0:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)
        
        # DINO features live in feature_0 ... feature_127.
        feature_columns = [f'feature_{i}' for i in range(128)]
        features_matrix = cells_df[feature_columns].values.astype(np.float32)
        
        return torch.tensor(features_matrix, dtype=torch.float32)
    
    def build_cell_to_graph_mapping(self, cells_df, patches):
        """Build a mapping from cell indices to their patch graph id."""
        if not patches:
            return None
            
        cell_to_graph = {}
        
        for patch_info in patches:
            patch_id = patch_info['patch_id']
            patch_cells = patch_info['cells']
            
            # Add mappings for each cell within the patch.
            for cell_idx in patch_cells.index:
                if cell_idx in cells_df.index:
                    cell_to_graph[cell_idx] = {
                        'patch_id': patch_id,
                        'has_graph': True
                    }
        
        return cell_to_graph
    
    def get_bulk_expression(self, patient_id):
        """Get bulk expression for a patient/case id."""
        if self.bulk_data is None:
            return None
            
        # Find matching columns (case-level).
        bulk_cols = self.case_to_bulk_cols.get(patient_id, [])
        if not bulk_cols:
            print(f"警告: 未找到病例 {patient_id} 的bulk数据")
            return None
        
        bulk_values = self.bulk_data[bulk_cols].values.astype(np.float32)
        if bulk_values.ndim == 2 and bulk_values.shape[1] > 1:
            # Multiple columns: average across columns.
            print(f"信息: 病例 {patient_id} 有 {bulk_values.shape[1]} 列bulk数据，使用平均值")
            print(f"  - 可用列: {bulk_cols}")
            return np.mean(bulk_values, axis=1)
        
        # Single column: return as (N,).
        return bulk_values.reshape(-1)
    
    def save_selected_feature_filenames(self, output_dir):
        """Save selected train/test feature filenames."""
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
        """Save constructed graphs using per-slide checkpoints to reduce memory usage."""
        print("=== 保存图结构和完整细胞数据（切片级别，从检查点加载）===")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save train/test splits separately.
        for split in ['train', 'test']:
            # List checkpoint files.
            checkpoint_files = self.get_checkpoint_file_list(split)
            if not checkpoint_files:
                print(f"{split}集没有找到检查点文件，跳过")
                continue

            print(f"{split}集发现 {len(checkpoint_files)} 个检查点文件，开始分批加载并保存")

            # Process in batches to reduce peak memory usage.
            batch_size = 5
            total_batches = (len(checkpoint_files) + batch_size - 1) // batch_size

            # Accumulators for serialization.
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

                # Load checkpoints for this batch.
                for checkpoint_file in batch_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                    try:
                        with open(checkpoint_path, 'rb') as f:
                            slide_data = pickle.load(f)
                            slide_id = slide_data.get('slide_id', checkpoint_file.replace('.pkl', ''))

                            # Accumulate data.
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

                # After each batch (except the last), save partial results to avoid
                # accumulating too much in memory.
                if batch_idx < total_batches - 1:  # Save partial results for non-final batches.
                    print(f"    第 {batch_idx + 1} 批处理完成，保存中间结果...")
                    self._save_split_data_partial(split, intra_graphs, inter_graphs, bulk_expressions,
                                                all_cell_features, all_cell_positions, cluster_labels,
                                                graph_status, cell_to_graph_mappings, slide_to_patient_mapping,
                                                metadata, output_dir, batch_idx + 1)

                    # Clear accumulators to free memory.
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
            
            # Save final results.
            self._save_split_data_final(split, intra_graphs, inter_graphs, bulk_expressions,
                                      all_cell_features, all_cell_positions, cluster_labels,
                                      graph_status, cell_to_graph_mappings, slide_to_patient_mapping,
                                      metadata, output_dir)
            
            # Summary statistics.
            total_slides = len(metadata)
            slides_with_graphs = sum([status for status in graph_status.values()])
            slides_without_graphs = total_slides - slides_with_graphs
            unique_patients = len(set(slide_to_patient_mapping.values()))

            # Output paths (must match _save_split_data_final).
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
            print(f"  - 切片映射: {slide_mappings_path}")  # slide -> patient mapping
            print(f"  - 元数据: {metadata_path}")
        
        # Save global config.
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
            'uses_slide_level_matching': True,  # slide-level patch matching
            'allows_multiple_slides_per_patient': True,  # allow multiple slides per patient
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
        """Save partial split data for memory-friendly processing."""
        temp_dir = os.path.join(output_dir, f"temp_{split}_batch_{batch_idx}")
        os.makedirs(temp_dir, exist_ok=True)

        # Write into a temporary directory.
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
        """Save final split data (merging all batches via checkpoints)."""
        print(f"  合并并保存{split}集最终结果...")

        # Check whether temporary batch folders exist.
        temp_pattern = f"temp_{split}_batch_*"
        import glob
        temp_dirs = glob.glob(os.path.join(output_dir, temp_pattern))

        if temp_dirs:
            print(f"  发现 {len(temp_dirs)} 个临时批次，需要合并")
            # Reload full data from per-slide checkpoints.
            print(f"  重新从检查点加载{split}集的完整数据...")
            checkpoint_dir_path = os.path.join(self.checkpoint_dir, split)
            all_checkpoints = self.load_all_checkpoints(split)

            # Rebuild accumulators from loaded checkpoints.
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

        # Final output paths.
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

        print(f"  开始保存{split}集数据文件...")

        print(f"  保存Patch内图...")
        with open(intra_path, 'wb') as f:
            pickle.dump(intra_graphs, f)
        print(f"  ✓ Patch内图保存完成")

        print(f"  保存Patch间图...")
        with open(inter_path, 'wb') as f:
            pickle.dump(inter_graphs, f)
        print(f"  ✓ Patch间图保存完成")

        print(f"  保存Bulk表达...")
        with open(bulk_path, 'wb') as f:
            pickle.dump(bulk_expressions, f)
        print(f"  ✓ Bulk表达保存完成")

        print(f"  保存细胞特征...")
        with open(features_path, 'wb') as f:
            pickle.dump(all_cell_features, f)
        print(f"  ✓ 细胞特征保存完成")

        print(f"  保存细胞坐标...")
        with open(positions_path, 'wb') as f:
            pickle.dump(all_cell_positions, f)
        print(f"  ✓ 细胞坐标保存完成")

        print(f"  保存聚类标签...")
        with open(clusters_path, 'wb') as f:
            pickle.dump(cluster_labels, f)
        print(f"  ✓ 聚类标签保存完成")

        print(f"  保存图状态...")
        with open(status_path, 'wb') as f:
            pickle.dump(graph_status, f)
        print(f"  ✓ 图状态保存完成")

        print(f"  保存细胞映射...")
        with open(mappings_path, 'wb') as f:
            pickle.dump(cell_to_graph_mappings, f)
        print(f"  ✓ 细胞映射保存完成")

        print(f"  保存切片映射...")
        with open(slide_mappings_path, 'wb') as f:
            pickle.dump(slide_to_patient_mapping, f)
        print(f"  ✓ 切片映射保存完成")

        print(f"  保存元数据...")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ 元数据保存完成")
        print(f"  {split}集所有数据保存完毕！")

        # Summary statistics.
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
    """Entry point for building bulk graphs."""
    
    # Configuration.
    train_features_dir = "/mnt/elements/ouput_features/PRAD_train"
    test_features_dir = "/mnt/elements/ouput_features/PRAD_test"
    bulk_csv_path = "/mnt/elements/PRAD/tpm-TCGA-PRAD.csv"
    patches_dir = "/mnt/elements/PRAD/PRAD"
    wsi_input_dir = "/mnt/elements/PRAD/PRAD_svs"
    output_dir = "/mnt/elements/PRAD/bulk_PRAD_graphs_new_all_graph"
    checkpoint_dir = "/mnt/elements/PRAD/bulk_PRAD_graphs_checkpoints"  # checkpoints for resuming
    
    # Graph parameters.
    intra_patch_distance_threshold = 256   # intra-patch distance threshold (pixels)
    inter_patch_k_neighbors = 8            # inter-patch kNN
    use_deep_features = True               # use deep features
    feature_dim = 128                      # feature dimension
    max_cells_per_patch = None             # None = no limit
    max_train_slides = 200                 # None = all
    max_test_slides = 50                   # None = all
    
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
    
    # Validate input paths.
    for path, name in [(train_features_dir, "训练特征目录"), (test_features_dir, "测试特征目录"), 
                       (bulk_csv_path, "Bulk数据文件"), (patches_dir, "Patch目录"), (wsi_input_dir, "WSI输入目录")]:
        if not os.path.exists(path):
            print(f"错误: {name}不存在: {path}")
            return
    
    # Build graphs.
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
        
        # Load bulk expression.
        builder.load_bulk_data()
        
        # Process all slides.
        builder.process_all_slides_new_logic()
        
        # Save graphs from checkpoints.
        metadata = builder.save_graphs_slide_logic(output_dir)
        builder.save_selected_feature_filenames(output_dir)
        
        print("\n=== 图构建完成（切片级别匹配，0%数据丢失版本）===")
        for split in ['train', 'test']:
            total_slides = len(builder.processed_data.get(split, {}))
            print(f"{split}集:")
            print(f"  - 切片数: {total_slides}")
            if total_slides > 0:
                # Compute simple stats from processed_data.
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
#!/usr/bin/env python3
"""
Graph construction for the HEST spatial transcriptomics dataset (BiTro).

This script builds a two-level graph representation from the HEST on-disk
format without requiring the HEST Python API:
1) Intra-spot graphs: cell-level graphs within each spot (from CellViT
   segmentation / per-cell features).
2) Inter-spot graph: spot-level kNN graph based on spot spatial coordinates.

Expected input layout under ``hest_data_dir``:
- ``st/*.h5ad``: AnnData files containing spot-level expression + coordinates
- ``cellvit_seg/*.parquet``: CellViT segmentation outputs (optional)
- ``metadata/*.json``: sample metadata (optional)
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

# Reduce scanpy verbosity to avoid excessive console output.
sc.settings.verbosity = 1


class HESTDirectReader:
    """Build HEST intra-/inter-spot graphs from files on disk."""

    def __init__(self,
                 hest_data_dir,
                 sample_ids,
                 features_dir=None,
                 inter_spot_k_neighbors=6):

        self.hest_data_dir = hest_data_dir
        self.sample_ids = sample_ids if isinstance(
            sample_ids, list) else [sample_ids]
        self.features_dir = features_dir
        self.inter_spot_k_neighbors = inter_spot_k_neighbors

        self.sample_data = {}
        self.processed_data = {}
        self.deep_features = {}

    def load_sample_data(self):
        """Load AnnData, metadata, and optional segmentation/features for samples."""
        print("=== Loading HEST data files directly ===")

        for sample_id in self.sample_ids:
            try:
                print(f"Loading sample: {sample_id}")

                sample_info = {}

                # 1) AnnData
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

                # 2) Metadata
                metadata_file = os.path.join(
                    self.hest_data_dir, "metadata", f"{sample_id}.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    sample_info['metadata'] = metadata
                    print(f"  - Metadata: {metadata.get('tissue', 'unknown')} tissue")
                else:
                    sample_info['metadata'] = {}

                # 3) Cell segmentation (optional)
                cellvit_file = os.path.join(
                    self.hest_data_dir, "cellvit_seg", f"{sample_id}_cellvit_seg.parquet")
                if os.path.exists(cellvit_file):
                    cellvit_df = pd.read_parquet(cellvit_file)
                    sample_info['cellvit'] = cellvit_df
                    print(f"  - Cell segmentation: {len(cellvit_df)} cells")
                else:
                    print(f"  - Warning: cell segmentation file not found: {cellvit_file}")
                    sample_info['cellvit'] = None

                # 4) Image patches (optional)
                patches_file = os.path.join(
                    self.hest_data_dir, "patches", f"{sample_id}.h5")
                if os.path.exists(patches_file):
                    sample_info['patches_file'] = patches_file
                    print(f"  - Image patches: available")
                else:
                    sample_info['patches_file'] = None
                    print(f"  - Warning: patches file not found")

                self.sample_data[sample_id] = sample_info

            except Exception as e:
                print(f"  Error: Loading sample {sample_id}: {e}")
                continue

        print(f"Successfully loaded {len(self.sample_data)} samples")

        # Optional deep features.
        if self.features_dir:
            self.load_deep_features()

    def load_deep_features(self):
        """Load per-cell deep feature NPZ files for each sample (optional)."""
        print("=== Loading deep feature files ===")

        for sample_id in self.sample_ids:
            try:
                # Locate the sample's feature file.
                feature_file = os.path.join(
                    self.features_dir, f"{sample_id}_combined_features.npz")

                if os.path.exists(feature_file):
                    print(f"Loading sample {sample_id} deep features...")

                    # Use allow_pickle=True because metadata may store objects/dicts.
                    data = np.load(feature_file, allow_pickle=True)
                    features = data['features'].astype(
                        np.float32)
                    positions = data['positions'].astype(
                        np.float32) if 'positions' in data else None
                    cell_index = data['cell_index'] if 'cell_index' in data else None
                    spot_ptr = data['spot_ptr'] if 'spot_ptr' in data else None
                    spot_index = data['spot_index'] if 'spot_index' in data else None

                    # Best-effort metadata parsing.
                    try:
                        if 'metadata' in data:
                            metadata_array = data['metadata']
                            # Handle 0-d numpy arrays that wrap dict-like metadata.
                            if metadata_array.ndim == 0:
                                metadata = metadata_array.item() if metadata_array.dtype == object else {}
                            else:
                                metadata = metadata_array if isinstance(
                                    metadata_array, dict) else {}
                        else:
                            metadata = {}
                    except Exception as meta_e:
                        print(f"    Warning: failed to parse metadata: {meta_e}")
                        metadata = {}

                    # Align lengths if features/positions mismatch.
                    n_feat = features.shape[0]
                    if positions is not None:
                        n_pos = positions.shape[0]
                        if n_pos != n_feat:
                            n_min = min(n_feat, n_pos)
                            print(
                                f"    Warning: features and positions have different lengths: feats={n_feat}, pos={n_pos}; trimming both to {n_min}")
                            features = features[:n_min]
                            positions = positions[:n_min]
                            n_feat = n_min

                    # Optional clustering labels written by the clustering script.
                    cluster_labels = None
                    try:
                        if 'cluster_labels' in data:
                            cluster_labels = data['cluster_labels']
                            # Type/shape sanity checks.
                            if getattr(cluster_labels, 'ndim', 1) != 1 or cluster_labels.shape[0] != features.shape[0]:
                                print("    Warning: cluster_labels shape or length does not match features; ignoring it")
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

                    print(f"  - Feature shape: {features.shape}")
                    print(f"  - Feature dimension: {features.shape[1]}")
                    print(f"  - Number of cells: {features.shape[0]}")
                    if positions is not None:
                        print(f"  - Coordinates available: {positions.shape}")

                else:
                    print(f"  Warning: sample {sample_id} not found: {feature_file}")

            except Exception as e:
                print(f"  Error: Loading sample {sample_id}: {e}")

        print(f"Successfully loaded deep features for {len(self.deep_features)} samples")

    def extract_cell_features_from_cellvit(self, sample_id):
        """Extract a cell table with coordinates and simple geometry features."""
        print(f"Extracting features for sample {sample_id}...")

        sample_info = self.sample_data[sample_id]
        cellvit_df = sample_info.get('cellvit')

        # Prefer NPZ positions when available to guarantee alignment with features.
        deep = self.deep_features.get(sample_id) if hasattr(
            self, 'deep_features') else None
        if deep is not None and deep.get('positions') is not None:
            positions = deep['positions']
            n = positions.shape[0]
            print(f"  Using positions from the NPZ file as cell coordinates; aligned length: {n}")
            # If CellViT data is missing or misaligned, build a minimal table from positions only.
            cells_data = []
            for idx in range(n):
                x, y = float(positions[idx, 0]), float(positions[idx, 1])
                # Geometry cannot be recovered from positions alone; keep placeholders.
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
            print(f"  Warning: sample {sample_id} has no cell segmentation data and no positions; using spot centers")
            return self.create_spot_based_features(sample_info['adata'])

        # Parse geometry and extract centroid + simple shape statistics.
        cells_data = []

        print(f"  Parsing geometry for {len(cellvit_df)} cells...")

        for idx in tqdm(range(len(cellvit_df)), desc="  Parsing cell coordinates"):
            try:
                row = cellvit_df.iloc[idx]

                # Parse WKB geometry if available.
                if 'geometry' in cellvit_df.columns:
                    geom_bytes = row['geometry']
                    geom = wkb.loads(geom_bytes)

                    # Centroid
                    centroid = geom.centroid
                    cell_x, cell_y = centroid.x, centroid.y

                    # Area and perimeter
                    cell_area = geom.area
                    cell_perimeter = geom.length

                else:
                    # If the geometry column is missing, use placeholder values.
                    cell_x, cell_y = float(idx % 100), float(idx // 100)
                    cell_area = 100.0
                    cell_perimeter = 35.4

                # Shape compactness-like feature.
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
                if idx < 10:
                    print(f"    Warning: error while processing cell {idx}: {e}")
                # Fallback defaults.
                cells_data.append({
                    'cell_id': idx,
                    'x': float(idx % 100),
                    'y': float(idx // 100),
                    'area': 100.0,
                    'perimeter': 35.4,
                    'shape_feature': 1.0
                })

        if not cells_data:
            print(f"  Warning: failed to extract cell data; using spot centers")
            return self.create_spot_based_features(sample_info['adata'])

        cells_df = pd.DataFrame(cells_data)

        # Basic coordinate range logging for debugging.
        print(f"  Extracted features for {len(cells_df)} cells")
        print(
            f"   Cell coordinate range: X[{cells_df['x'].min():.1f}, {cells_df['x'].max():.1f}], Y[{cells_df['y'].min():.1f}, {cells_df['y'].max():.1f}]")

        return cells_df

    def create_spot_based_features(self, adata):
        """Fallback: create pseudo-cells at spot centers when segmentation is missing."""
        spots_coords = adata.obsm['spatial']

        cells_data = []
        for spot_idx, (x, y) in enumerate(spots_coords):
            cells_data.append({
                'cell_id': spot_idx,
                'x': float(x),
                'y': float(y),
                'area': 100.0,  # Placeholder area.
                'perimeter': 35.4,  # Placeholder perimeter (roughly circular).
                'shape_feature': 1.0  # Placeholder shape feature.
            })

        return pd.DataFrame(cells_data)

    def assign_cells_to_spots(self, sample_id, cells_df):
        """Assign cells to spots using either direct mapping (Xenium) or distance rules."""
        print(f"For sample {sample_id}, assigning cells to spots...")

        adata = self.sample_data[sample_id]['adata']
        metadata = self.sample_data[sample_id]['metadata']
        st_technology = metadata.get('st_technology', 'Unknown')

        # Xenium: treat each cell as a spot and map 1-to-1 (up to min length).
        if st_technology == 'Xenium':
            print(f"  Xenium technology detected; mapping cells directly to spots...")
            cells_df = cells_df.copy()

            # Spot coordinates.
            spots_coords = adata.obsm['spatial']
            print(f"  Number of cells: {len(cells_df)}, Number of spots: {len(spots_coords)}")

            # Direct 1-to-1 mapping (use the smaller count).
            min_count = min(len(cells_df), len(spots_coords))
            cells_df['spot_assignment'] = -1
            cells_df['distance_to_spot'] = 0.0

            # Assign first min_count cells to matching spot indices.
            cells_df.iloc[:min_count, cells_df.columns.get_loc(
                'spot_assignment')] = range(min_count)

            assigned_count = min_count
            print(f"  Successfully assigned: {assigned_count}/{len(cells_df)} cells (direct Xenium mapping)")

            return cells_df

        # Other technologies: distance-based matching.
        spots_coords = adata.obsm['spatial']

        # Coordinate range logging can be useful when matching fails.
        print(
            f"  Spot coordinate range: X[{spots_coords[:, 0].min():.1f}, {spots_coords[:, 0].max():.1f}], Y[{spots_coords[:, 1].min():.1f}, {spots_coords[:, 1].max():.1f}]")
        print(
            f"   Cell coordinate range: X[{cells_df['x'].min():.1f}, {cells_df['x'].max():.1f}], Y[{cells_df['y'].min():.1f}, {cells_df['y'].max():.1f}]")

        # Assign each cell to its nearest spot if within a radius threshold.
        cells_df = cells_df.copy()
        cells_df['spot_assignment'] = -1
        cells_df['distance_to_spot'] = float('inf')

        # Process all cells (batched to control memory).
        print(f"  Assigning {len(cells_df)} cells to {len(spots_coords)} spots...")

        assigned_count = 0
        distances_list = []

        # Batch processing to reduce peak memory.
        batch_size = 200000
        num_batches = (len(cells_df) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(cells_df))
            batch_cells = cells_df.iloc[start_idx:end_idx]

            print(
                f"  Processing batch {batch_idx+1}/{num_batches} ({len(batch_cells)} files)")

            for cell_idx, cell_row in tqdm(batch_cells.iterrows(), total=len(batch_cells), desc=f"  Batch {batch_idx+1}"):
                cell_pos = np.array([cell_row['x'], cell_row['y']])

                # Distances to all spots.
                distances = np.linalg.norm(spots_coords - cell_pos, axis=1)
                nearest_spot_idx = np.argmin(distances)
                nearest_distance = distances[nearest_spot_idx]

                distances_list.append(nearest_distance)

                # Convert pixel distance to microns.
                pixel_size_um = metadata.get(
                    'pixel_size_um_estimated', 0.5)  # Default: 0.5 um/pixel.
                distance_um = nearest_distance * pixel_size_um

                # Spot radius in microns (heuristic).
                spot_diameter_px = metadata.get('spot_diameter', None)

                if pd.isna(spot_diameter_px) or spot_diameter_px is None:
                    # Unknown technology: use a default radius.
                    spot_radius_um = 25.0  # Default radius: 25 um.
                else:
                    # Spot-based technologies such as Visium.
                    base_radius_um = (spot_diameter_px / 2.0) * \
                        pixel_size_um  # Base radius.
                    spot_radius_um = base_radius_um * 1.5  # Expand radius to improve matching.

                if distance_um <= spot_radius_um:
                    cells_df.at[cell_idx, 'spot_assignment'] = nearest_spot_idx
                    cells_df.at[cell_idx, 'distance_to_spot'] = distance_um
                    assigned_count += 1

        # Summary statistics.
        assigned_cells = cells_df[cells_df['spot_assignment'] >= 0]

        print(f"  Successfully assigned: {len(assigned_cells)}/{len(cells_df)} cells")
        pixel_size_um = metadata.get('pixel_size_um_estimated', 0.5)
        print(f"  Pixel size: {pixel_size_um} μm/pixel")

        if len(distances_list) > 0:
            distances_array = np.array(distances_list)
            print(
                f"  Distance statistics: min={distances_array.min():.1f}px,, max={distances_array.max():.1f}px,, mean={distances_array.mean():.1f}px")
            print(
                f"  Distance statistics (μm): min={distances_array.min()*pixel_size_um:.1f}μm,, max={distances_array.max()*pixel_size_um:.1f}μm,, mean={distances_array.mean()*pixel_size_um:.1f}μm")

        if len(assigned_cells) > 0:
            spot_counts = assigned_cells['spot_assignment'].value_counts()
            print(
                f"  Cells per spot: mean={spot_counts.mean():.1f},, range=[{spot_counts.min()}-{spot_counts.max()}]")
        else:
            print(f"  Warning: no cells were assigned to spots; the distance threshold may need adjustment")

        return cells_df

    def build_intra_spot_graphs(self, sample_id, intra_spot_k_neighbors=8):
        """Build cell-level graphs within each spot using kNN connectivity."""
        print(f"Building sample {sample_id} intra-spot graphs...")

        cells_df = self.processed_data[sample_id]['cells']
        # If deep features provide explicit spot grouping, prefer it to avoid misalignment.
        deep = self.deep_features.get(sample_id) if hasattr(
            self, 'deep_features') else None
        group_by_spot = None
        if deep is not None:
            spot_ptr = deep.get('spot_ptr')
            spot_index = deep.get('spot_index')
            if spot_ptr is not None and isinstance(spot_ptr, (np.ndarray, list)) and len(spot_ptr) >= 2:
                # Map spot_idx -> (start, end) using ptr.
                group_by_spot = {int(si): (int(spot_ptr[si]), int(
                    spot_ptr[si+1])) for si in range(len(spot_ptr)-1)}
            elif spot_index is not None and isinstance(spot_index, np.ndarray):
                # Group by explicit spot_index.
                group_by_spot = {}
                for i, si in enumerate(spot_index.tolist()):
                    si = int(si)
                    group_by_spot.setdefault(si, []).append(i)
        assigned_cells = cells_df[cells_df['spot_assignment'] >= 0]

        intra_spot_graphs = {}

        # Process spot by spot.
        for spot_idx in tqdm(assigned_cells['spot_assignment'].unique(), desc="Building intra-spot graphs"):
            spot_cells = assigned_cells[assigned_cells['spot_assignment'] == spot_idx].copy(
            )

            if len(spot_cells) < 2:
                # Spots with 0 or 1 cell.
                if len(spot_cells) == 1:
                    # Single-node graph.
                    cell_row = spot_cells.iloc[0]
                    cell_features = self.extract_cell_feature_vector(
                        cell_row, sample_id, cell_row.name)
                    x = torch.tensor([cell_features], dtype=torch.float32)
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    pos = torch.tensor(
                        [[cell_row['x'], cell_row['y']]], dtype=torch.float32)
                    graph = Data(x=x, edge_index=edge_index, pos=pos)
                    # Attach clustering labels when available.
                    if deep is not None and deep.get('cluster_labels') is not None:
                        cls = deep['cluster_labels']
                        if cell_row.name < len(cls):
                            graph.cluster_labels = torch.tensor([
                                int(cls[cell_row.name])
                            ], dtype=torch.long)
                    intra_spot_graphs[int(spot_idx)] = graph
                continue

            # Default: derive positions/features from the cell table.
            positions = spot_cells[['x', 'y']].values
            cell_features = np.array([
                self.extract_cell_feature_vector(row, sample_id, row.name)
                for _, row in spot_cells.iterrows()
            ])
            cluster_slice = None

            # Prefer deep features/positions sliced by spot grouping when available.
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

            # kNN connectivity within the spot.
            k = min(intra_spot_k_neighbors, len(spot_cells) - 1)

            if k > 0:
                nbrs = NearestNeighbors(n_neighbors=k+1).fit(positions)
                _, indices = nbrs.kneighbors(positions)

                edges = []
                for i, neighbors in enumerate(indices):
                    for neighbor in neighbors[1:]:
                        edges.extend([[i, neighbor], [neighbor, i]])

                if edges:
                    edge_index = torch.tensor(
                        np.array(edges).T, dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # Build graph.
            x = torch.tensor(cell_features, dtype=torch.float32)
            pos = torch.tensor(positions, dtype=torch.float32)

            graph = Data(x=x, edge_index=edge_index, pos=pos)
            # Attach clustering labels when available and aligned.
            if cluster_slice is not None:
                try:
                    cls_t = torch.as_tensor(cluster_slice, dtype=torch.long)
                    if cls_t.dim() == 1 and cls_t.numel() == x.shape[0]:
                        graph.cluster_labels = cls_t
                except Exception:
                    pass
            intra_spot_graphs[int(spot_idx)] = graph

        print(
            f"  Built {len(intra_spot_graphs)} intra-spot graphs, each using {intra_spot_k_neighbors}-NN connections")
        return intra_spot_graphs

    def extract_cell_feature_vector(self, cell_row, sample_id=None, cell_idx=None):
        """Extract a feature vector for a cell (deep features preferred)."""

        # Prefer deep features when available.
        if sample_id and sample_id in self.deep_features and cell_idx is not None:
            deep_features = self.deep_features[sample_id]['features']
            if cell_idx < len(deep_features):
                return deep_features[cell_idx].astype(np.float32)

        # Fallback to simple geometric features and pad/truncate to 128 dims.
        features = [
            cell_row['area'],
            cell_row['perimeter'],
            cell_row['shape_feature'],
            cell_row['x'],
            cell_row['y']
        ]

        features = np.array(features, dtype=np.float32)

        target_dim = 128
        if len(features) < target_dim:
            features = np.pad(
                features, (0, target_dim - len(features)), mode='constant')
        elif len(features) > target_dim:
            features = features[:target_dim]

        return features

    def build_inter_spot_graph(self, sample_id):
        """Build a spot-level kNN graph from spot coordinates."""
        print(f"Building sample {sample_id} inter-spot graph...")

        adata = self.sample_data[sample_id]['adata']

        # Spot coordinates.
        spot_positions = adata.obsm['spatial']

        # kNN connectivity.
        nbrs = NearestNeighbors(
            n_neighbors=self.inter_spot_k_neighbors+1).fit(spot_positions)
        _, indices = nbrs.kneighbors(spot_positions)

        # Build undirected edges.
        edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:
                edges.extend([[i, neighbor], [neighbor, i]])

        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)

        # Use positions as simple node features.
        spot_features = torch.tensor(spot_positions, dtype=torch.float32)
        pos = torch.tensor(spot_positions, dtype=torch.float32)

        inter_spot_graph = Data(
            x=spot_features, edge_index=edge_index, pos=pos)

        print(
            f"  Built inter-spot graph: {len(spot_positions)} spots, {edge_index.shape[1]} edges")
        return inter_spot_graph

    def process_all_samples(self):
        """Process all loaded samples and populate ``processed_data``."""
        print("=== Processing all samples ===")

        for sample_id in self.sample_data.keys():
            print(f"\nProcessing sample: {sample_id}")

            # Extract cell features.
            cells_df = self.extract_cell_features_from_cellvit(sample_id)

            # Assign cells to spots.
            cells_with_spots = self.assign_cells_to_spots(sample_id, cells_df)

            # Store processed outputs.
            self.processed_data[sample_id] = {
                'cells': cells_with_spots,
                'adata': self.sample_data[sample_id]['adata']
            }

            print(f"  Sample {sample_id} processed successfully")

    def save_graphs(self, output_dir):
        """Serialize constructed graphs and metadata to disk."""
        print("=== Saving graph structures ===")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_graphs = {}
        all_metadata = {}

        for sample_id in self.processed_data.keys():
            print(f"\nSaving sample {sample_id}...")

            # Build graphs.
            intra_spot_graphs = self.build_intra_spot_graphs(
                sample_id, intra_spot_k_neighbors=8)
            inter_spot_graph = self.build_inter_spot_graph(sample_id)

            # Save into the aggregated dict.
            all_graphs[sample_id] = {
                'intra_spot_graphs': intra_spot_graphs,
                'inter_spot_graph': inter_spot_graph
            }

            # Metadata.
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

        # Output file paths.
        intra_graphs_path = os.path.join(
            output_dir, "hest_intra_spot_graphs.pkl")
        inter_graphs_path = os.path.join(
            output_dir, "hest_inter_spot_graphs.pkl")
        metadata_path = os.path.join(output_dir, "hest_graph_metadata.json")
        processed_data_path = os.path.join(
            output_dir, "hest_processed_data.pkl")

        # Save intra-spot graphs.
        with open(intra_graphs_path, 'wb') as f:
            pickle.dump({sid: data['intra_spot_graphs']
                        for sid, data in all_graphs.items()}, f)

        # Save inter-spot graphs.
        with open(inter_graphs_path, 'wb') as f:
            pickle.dump({sid: data['inter_spot_graph']
                        for sid, data in all_graphs.items()}, f)

        # Save metadata.
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        # Save processed data.
        with open(processed_data_path, 'wb') as f:
            pickle.dump(self.processed_data, f)

        print(f"\nGraph data saved to: {output_dir}")
        print(f"- Intra-spot graphs: {intra_graphs_path}")
        print(f"- Inter-spot graph: {inter_graphs_path}")
        print(f"- Metadata: {metadata_path}")
        print(f"- Processed data: {processed_data_path}")

        return all_metadata


def main():
    """Entry point for graph construction."""

    # Configuration.
    hest_data_dir = "/data/yujk/hovernet2feature/HEST/hest_data"
    output_dir = "/data/yujk/hovernet2feature/hest_graphs_dinov3_other_cancer"

    # Optional deep feature directory.
    features_dir = "/data/yujk/hovernet2feature/hest_dinov3_other_cancer"

    # Sample selection:
    # - USE_SPECIFIED_SAMPLES=True: process SPECIFIED_SAMPLE_IDS (filtered by availability)
    # - USE_ALL_SAMPLES=True: process all available samples
    USE_SPECIFIED_SAMPLES = True
    SPECIFIED_SAMPLE_IDS = [
        "INT1","INT10","INT11","INT12","INT13","INT14","INT15","INT16","INT17","INT18",
        "INT19","INT2","INT20","INT21","INT22","INT23","INT24","INT3","INT4","INT5",
        "INT6","INT7","INT8","INT9",
        "TENX111","TENX147","TENX148","TENX149",
        "NCBI642","NCBI643",
        "NCBI783","NCBI785","TENX95","TENX99",
        "TENX118","TENX141",
        "NCBI681","NCBI682","NCBI683","NCBI684",
        "TENX116","TENX126","TENX140",
        "MEND139","MEND140","MEND141","MEND142","MEND143","MEND144","MEND145","MEND146",
        "MEND147","MEND148","MEND149","MEND150","MEND151","MEND152","MEND153","MEND154",
        "MEND156","MEND157","MEND158","MEND159","MEND160","MEND161","MEND162",
        "ZEN36","ZEN40","ZEN48","ZEN49",
        "TENX115","TENX117"
    ]
    USE_ALL_SAMPLES = True
    MAX_SAMPLES = None

    # Discover available samples from the ``st`` directory.
    print("=== Checking available samples ===")
    available_samples = []

    # Scan sample files under the st directory.
    st_dir = os.path.join(hest_data_dir, "st")
    if os.path.exists(st_dir):
        for file in os.listdir(st_dir):
            if file.endswith('.h5ad'):
                sample_id = file.replace('.h5ad', '')
                available_samples.append(sample_id)

    available_samples.sort()
    print(f"Total available samples found: {len(available_samples)}")
    print(f"Sample list: {available_samples}")

    # Choose samples based on the configuration flags above.
    if USE_SPECIFIED_SAMPLES is True:
        # Use the manually specified list.
        specified = SPECIFIED_SAMPLE_IDS or []
        sample_ids = [sid for sid in specified if sid in available_samples]
        missing = [sid for sid in specified if sid not in available_samples]
        if missing:
            print(f"Warning: requested samples not found will be ignored: {missing}")
        print(f"\nSelection mode: using the requested sample list")
        if MAX_SAMPLES is not None:
            sample_ids = sample_ids[:MAX_SAMPLES]
            print(f"Maximum sample limit: {MAX_SAMPLES}")
    elif USE_ALL_SAMPLES:
        # Use all available samples.
        sample_ids = available_samples
        if MAX_SAMPLES is not None:
            sample_ids = sample_ids[:MAX_SAMPLES]
        print(f"\nSelection mode: using all samples")
        print(f"Maximum sample limit: {MAX_SAMPLES if MAX_SAMPLES else 'unlimited'}")
    else:
        # Example heuristic: TENX samples are often colorectal cancer.
        preferred_samples = ['TENX128', 'TENX139',
                             'TENX147', 'TENX148', 'TENX149']
        sample_ids = [
            sid for sid in preferred_samples if sid in available_samples]

        if not sample_ids:
            # Fallback: use all TENX samples if available.
            tenx_samples = [
                sid for sid in available_samples if sid.startswith('TENX')]
            sample_ids = tenx_samples if tenx_samples else available_samples

        print(f"\nSelection mode: colorectal cancer samples only")

    print(f"Number of samples to process: {len(sample_ids)}")
    print(f"Sample list: {sample_ids}")

    if not sample_ids:
        print("Error: no available sample data found")
        return

    # Graph parameters.
    inter_spot_k_neighbors = 6  # Inter-spot kNN neighbors.

    print("\n=== HEST graph construction (direct file loading + deep features) ===")
    print(f"HESTData directory: {hest_data_dir}")
    print(f"Deep feature directory: {features_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample list: {sample_ids}")
    print(f"Configuration:")
    print(f"  - Inter-spot k-NN: {inter_spot_k_neighbors}")
    print(
        f"  - Use deep features: {os.path.exists(features_dir) if features_dir else False}")
    print(f"  - Feature dimension: 128 (deep features) or automatically expanded to 128")

    # Validate input directories.
    if not os.path.exists(hest_data_dir):
        print(f"Error: HEST data directory does not exist: {hest_data_dir}")
        return

    # Create the graph builder.
    try:
        builder = HESTDirectReader(
            hest_data_dir=hest_data_dir,
            sample_ids=sample_ids,
            features_dir=features_dir,  # Deep feature directory.
            inter_spot_k_neighbors=inter_spot_k_neighbors
        )

        # Load HEST data.
        builder.load_sample_data()

        if not builder.sample_data:
            print("Error: failed to load any HEST data")
            return

        # Process all samples.
        builder.process_all_samples()

        # Build and save graphs.
        metadata = builder.save_graphs(output_dir)

        print("\n=== Graph construction completed ===")
        for sample_id, meta in metadata.items():
            print(f"Sample {sample_id}:")
            print(f"  - {meta['intra_graph_count']} intra-spot graphs")
            print(f"  - 1 inter-spot graph ({meta['inter_graph_edges']} edges)")
            print(
                f"  - Total assigned cells: {meta['num_assigned_cells']}/{meta['num_cells']} assigned cells")
            print(f"  - Tissue type: {meta['tissue']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    # test

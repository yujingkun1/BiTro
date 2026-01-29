#!/usr/bin/env python3
"""
Xenium dataset utilities: load xenium_graphs and cell_feature_matrix.h5,
construct per-patch (virtual spot) graph -> per-node target pairs.
"""
import os
import pickle
import numpy as np
import h5py
import pandas as pd
from scipy import sparse as sp

def load_processed_data(processed_path):
    """Load processed_data.pkl produced by xenium_graph_construction"""
    with open(processed_path, 'rb') as f:
        return pickle.load(f)

def load_intra_graphs(intra_path):
    with open(intra_path, 'rb') as f:
        return pickle.load(f)

def find_features_in_h5(h5_path):
    """Find a 2D dataset in the h5 file that likely corresponds to cell features."""
    if h5_path is None or not os.path.exists(h5_path):
        return None, None
    with h5py.File(h5_path, 'r') as hf:
        for key in hf.keys():
            try:
                arr = hf[key]
                if hasattr(arr, 'shape') and len(arr.shape) == 2:
                    return key, arr.shape
            except Exception:
                continue
    return None, None

def load_cell_features(h5_path):
    """Load the first 2D dataset found in h5 as numpy array."""
    if h5_path is None:
        raise FileNotFoundError("features h5 path is None")
    with h5py.File(h5_path, 'r') as hf:
        # Case 1: AnnData-style sparse matrix under group 'matrix' (data, indices, indptr, shape)
        if 'matrix' in hf:
            grp = hf['matrix']
            if all(k in grp for k in ('data', 'indices', 'indptr', 'shape')):
                data = np.array(grp['data'])
                indices = np.array(grp['indices'])
                indptr = np.array(grp['indptr'])
                raw_shape = np.array(grp['shape']).astype(int).tolist()
                # Determine correct orientation: indptr length = n_rows + 1
                n_rows_from_indptr = int(len(indptr) - 1)
                # raw_shape may be [n_features, n_cells] or [n_cells, n_features]
                if raw_shape[0] == n_rows_from_indptr:
                    shape = (raw_shape[0], raw_shape[1])
                elif raw_shape[1] == n_rows_from_indptr:
                    # stored transposed, swap to (n_rows, n_cols)
                    shape = (raw_shape[1], raw_shape[0])
                else:
                    # fallback: trust indptr for rows and infer cols
                    shape = (n_rows_from_indptr, int(raw_shape[0] * raw_shape[1] / n_rows_from_indptr))
                try:
                    csr = sp.csr_matrix((data, indices, indptr), shape=shape)
                    return csr.toarray()
                except Exception:
                    # Fallback: manually construct dense
                    dense = np.zeros(shape, dtype=data.dtype)
                    for row in range(shape[0]):
                        start = indptr[row]
                        end = indptr[row+1]
                        cols = indices[start:end]
                        vals = data[start:end]
                        if len(cols):
                            dense[row, cols] = vals
                    return dense

        # Case 2: direct 2D datasets at root
        for key in hf.keys():
            try:
                arr = hf[key]
                if hasattr(arr, 'shape') and len(arr.shape) == 2:
                    return np.array(arr)
            except Exception:
                continue
    raise RuntimeError("No 2D dataset found in h5 file")

class XeniumGraphDataset:
    """
    Holds list of (graph, targets_tensor, global_cell_indices) entries.
    - graphs: a dict of sample_id -> {patch_id: Data}
    - processed_data: dict loaded from processed_data.pkl with cells DataFrame
    - features_h5: numpy array [N_cells, T] aligned with cells_df order
    """
    def __init__(self, graphs_pkl, processed_pkl, features_h5_path):
        self.intra = load_intra_graphs(graphs_pkl)
        self.processed = load_processed_data(processed_pkl)
        self.features_h5_path = features_h5_path

        # load features
        if features_h5_path and os.path.exists(features_h5_path):
            self.features = load_cell_features(features_h5_path)
        else:
            raise FileNotFoundError(f"Features h5 not found: {features_h5_path}")

        # Build list of entries (sample_id, patch_id, graph, targets, global_indices)
        self.entries = []
        for sid, graphs in self.intra.items():
            proc = self.processed.get(sid)
            if proc is None:
                continue
            cells_df = proc['cells']
            # recompute patch grouping same as graph constructor
            positions = cells_df[['x','y']].values
            xs = positions[:,0]; ys = positions[:,1]
            patch_size = 2000.0
            min_x, min_y = xs.min(), ys.min()
            gx = ((xs - min_x) // patch_size).astype(int)
            gy = ((ys - min_y) // patch_size).astype(int)
            patch_tuples = list(zip(gy.tolist(), gx.tolist()))
            unique_patches = {}
            patch_indices = []
            for t in patch_tuples:
                if t not in unique_patches:
                    unique_patches[t] = len(unique_patches)
                patch_indices.append(unique_patches[t])
            patch_indices = np.array(patch_indices, dtype=int)

            for pid, g in graphs.items():
                # find global indices for this patch
                idxs = np.where(patch_indices == int(pid))[0]
                if len(idxs) == 0:
                    continue
                # targets are features rows for these global indices
                targets = self.features[idxs]
                self.entries.append({
                    'sample_id': sid,
                    'patch_id': int(pid),
                    'graph': g,
                    'targets': targets.astype(np.float32),
                    'global_idxs': idxs
                })

    def __len__(self):
        return len(self.entries)

    def get_entries(self):
        return self.entries

#!/usr/bin/env python3
"""
Xenium single-cell dataset for training Cell2Gene model at cell level.
This dataset mirrors HESTSpatialDataset API but yields one graph per cell
(single-node PyG Data) and per-cell supervision loaded from cell_feature_matrix.h5.
"""
import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")


class XeniumCellDataset(Dataset):
    """
    Args:
        xenium_out_dir: path to /.../xenium/outs directory
        feature_h5_path: path to cell_feature_matrix.h5 (if None, will try xenium_out_dir)
        cells_csv: path to cells.csv (if None, will try xenium_out_dir)
        gene_file: optional intersection gene list (same format as spitial_model)
        mode: 'train' or 'test'
        apply_gene_normalization: whether to z-score genes within training set
    """
    def __init__(self, xenium_out_dir="/data/yujk/hovernet2feature/xenium/outs",
                 feature_h5_path=None, cells_csv=None, gene_file=None,
                 mode="train", apply_gene_normalization=True, normalization_stats=None, seed=42):
        self.xenium_out_dir = xenium_out_dir
        self.feature_h5_path = feature_h5_path or os.path.join(xenium_out_dir, "cell_feature_matrix.h5")
        self.cells_csv = cells_csv or os.path.join(xenium_out_dir, "cells.csv")
        self.gene_file = gene_file
        self.mode = mode
        self.apply_gene_normalization = apply_gene_normalization
        self._provided_normalization_stats = normalization_stats
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load cells table
        if not os.path.exists(self.cells_csv):
            raise FileNotFoundError(f"cells.csv not found: {self.cells_csv}")
        self.cells_df = pd.read_csv(self.cells_csv)
        if 'cell_id' not in self.cells_df.columns:
            # try Barcode/ID detection
            if 'Barcode' in self.cells_df.columns:
                self.cells_df['cell_id'] = self.cells_df['Barcode'].astype(int)
            else:
                self.cells_df['cell_id'] = np.arange(1, len(self.cells_df) + 1)

        # Load feature H5 (supervision)
        if not os.path.exists(self.feature_h5_path):
            raise FileNotFoundError(f"cell_feature_matrix.h5 not found: {self.feature_h5_path}")
        with h5py.File(self.feature_h5_path, 'r') as hf:
            # try common dataset names
            ds = None
            for key in ('data', 'matrix', 'X', 'features', 'expression'):
                if key in hf:
                    ds = hf[key]
                    break
            if ds is None:
                # pick first 2D dataset
                for k in hf.keys():
                    try:
                        tmp = hf[k]
                        if len(tmp.shape) == 2:
                            ds = tmp
                            break
                    except Exception:
                        continue
            if ds is None:
                raise RuntimeError("No 2D dataset found in feature h5")
            # read lazily into memory (could be large)
            self.features_np = np.array(ds)
            # try load gene names and barcodes if present
            self.h5_genes = None
            self.h5_barcodes = None
            for key in ('genes', 'var', 'gene_names'):
                if key in hf:
                    try:
                        self.h5_genes = np.array(hf[key]).astype(str).tolist()
                        break
                    except Exception:
                        continue
            for key in ('barcodes', 'cell_id', 'cell_index'):
                if key in hf:
                    try:
                        self.h5_barcodes = np.array(hf[key]).astype(str).tolist()
                        break
                    except Exception:
                        continue

        # Align features to cells.csv order
        # If h5 has barcodes matching cells_df.cell_id, map; else assume same order
        if self.h5_barcodes is not None:
            # try to match integers if possible
            try:
                h5_ids = np.array([int(x) for x in self.h5_barcodes])
                csv_ids = self.cells_df['cell_id'].astype(int).values
                # build index map: csv_order -> h5_index
                mapping = {cid: i for i, cid in enumerate(h5_ids)}
                idx_map = [mapping.get(int(cid), None) for cid in csv_ids]
                if any(v is None for v in idx_map):
                    # fallback to direct order
                    self.aligned_features = self.features_np
                else:
                    self.aligned_features = self.features_np[np.array(idx_map)]
            except Exception:
                self.aligned_features = self.features_np
        else:
            # assume same order
            if self.features_np.shape[0] != len(self.cells_df):
                # tolerate mismatch by taking min
                n_min = min(self.features_np.shape[0], len(self.cells_df))
                self.aligned_features = self.features_np[:n_min]
                self.cells_df = self.cells_df.iloc[:n_min].reset_index(drop=True)
            else:
                self.aligned_features = self.features_np

        # Gene selection
        if self.gene_file:
            with open(self.gene_file, 'r') as f:
                sel = [ln.strip() for ln in f if ln.strip()]
            if self.h5_genes is not None:
                # map file genes to h5 columns
                gene_idx = []
                h5_genes_lower = [g.lower() for g in self.h5_genes]
                for g in sel:
                    try:
                        gene_idx.append(h5_genes_lower.index(g.lower()))
                    except ValueError:
                        gene_idx.append(None)
                # build features subset (none -> zeros)
                cols = []
                for gi in gene_idx:
                    if gi is None:
                        cols.append(np.zeros((self.aligned_features.shape[0],), dtype=np.float32))
                    else:
                        cols.append(self.aligned_features[:, gi].astype(np.float32))
                self.targets = np.stack(cols, axis=1).astype(np.float32)
                self.selected_genes = sel
            else:
                # cannot map, fallback to full features but warn
                print("Warning: gene_file provided but h5 genes not found; using all features")
                self.targets = self.aligned_features.astype(np.float32)
                self.selected_genes = [f"gene_{i}" for i in range(self.targets.shape[1])]
        else:
            # use all features as targets
            self.targets = self.aligned_features.astype(np.float32)
            self.selected_genes = [f"gene_{i}" for i in range(self.targets.shape[1])]

        # normalization stats (computed from training set externally if needed)
        self.gene_mean = None
        self.gene_std = None
        if self._provided_normalization_stats is not None:
            self.gene_mean = np.asarray(self._provided_normalization_stats.get('mean'), dtype=np.float32)
            self.gene_std = np.asarray(self._provided_normalization_stats.get('std'), dtype=np.float32)

        # Prepare index list
        self.indices = np.arange(len(self.cells_df))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        row = self.cells_df.iloc[i]
        # node feature: use geometry features or zeros if not present
        if 'cell_area' in row.index:
            x_feat = np.array([row.get('cell_area', 100.0), row.get('nucleus_area', 35.4),
                               float(row.get('x_centroid', row.get('x', 0.0))),
                               float(row.get('y_centroid', row.get('y', 0.0)))], dtype=np.float32)
            # pad to 128
            if x_feat.shape[0] < 128:
                pad = np.zeros((128 - x_feat.shape[0],), dtype=np.float32)
                x_feat = np.concatenate([x_feat, pad], axis=0)
        else:
            # try positions
            x_feat = np.zeros((128,), dtype=np.float32)
            if 'x_centroid' in row.index:
                x_feat[3] = float(row['x_centroid'])
                x_feat[4] = float(row['y_centroid'])

        x_tensor = torch.as_tensor(x_feat, dtype=torch.float32)

        # target per-cell (gene vector)
        y = self.targets[i]
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        if self.apply_gene_normalization and self.gene_mean is not None and self.gene_std is not None:
            y_tensor = (y_tensor - torch.from_numpy(self.gene_mean)) / torch.from_numpy(self.gene_std)

        # Build single-node graph
        data = Data(x=x_tensor.unsqueeze(0), edge_index=torch.empty((2, 0), dtype=torch.long),
                    pos=torch.as_tensor([float(row.get('x_centroid', row.get('x', 0.0))),
                                         float(row.get('y_centroid', row.get('y', 0.0)))], dtype=torch.float32).unsqueeze(0))

        return {
            'spot_expressions': y_tensor,
            'spot_graphs': data,
            'sample_id': 'xenium',
            'spot_id': int(row['cell_id'])
        }



